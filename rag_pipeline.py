"""Text -> chunks -> vectors -> retrieved evidence.

Semantic chunking, Pinecone indexing and retrieval. The chunker is the piece
that most affects answer quality: it splits where the meaning shifts rather
than every N characters, so retrieved passages stand on their own.
"""

from __future__ import annotations

import logging
import re
import time

import numpy as np
from pinecone import Pinecone, ServerlessSpec

from config import Settings
from models import Chunk, Page, Paper
from llm import TASK_DOCUMENT, TASK_QUERY, GeminiClient


log = logging.getLogger(__name__)

# Crude but adequate. A real tokenizer is a dependency we don't need: we only
# use token counts to decide where to stop merging, so being 10% off is fine.
TOKENS_PER_WORD = 1.3

# A line that looks like a section heading: "3 Methods", "4.1 Results",
# "Introduction". Used to tag chunks so citations can say where they came from.
_HEADING = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?"
    r"(abstract|introduction|related work|background|method(?:s|ology)?|"
    r"approach|experiment(?:s|al setup)?|result(?:s)?|evaluation|discussion|"
    r"analysis|ablation|conclusion(?:s)?|limitations|future work)\s*$",
    re.IGNORECASE,
)


def count_tokens(text: str) -> int:
    return int(len(text.split()) * TOKENS_PER_WORD)


def split_sentences(text: str) -> list[str]:
    """Crude sentence splitter. Good enough for v1 -- academic PDFs produce
    messy text whatever you do, and a proper NLP splitter is a heavy dependency
    for a marginal gain here."""
    parts = re.split(r"(?<=[.!?])\s+", text.replace("\n", " "))
    return [s.strip() for s in parts if s.strip()]


def detect_section(line: str, current: str) -> str:
    """Return the new section name if this line is a heading, else keep current."""
    match = _HEADING.match(line.strip())
    return match.group(1).title() if match else current


def semantic_chunk(pages: list[Page], client: GeminiClient, settings: Settings) -> list[Chunk]:
    """Split the paper where the MEANING shifts, not every N characters.

    Why this beats fixed-size splitting: fixed chunks cut sentences and ideas in
    half, so retrieval returns fragments that don't stand alone. Semantic chunks
    break where the topic actually changes.

    """
    # ---- 1. flatten pages into sentences, remembering page + section --------
    sentences: list[str] = []
    meta: list[tuple[int, str]] = []          # (page_number, section) per sentence
    section = "unknown"
    for page in pages:
        for line in page.text.split("\n"):
            section = detect_section(line, section)
        for sentence in split_sentences(page.text):
            sentences.append(sentence)
            meta.append((page.number, section))

    if not sentences:
        return []

    # ---- 1b. group sentences into WINDOWS before embedding ------------------
    # gemini-embedding-001 takes one input per request, so embedding every
    # sentence in a 600-sentence paper means 600 API calls -- straight through
    # the free tier's 100/min ceiling. Instead we embed windows of sentences and
    # look for topic shifts BETWEEN windows. Boundaries land on window edges
    # rather than any sentence edge: coarser, but still semantic, and it keeps a
    # paper inside one minute's quota.
    window = max(1, -(-len(sentences) // settings.max_boundary_units))  # ceil div
    windows = [sentences[i:i + window] for i in range(0, len(sentences), window)]
    window_texts = [" ".join(w) for w in windows]
    if window > 1:
        log.info("%d sentences -> %d windows of %d (free-tier embed budget)",
                 len(sentences), len(windows), window)

    if len(window_texts) == 1:
        text = window_texts[0]
        return [Chunk(id="0", text=text, section=meta[0][1],
                      page=meta[0][0], token_count=count_tokens(text))]

    # ---- 2. embed every sentence -------------------------------------------
    # TASK_DOCUMENT, not TASK_QUERY: these are passages to be found, and the
    # query that finds them is embedded differently on purpose.
    vectors = np.array(client.embed(window_texts, task_type=TASK_DOCUMENT))

    # ---- 3. distance between CONSECUTIVE sentences --------------------------
    # Two sentences about the same thing point in similar directions in
    # embedding space, so the distance between them is small. When the author
    # changes topic, the next sentence points elsewhere and the distance spikes.
    # Those spikes are the chunk boundaries.
    normed = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarities = np.sum(normed[:-1] * normed[1:], axis=1)
    distances = (1.0 - similarities).tolist()

    # ---- 4. threshold: cut only at UNUSUALLY large jumps --------------------
    # A percentile, not a fixed number. What counts as a "big jump" differs
    # between a terse maths paper and a discursive survey -- a percentile
    # adapts to the document, a hard-coded 0.3 does not.
    threshold = float(np.percentile(distances, settings.chunk_breakpoint_percentile))

    # ---- 5. group sentences into chunks ------------------------------------
    groups: list[list[int]] = []          # indices into `windows`
    current: list[int] = [0]
    for i in range(len(windows) - 1):
        current_text = " ".join(window_texts[j] for j in current)
        too_big = count_tokens(current_text) >= settings.chunk_target_tokens
        topic_changed = distances[i] > threshold
        if topic_changed or too_big:
            groups.append(current)
            current = [i + 1]
        else:
            current.append(i + 1)
    groups.append(current)

    # ---- 6. build Chunks, enforce the guardrails ---------------------------
    # window index -> index of its first sentence, for page/section lookup
    first_sentence_of = {w: w * window for w in range(len(windows))}

    chunks: list[Chunk] = []
    for group in groups:
        text = " ".join(window_texts[j] for j in group)
        tokens = count_tokens(text)
        if tokens < settings.chunk_min_tokens:
            continue                                  # too small to be useful
        while count_tokens(text) > settings.chunk_max_tokens:
            text = " ".join(text.split()[:-20])       # hard cap
        page, section = meta[min(first_sentence_of[group[0]], len(meta) - 1)]
        # Reuse the vectors we already paid for rather than embedding again.
        mean_vector = np.mean(vectors[group[0]:group[-1] + 1], axis=0)
        chunks.append(Chunk(
            id=str(len(chunks)), text=text, section=section,
            page=page, token_count=count_tokens(text),
            vector=mean_vector.tolist(),
        ))
        if len(chunks) >= settings.max_chunks_per_paper:
            log.warning("Hit max_chunks_per_paper (%d); truncating",
                        settings.max_chunks_per_paper)
            break

    log.info("%d sentences -> %d chunks", len(sentences), len(chunks))
    return chunks


# ---------------------------------------------------------------------
# Stage 4 -- Pinecone: index and retrieve
# ---------------------------------------------------------------------

UPSERT_BATCH_SIZE = 100          # Pinecone caps payload size, not just count
INDEX_READY_TIMEOUT_S = 60


def _client(settings: Settings) -> Pinecone:
    return Pinecone(api_key=settings.pinecone_api_key)


def _index_ready(pc: Pinecone, name: str) -> bool:
    """describe_index returns an IndexModel across SDK versions -- sometimes
    subscriptable, sometimes attribute-only. Accept either."""
    try:
        described = pc.describe_index(name)
    except Exception:
        return False
    status = getattr(described, "status", None)
    if status is None:
        try:
            status = described["status"]
        except Exception:
            return False
    if isinstance(status, dict):
        return bool(status.get("ready"))
    return bool(getattr(status, "ready", False))


def ensure_index(settings: Settings) -> None:
    """Create the serverless index if it doesn't exist, and wait until ready.

    Doing this in code rather than in the Pinecone console is what lets a fresh
    clone of the repo actually run.
    """
    pc = _client(settings)
    try:
        existing = set(pc.list_indexes().names())
    except AttributeError:
        existing = {getattr(i, "name", None) or i["name"] for i in pc.list_indexes()}
    if settings.pinecone_index_name in existing:
        return

    log.info("Creating Pinecone index %s", settings.pinecone_index_name)
    pc.create_index(
        name=settings.pinecone_index_name,
        dimension=settings.embedding_dim,
        metric="cosine",
        spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
    )
    deadline = time.monotonic() + INDEX_READY_TIMEOUT_S
    while time.monotonic() < deadline:
        if _index_ready(pc, settings.pinecone_index_name):
            return
        time.sleep(2)
    raise RuntimeError(
        f"Index {settings.pinecone_index_name} was not ready within "
        f"{INDEX_READY_TIMEOUT_S}s. Check the Pinecone console."
    )


def is_indexed(paper: Paper, settings: Settings) -> bool:
    """True if this paper's namespace already holds vectors.

    This is the cache that turns a 60-second second run into a 5-second one,
    and the reason Pinecone earns its place here rather than being pure
    overhead: the corpus grows instead of being thrown away each query.
    """
    try:
        stats = _client(settings).Index(settings.pinecone_index_name).describe_index_stats()
    except Exception as exc:
        log.warning("Could not read index stats: %s", exc)
        return False
    namespace = (stats.get("namespaces") or {}).get(paper.namespace)
    return bool(namespace and namespace.get("vector_count", 0) > 0)


def index_paper(paper: Paper, chunks: list[Chunk], client: GeminiClient, settings: Settings) -> str:
    """Embed chunks and upsert them into this paper's own namespace."""
    if not chunks:
        raise ValueError(f"No chunks to index for {paper.arxiv_id}")

    ensure_index(settings)
    index = _client(settings).Index(settings.pinecone_index_name)

    # Chunks normally arrive with a vector already, reused from boundary
    # detection -- that halves the API calls, which matters a great deal on a
    # free tier where each embedding is its own request.
    missing = [c for c in chunks if c.vector is None]
    if missing:
        log.info("Embedding %d chunks that lack a reused vector", len(missing))
        fresh = client.embed([c.text for c in missing], task_type=TASK_DOCUMENT)
        for chunk, vector in zip(missing, fresh):
            chunk.vector = vector

    records = [
        {"id": f"{paper.arxiv_id}::{c.id}", "values": c.vector, "metadata": c.to_metadata()}
        for c in chunks
    ]

    for start in range(0, len(records), UPSERT_BATCH_SIZE):
        index.upsert(vectors=records[start : start + UPSERT_BATCH_SIZE],
                     namespace=paper.namespace)

    log.info("Indexed %d chunks into namespace %s", len(records), paper.namespace)
    return paper.namespace


def retrieve(namespace: str, question: str, client: GeminiClient, settings: Settings) -> list[Chunk]:
    """Embed the question and pull the top-k most relevant chunks."""
    # TASK_QUERY, deliberately different from the TASK_DOCUMENT used at index
    # time. Asymmetric embedding is a real accuracy win for retrieval.
    vector = client.embed_one(question, task_type=TASK_QUERY)

    index = _client(settings).Index(settings.pinecone_index_name)
    response = index.query(
        vector=vector,
        top_k=settings.retrieval_top_k,
        namespace=namespace,
        include_metadata=True,
    )

    chunks: list[Chunk] = []
    for i, match in enumerate(response.get("matches", [])):
        meta = match.get("metadata") or {}
        text = meta.get("text", "")
        chunks.append(Chunk(
            id=str(i + 1),                       # 1-based: these become citation numbers
            text=text,
            section=meta.get("section", "unknown"),
            page=int(meta.get("page", 0)),
            token_count=count_tokens(text),
            score=match.get("score"),
        ))
    log.info("Retrieved %d chunks from %s", len(chunks), namespace)
    return chunks
