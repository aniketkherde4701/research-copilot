"""The orchestrator: five stages, timed, with the UI notified as each begins.

PHASE 2 done (rewrite_query, select_paper). PHASE 5 stub at the bottom.
app.py talks to this module and nothing else.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from typing import Callable

from config import Settings, load_prompt
from llm import GeminiClient
from datetime import date

from models import Chunk, Citation, Paper, Report
from rag_pipeline import (
    index_paper,
    is_indexed,
    retrieve,
    semantic_chunk,
)
from tools import (
    NoResultsError,
    download_pdf,
    extract_text,
    search_arxiv,
    strip_references,
)

log = logging.getLogger(__name__)

StageCallback = Callable[[str, str], None]   # (stage_key, human_readable_message)

STAGES = [
    ("search", "Searching arXiv"),
    ("select", "Choosing the best paper"),
    ("ingest", "Reading the paper"),
    ("retrieve", "Finding relevant sections"),
    ("synthesize", "Writing the report"),
]

MAX_QUERIES = 3
MIN_CONFIDENCE = 0.4          # below this, the model is telling us nothing fits well
ABSTRACT_PREVIEW_CHARS = 700  # enough to judge relevance, short enough to keep triage cheap

REWRITE_SCHEMA = {
    "type": "object",
    "properties": {"queries": {"type": "array", "items": {"type": "string"}}},
    "required": ["queries"],
}

SELECT_SCHEMA = {
    "type": "object",
    "properties": {
        "chosen_index": {"type": "integer"},
        "reason": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["chosen_index", "reason", "confidence"],
}


# ---------------------------------------------------------------------
# Question -> paper cache
# ---------------------------------------------------------------------
# arXiv's relevance ranking shifts as new papers are indexed, so the same
# question can yield a different candidate set -- and therefore a different
# paper -- on a later run. That makes the system non-reproducible AND means
# the per-paper vector cache can never hit, because there is never the same
# paper twice. Remembering which paper a question resolved to fixes both, and
# skips the search and selection stages entirely on a repeat.

def _question_key(question: str) -> str:
    normalised = " ".join(question.lower().split())
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


def _cache_path(settings: Settings):
    return settings.cache_dir / "questions.json"


def _read_question_cache(settings: Settings) -> dict:
    path = _cache_path(settings)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception as exc:
        log.warning("Question cache unreadable, ignoring: %s", exc)
        return {}


def _remember_question(settings: Settings, question: str, paper: Paper,
                       reason: str, confidence: float) -> None:
    cache = _read_question_cache(settings)
    cache[_question_key(question)] = {
        "question": question,
        "reason": reason,
        "confidence": confidence,
        "paper": {
            "arxiv_id": paper.arxiv_id, "title": paper.title,
            "authors": paper.authors, "abstract": paper.abstract,
            "published": paper.published.isoformat(),
            "pdf_url": paper.pdf_url, "abs_url": paper.abs_url,
        },
    }
    try:
        _cache_path(settings).write_text(json.dumps(cache, indent=2))
    except Exception as exc:
        log.warning("Could not write question cache: %s", exc)


def _recall_question(settings: Settings, question: str):
    """Returns (paper, reason, confidence) or None."""
    entry = _read_question_cache(settings).get(_question_key(question))
    if not entry:
        return None
    try:
        d = entry["paper"]
        paper = Paper(
            arxiv_id=d["arxiv_id"], title=d["title"], authors=d["authors"],
            abstract=d["abstract"], published=date.fromisoformat(d["published"]),
            pdf_url=d["pdf_url"], abs_url=d["abs_url"],
        )
    except Exception as exc:
        log.warning("Malformed question-cache entry, ignoring: %s", exc)
        return None
    return paper, entry.get("reason", ""), entry.get("confidence", 0.0)


# ---------------------------------------------------------------------
# Stage 1 -- question -> arXiv queries
# ---------------------------------------------------------------------

def rewrite_query(question: str, client: GeminiClient, settings: Settings) -> list[str]:
    """Turn a natural-language question into 2-3 arXiv-shaped queries.

    Not in the original brief, but arXiv's API is keyword-oriented and
    retrieves plain questions badly. This single step is most of the
    difference between a demo and something that works.

    Never raises: on any failure it falls back to the raw question, which is
    a worse search but still a search.
    """
    prompt = load_prompt("rewrite_query").format(question=question.strip())
    try:
        # temperature=0 deliberately. At 0.3 the same question produced
        # different queries -- and therefore a DIFFERENT PAPER -- on every run.
        # A research tool that isn't reproducible isn't trustworthy.
        data = client.generate(
            prompt, model=settings.model_triage, schema=REWRITE_SCHEMA, temperature=0.0
        )
        queries = [
            q.strip() for q in data.get("queries", [])
            if isinstance(q, str) and q.strip()
        ]
    except Exception as exc:
        log.warning("Query rewrite failed, falling back to raw question: %s", exc)
        queries = []

    if not queries:
        queries = [f'all:"{question.strip()}"']
    return queries[:MAX_QUERIES]


# ---------------------------------------------------------------------
# Stage 2 -- candidates -> one paper
# ---------------------------------------------------------------------

def _format_candidates(papers: list[Paper]) -> str:
    lines = []
    for i, p in enumerate(papers):
        abstract = p.abstract[:ABSTRACT_PREVIEW_CHARS]
        if len(p.abstract) > ABSTRACT_PREVIEW_CHARS:
            abstract += "..."
        lines.append(
            f"[{i}] {p.title} ({p.published.year})\n"
            f"    {abstract}"
        )
    return "\n\n".join(lines)


def select_paper(
    question: str,
    candidates: list[Paper],
    client: GeminiClient,
    settings: Settings,
) -> tuple[Paper, str, float]:
    """Pick the ONE paper whose contents best answer the question.

    Returns (paper, reason, confidence).

    Never raises on a bad model response -- an out-of-range index, malformed
    JSON or an API error all fall back to arXiv's own top relevance hit. A
    triage step that can crash the whole pipeline is worse than one that
    occasionally picks second-best.
    """
    if not candidates:
        raise NoResultsError("select_paper called with no candidates")
    if len(candidates) == 1:
        return candidates[0], "Only one candidate was found.", 1.0

    prompt = load_prompt("select_paper").format(
        question=question.strip(),
        candidates=_format_candidates(candidates),
    )

    try:
        data = client.generate(
            prompt, model=settings.model_triage, schema=SELECT_SCHEMA, temperature=0.0
        )
        index = int(data["chosen_index"])
        reason = str(data.get("reason", "")).strip()
        confidence = float(data.get("confidence", 0.0))
    except Exception as exc:
        log.warning("Paper selection failed, using top arXiv hit: %s", exc)
        return candidates[0], "Selection step failed; using arXiv's top relevance hit.", 0.0

    if not 0 <= index < len(candidates):
        log.warning("Model returned out-of-range index %s; using top arXiv hit", index)
        return candidates[0], "Model returned an invalid choice; using arXiv's top hit.", 0.0

    if confidence < MIN_CONFIDENCE:
        # Keep the model's choice but carry the doubt forward -- the UI shows
        # this, and synthesis will likely hit the "not covered" path anyway.
        reason = f"Low confidence ({confidence:.2f}) -- no candidate fits well. {reason}"

    return candidates[index], reason, confidence


def find_paper(
    question: str,
    client: GeminiClient,
    settings: Settings,
    *,
    on_stage: StageCallback | None = None,
) -> tuple[Paper, str, float, list[Paper]]:
    """Stages 1-2 together: question -> (paper, reason, confidence, shortlist)."""
    def notify(key: str, msg: str) -> None:
        if on_stage:
            on_stage(key, msg)

    notify("search", "Searching arXiv")
    queries = rewrite_query(question, client, settings)
    log.info("arXiv queries: %s", queries)
    candidates = search_arxiv(
        queries,
        max_results=settings.arxiv_candidates,
        per_query=settings.arxiv_per_query,
    )

    notify("select", f"Choosing from {len(candidates)} candidates")
    paper, reason, confidence = select_paper(question, candidates, client, settings)
    shortlist = [p for p in candidates if p.arxiv_id != paper.arxiv_id]
    return paper, reason, confidence, shortlist


# ---------------------------------------------------------------------
# Stage 5 -- chunks -> cited report
# ---------------------------------------------------------------------

_CITATION = re.compile(r"\[(\d+)\]")


def _format_excerpts(chunks: list[Chunk]) -> str:
    return "\n\n".join(
        f"[{c.id}] (section: {c.section}, page {c.page})\n{c.text}" for c in chunks
    )


def synthesize(
    question: str,
    paper: Paper,
    chunks: list[Chunk],
    client: GeminiClient,
    settings: Settings,
) -> tuple[str, list[Citation]]:
    """Write the report from the retrieved excerpts, and collect its citations."""
    if not chunks:
        return (
            "No relevant passages were found in this paper, so there is nothing "
            "to base an answer on.",
            [],
        )

    prompt = load_prompt("synthesize").format(
        title=paper.title,
        authors=paper.short_authors(),
        question=question.strip(),
        excerpts=_format_excerpts(chunks),
    )
    answer = client.generate(prompt, model=settings.model_synthesis, temperature=0.2)

    # Only report citations the answer actually used, in first-appearance order.
    by_id = {c.id: c for c in chunks}
    citations: list[Citation] = []
    for match in _CITATION.finditer(answer):
        cid = match.group(1)
        chunk = by_id.get(cid)
        if chunk and not any(c.number == int(cid) for c in citations):
            citations.append(Citation(
                number=int(cid), section=chunk.section, page=chunk.page,
                excerpt=chunk.text[:300] + ("..." if len(chunk.text) > 300 else ""),
            ))
    citations.sort(key=lambda c: c.number)
    return answer.strip(), citations


def run_pipeline(
    question: str,
    settings: Settings,
    *,
    on_stage: StageCallback | None = None,
    force_search: bool = False,
) -> Report:
    """Full loop: question in, cited Report out.

    force_search=True ignores the remembered question->paper mapping and
    re-searches arXiv, for when you want a fresh look at a question.
    """
    question = question.strip()
    if not question:
        raise ValueError("Empty question")

    client = GeminiClient(settings)
    timings: dict[str, float] = {}

    def notify(key: str, msg: str) -> None:
        if on_stage:
            on_stage(key, msg)

    def timed(key: str, fn):
        start = time.monotonic()
        try:
            return fn()
        finally:
            timings[key] = time.monotonic() - start

    # -- stages 1-2: find the paper ---------------------------------------
    candidates: list[Paper] = []
    remembered = None if force_search else _recall_question(settings, question)

    if remembered is not None:
        paper, reason, confidence = remembered
        notify("search", "Recognised this question - reusing the paper it resolved to")
        notify("select", f"Selected: {paper.title}")
        timings["search"] = 0.0
        timings["select"] = 0.0
        log.info("Question cache hit -> %s", paper.arxiv_id)
    else:
        notify("search", "Searching arXiv")
        queries = timed("search", lambda: rewrite_query(question, client, settings))
        log.info("arXiv queries: %s", queries)
        candidates = timed("search", lambda: search_arxiv(
            queries, max_results=settings.arxiv_candidates,
            per_query=settings.arxiv_per_query))

        notify("select", f"Choosing from {len(candidates)} candidates")
        paper, reason, confidence = timed(
            "select", lambda: select_paper(question, candidates, client, settings))
        notify("select", f"Selected: {paper.title}")
        _remember_question(settings, question, paper, reason, confidence)

    # -- stage 3: read it (or skip, if we already have) --------------------
    cache_hit = is_indexed(paper, settings)
    if cache_hit:
        notify("ingest", "Already indexed - skipping download and chunking")
        log.info("Cache hit for %s; skipping ingest", paper.arxiv_id)
        timings["ingest"] = 0.0
    else:
        notify("ingest", "Downloading the PDF")
        pages = timed("ingest", lambda: strip_references(
            extract_text(download_pdf(paper, settings))))
        notify("ingest", f"Read {len(pages)} pages - now finding topic boundaries "
                         f"(embedding is rate-limited on the free tier, ~30s)")
        chunks = timed("ingest", lambda: semantic_chunk(pages, client, settings))
        notify("ingest", f"Split into {len(chunks)} passages - storing them")
        timed("ingest", lambda: index_paper(paper, chunks, client, settings))

    # -- stage 4: retrieve --------------------------------------------------
    notify("retrieve", "Finding the relevant sections")
    retrieved = timed("retrieve", lambda: retrieve(
        paper.namespace, question, client, settings))

    # -- stage 5: write -----------------------------------------------------
    notify("synthesize", "Writing the report")
    answer, citations = timed(
        "synthesize", lambda: synthesize(question, paper, retrieved, client, settings))

    return Report(
        question=question, paper=paper, answer=answer, citations=citations,
        chunks_used=retrieved, stage_timings=timings, selection_reason=reason,
        shortlist=[p for p in candidates if p.arxiv_id != paper.arxiv_id],
        cache_hit=cache_hit,
    )


# ---------------------------------------------------------------------
# CLI smoke test:  python agent_core.py "your question here"
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="  %(levelname)s  %(message)s")
    # arxiv/httpx log every request at INFO -- useful once, noise thereafter
    for noisy in ("httpx", "arxiv", "google_genai", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    from config import load_settings

    question = " ".join(sys.argv[1:]) or "How does contrastive pre-training affect zero-shot image classification?"
    settings = load_settings()
    client = GeminiClient(settings)

    print(f"\nQUESTION: {question}\n")
    paper, reason, confidence, shortlist = find_paper(question, client, settings)

    print(f"\nCHOSEN     {paper.title}")
    print(f"  authors  {paper.short_authors()} ({paper.published.year})")
    print(f"  url      {paper.abs_url}")
    print(f"  why      {reason}")
    print(f"  conf     {confidence:.2f}")
    print(f"\nRUNNERS-UP ({len(shortlist)})")
    for p in shortlist[:5]:
        print(f"  - {p.title[:78]}")
    print()
