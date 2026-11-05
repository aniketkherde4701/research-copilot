"""Everything that touches the outside world: arXiv search, PDF fetch, PDF parse.

PHASE 2 done (search_arxiv). PHASE 3 done (download_pdf, extract_text,
strip_references). The chunker itself lives in rag_pipeline.py.
"""

from __future__ import annotations

import logging
import re
import urllib.request
from pathlib import Path

import arxiv

try:                      # PyMuPDF renamed its module; `fitz` is deprecated
    import pymupdf
except ImportError:       # older PyMuPDF installs
    import fitz as pymupdf

from config import Settings
from models import Page, Paper

log = logging.getLogger(__name__)

# arXiv asks for no more than one request every three seconds. arxiv.Client
# enforces this for us -- which is the whole reason to use the Client rather
# than the deprecated module-level helpers.
ARXIV_DELAY_S = 3.0


class NoResultsError(RuntimeError):
    """arXiv returned nothing for any query."""


class PDFError(RuntimeError):
    """The PDF could not be fetched, opened, or read."""


def _to_paper(result: arxiv.Result) -> Paper:
    """arxiv.Result -> our Paper. Strips the version suffix from the ID so
    2103.00020v2 and 2103.00020v1 don't index as two different papers."""
    short_id = result.get_short_id().split("v")[0]
    pdf_url = result.pdf_url or result.entry_id.replace("/abs/", "/pdf/")
    return Paper(
        arxiv_id=short_id,
        # arXiv titles and abstracts come with hard line wraps in them
        title=" ".join(result.title.split()),
        authors=[a.name for a in result.authors],
        abstract=" ".join(result.summary.split()),
        published=result.published.date(),
        pdf_url=pdf_url,
        abs_url=result.entry_id,
    )


def search_arxiv(queries: list[str], *, max_results: int, per_query: int = 25) -> list[Paper]:
    """Run each query, then INTERLEAVE the result lists round-robin before
    truncating to max_results.

    Interleaving, not concatenating, is the point. The three queries are three
    different search STRATEGIES (precise jargon / plain mechanism / landmark
    names). Concatenating lets whichever query happens to return most results
    fill every slot, which throws away the diversity the rewrite step existed
    to create. Round-robin guarantees each strategy contributes.

    Dedupe keeps the earliest occurrence, so a paper found by the precise query
    keeps its higher position.
    """
    if not queries:
        raise ValueError("search_arxiv called with no queries")

    client = arxiv.Client(page_size=100, delay_seconds=ARXIV_DELAY_S, num_retries=3)

    per_query_results: list[list[Paper]] = []
    for query in queries:
        search = arxiv.Search(
            query=query,
            max_results=per_query,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        try:
            per_query_results.append([_to_paper(r) for r in client.results(search)])
        except Exception as exc:
            # One malformed query shouldn't sink the search -- the whole point
            # of generating several is that they're independent attempts.
            log.warning("arXiv query failed (%s): %s", query, exc)
            per_query_results.append([])

    found: dict[str, Paper] = {}
    for rank in range(max(len(r) for r in per_query_results) if per_query_results else 0):
        for results in per_query_results:
            if rank < len(results):
                found.setdefault(results[rank].arxiv_id, results[rank])
        if len(found) >= max_results:
            break

    if not found:
        raise NoResultsError(
            "arXiv returned no papers for any query. The topic may be outside "
            "arXiv's coverage (it is strongest in physics, maths and CS)."
        )
    return list(found.values())[:max_results]


# ---------------------------------------------------------------------
# Stage 3a -- fetch and parse
# ---------------------------------------------------------------------

# arXiv rejects the default urllib User-Agent.
_UA = "academic-research-copilot/0.1 (https://arxiv.org/help/api; contact: local user)"

# A heading line that is JUST "References" / "Bibliography", optionally numbered.
_REF_HEADING = re.compile(
    r"^\s*(?:\d+\.?\s*|[IVX]+\.?\s*)?(references|bibliography|works\s+cited)\s*$",
    re.IGNORECASE,
)


def download_pdf(paper: Paper, settings: Settings) -> Path:
    """Fetch the PDF into cache/, keyed by arxiv_id. Skip if already cached.

    The cache is not just a speed trick -- it means iterating on the chunker
    doesn't re-download the same paper fifty times, and it makes the second
    demo run visibly fast.
    """
    settings.cache_dir.mkdir(exist_ok=True)
    path = settings.cache_dir / f"{paper.arxiv_id.replace('/', '_')}.pdf"

    if path.exists() and path.stat().st_size > 1024:
        log.info("PDF cache hit: %s", path.name)
        return path

    log.info("Downloading %s", paper.pdf_url)
    request = urllib.request.Request(paper.pdf_url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(request, timeout=settings.request_timeout_s) as response:
            data = response.read()
    except Exception as exc:
        raise PDFError(f"Could not download {paper.pdf_url}: {exc}") from exc

    if not data.startswith(b"%PDF"):
        # arXiv sometimes serves an HTML holding page instead of the PDF.
        raise PDFError(f"{paper.pdf_url} did not return a PDF (got {data[:40]!r})")

    path.write_bytes(data)
    return path


def extract_text(pdf_path: Path) -> list[Page]:
    """Parse a PDF into per-page text using PyMuPDF (fitz).

    PyMuPDF rather than pypdf because academic PDFs are two-column and pypdf
    interleaves the columns into nonsense. sort=True asks PyMuPDF for reading
    order rather than raw draw order, which matters for the same reason.

    Page numbers are preserved because they become part of every citation.
    """
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as exc:
        raise PDFError(f"Could not open {pdf_path.name}: {exc}") from exc

    pages: list[Page] = []
    with doc:
        for i, page in enumerate(doc):
            text = page.get_text("text", sort=True)
            # Rejoin words broken across a line by hyphenation
            text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
            pages.append(Page(number=i + 1, text=text))

    if not any(p.text.strip() for p in pages):
        raise PDFError(
            f"{pdf_path.name} has no extractable text -- it is probably a scan. "
            f"OCR would be needed, which is out of scope."
        )
    return pages


def strip_references(pages: list[Page]) -> list[Page]:
    """Drop everything from the References/Bibliography heading onwards.

    Non-optional. A bibliography is hundreds of title-like fragments that match
    almost any query, so leaving it in means retrieval returns reference entries
    instead of the paper's own argument.

    Takes the FIRST heading in the back half of the document. Taking the last
    match would keep the reference list whenever an appendix follows it; taking
    the first overall would truncate a paper that says "references" early on.
    """
    if not pages:
        return pages

    half = len(pages) // 2
    for page_index in range(half, len(pages)):
        lines = pages[page_index].text.split("\n")
        for line_index, line in enumerate(lines):
            if _REF_HEADING.match(line):
                kept_text = "\n".join(lines[:line_index])
                kept = pages[:page_index]
                if kept_text.strip():
                    kept.append(Page(number=pages[page_index].number, text=kept_text))
                log.info(
                    "Stripped references from page %d (%d of %d pages kept)",
                    pages[page_index].number, len(kept), len(pages),
                )
                return kept

    log.info("No references heading found; keeping all %d pages", len(pages))
    return pages
