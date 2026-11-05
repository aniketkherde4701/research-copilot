"""Shared data types. Small dataclasses, not loose dicts.

Passing typed objects between modules is what stops this becoming a
notebook -- and it makes every function signature self-documenting.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date


@dataclass(frozen=True)
class Paper:
    """One arXiv paper, as returned by search. No PDF content yet."""
    arxiv_id: str          # e.g. "2103.00020" -- also used as the Pinecone namespace
    title: str
    authors: list[str]
    abstract: str
    published: date
    pdf_url: str
    abs_url: str           # the human-readable /abs/ page -- this is what we cite

    @property
    def namespace(self) -> str:
        """Pinecone namespaces can't contain dots or slashes."""
        return self.arxiv_id.replace(".", "-").replace("/", "_")

    def short_authors(self, limit: int = 3) -> str:
        if len(self.authors) <= limit:
            return ", ".join(self.authors)
        return ", ".join(self.authors[:limit]) + " et al."


@dataclass
class Page:
    """One page of extracted PDF text, kept separate so citations can name a page."""
    number: int            # 1-indexed
    text: str


@dataclass
class Chunk:
    """A semantically coherent passage, ready to embed."""
    id: str                # f"{arxiv_id}::{index}"
    text: str
    section: str           # e.g. "Methods" -- "unknown" if heading detection missed
    page: int
    token_count: int
    score: float | None = None   # populated on retrieval, None at index time
    vector: list[float] | None = None
    """Embedding reused from boundary detection, so indexing doesn't re-embed.
    Mean of this chunk's window vectors -- an approximation of embedding the
    concatenated text, and the difference between 25 and 65 API calls per paper."""

    def to_metadata(self) -> dict:
        """What gets stored alongside the vector in Pinecone."""
        return {
            "text": self.text,
            "section": self.section,
            "page": self.page,
        }


@dataclass
class Citation:
    """One numbered source in the final report."""
    number: int
    section: str
    page: int
    excerpt: str


@dataclass
class Report:
    """The finished deliverable handed back to the UI."""
    question: str
    paper: Paper
    answer: str
    citations: list[Citation] = field(default_factory=list)
    chunks_used: list[Chunk] = field(default_factory=list)
    stage_timings: dict[str, float] = field(default_factory=dict)
    selection_reason: str = ""
    shortlist: list[Paper] = field(default_factory=list)   # runners-up, shown in the UI
    cache_hit: bool = False                                # was the paper already indexed?

    @property
    def total_seconds(self) -> float:
        return sum(self.stage_timings.values())
