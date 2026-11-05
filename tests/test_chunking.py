"""Tests for the ingest + chunking path.

Run with:  pytest tests/ -v

This is the one part of the project worth unit-testing. Chunking bugs are
SILENT -- retrieval still returns something, it's just subtly worse -- so
without tests you won't notice until the answers are bad.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from models import Page
from tools import strip_references


# ---------------------------------------------------------------------
# strip_references
# ---------------------------------------------------------------------

def _pages(*texts: str) -> list[Page]:
    return [Page(number=i + 1, text=t) for i, t in enumerate(texts)]


def test_strips_from_references_heading():
    pages = _pages(
        "Introduction\nWe study things.",
        "Methods\nWe did experiments.",
        "Results\nIt worked.",
        "References\n[1] Someone et al.\n[2] Another et al.",
    )
    kept = strip_references(pages)
    body = " ".join(p.text for p in kept)
    assert "It worked" in body
    assert "Someone et al" not in body


def test_keeps_partial_page_before_heading():
    """The heading usually sits mid-page -- text above it must survive."""
    pages = _pages(
        "Intro",
        "Methods",
        "Conclusion\nWe conclude that X.\nReferences\n[1] Someone et al.",
    )
    kept = strip_references(pages)
    body = " ".join(p.text for p in kept)
    assert "We conclude that X" in body
    assert "Someone et al" not in body


def test_ignores_the_word_references_in_prose():
    """'references' inside a sentence is not a heading."""
    pages = _pages(
        "Intro",
        "Methods",
        "We compare our references to prior work.\nResults follow.",
    )
    kept = strip_references(pages)
    assert "Results follow" in " ".join(p.text for p in kept)


def test_ignores_heading_in_first_half():
    """A paper that mentions a References heading early must not be truncated."""
    pages = _pages(
        "References\nthis is page one somehow",
        "Methods",
        "Results\nThe real content.",
        "Discussion",
    )
    kept = strip_references(pages)
    assert "The real content" in " ".join(p.text for p in kept)


def test_handles_numbered_and_alternative_headings():
    for heading in ("5. References", "Bibliography", "REFERENCES", "Works Cited"):
        pages = _pages("Intro", "Methods", f"{heading}\n[1] Someone.")
        kept = strip_references(pages)
        assert "Someone" not in " ".join(p.text for p in kept), heading


def test_no_heading_keeps_everything():
    pages = _pages("Intro", "Methods", "Results")
    assert len(strip_references(pages)) == 3


def test_empty_input():
    assert strip_references([]) == []


# ---------------------------------------------------------------------
# semantic_chunk -- YOUR TASK. Un-skip these as you implement it.
# ---------------------------------------------------------------------

@pytest.mark.skip(reason="Phase 3: implement semantic_chunk first")
def test_chunks_respect_token_cap():
    raise NotImplementedError


@pytest.mark.skip(reason="Phase 3: implement semantic_chunk first")
def test_chunks_are_not_cut_mid_sentence():
    raise NotImplementedError


@pytest.mark.skip(reason="Phase 3: implement semantic_chunk first")
def test_tiny_chunks_are_dropped():
    raise NotImplementedError


@pytest.mark.skip(reason="Phase 3: implement semantic_chunk first")
def test_chunk_count_is_capped():
    raise NotImplementedError
