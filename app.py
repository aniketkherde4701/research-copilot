"""Streamlit front end.

Deliberately thin: NO business logic, NO API calls except through
agent_core.run_pipeline. That separation is what keeps this from becoming a
notebook -- keep it that way.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import logging

import streamlit as st

from agent_core import run_pipeline
from config import ConfigError, load_settings
from llm import LLMError
from tools import NoResultsError, PDFError

logging.basicConfig(level=logging.INFO)
for _noisy in ("httpx", "arxiv", "google_genai", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

st.set_page_config(page_title="Academic Research Co-Pilot", page_icon="=", layout="centered")

EXAMPLES = [
    "How do diffusion models compare to GANs for image generation?",
    "What causes catastrophic forgetting in continual learning?",
    "Do sparse mixture-of-experts models improve inference efficiency?",
]


def main() -> None:
    st.title("Academic Research Co-Pilot")
    st.caption(
        "Ask a research question. The agent searches arXiv, picks the single most "
        "relevant paper, reads it, and answers from its contents -- with citations."
    )
    st.info(
        "Free-tier demo - the API quota is shared across everyone using this page "
        "and resets daily. If it is exhausted, clone the repo and run it with your "
        "own free key.",
        icon=":material/info:",
    )

    try:
        settings = load_settings()
    except ConfigError as exc:
        st.error(f"**Configuration problem**\n\n{exc}")
        st.stop()

    if "question" not in st.session_state:
        st.session_state.question = ""

    # Example buttons must render BEFORE the text area. Streamlit forbids
    # writing to session_state for a key whose widget has already been
    # instantiated this run, so the assignment has to happen first.
    st.caption("Try one:")
    cols = st.columns(len(EXAMPLES))
    for col, example in zip(cols, EXAMPLES):
        with col:
            if st.button(example[:26] + "...", use_container_width=True,
                         help=example):
                st.session_state.question = example
                st.rerun()

    st.text_area("Research question", key="question", height=90,
                 placeholder="e.g. " + EXAMPLES[0])

    question = st.session_state.question.strip()
    if not st.button("Research", type="primary", disabled=not question):
        return

    status = st.status("Working...", expanded=True)
    try:
        report = run_pipeline(
            question, settings,
            on_stage=lambda key, message: status.write(message),
        )
    except NoResultsError as exc:
        status.update(label="No papers found", state="error")
        st.warning(str(exc))
        return
    except PDFError as exc:
        status.update(label="Could not read the paper", state="error")
        st.warning(f"{exc}\n\nTry rephrasing so a different paper is selected.")
        return
    except LLMError as exc:
        status.update(label="Rate limited", state="error")
        if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
            st.warning(
                "**This demo has hit its free-tier API quota.**\n\n"
                "It runs on a free Gemini key shared by everyone visiting this page, "
                "so the daily allowance goes quickly. Quotas reset every 24 hours.\n\n"
                "To run it without limits, clone the repo and use your own key - "
                "setup takes about two minutes and both keys are free."
            )
        else:
            st.error(f"The language model call failed: {exc}")
        return
    except Exception as exc:                       # noqa: BLE001 -- UI must never traceback
        status.update(label="Something went wrong", state="error")
        st.error(f"{type(exc).__name__}: {exc}")
        return

    status.update(label=f"Done in {report.total_seconds:.1f}s", state="complete",
                  expanded=False)

    # -- the paper ---------------------------------------------------------
    paper = report.paper
    st.subheader(paper.title)
    st.caption(f"{paper.short_authors()} - {paper.published.year} - [{paper.arxiv_id}]({paper.abs_url})")
    st.caption(f"**Why this paper:** {report.selection_reason}")
    if report.cache_hit:
        st.caption("Served from cache - this paper was already indexed.")

    # -- the answer --------------------------------------------------------
    st.markdown("---")
    st.markdown(report.answer)

    # -- the grounding -----------------------------------------------------
    st.markdown("---")
    st.markdown(f"**Source:** [{paper.title}]({paper.abs_url})")

    if report.citations:
        with st.expander(f"Cited passages ({len(report.citations)})"):
            for citation in report.citations:
                st.markdown(f"**[{citation.number}]** {citation.section}, page {citation.page}")
                st.caption(citation.excerpt)

    with st.expander(f"All retrieved passages ({len(report.chunks_used)})"):
        for chunk in report.chunks_used:
            score = f" - similarity {chunk.score:.3f}" if chunk.score is not None else ""
            st.markdown(f"**[{chunk.id}]** {chunk.section}, page {chunk.page}{score}")
            st.caption(chunk.text[:400] + ("..." if len(chunk.text) > 400 else ""))

    with st.expander(f"Papers considered but not chosen ({len(report.shortlist)})"):
        for other in report.shortlist[:15]:
            st.markdown(f"- [{other.title}]({other.abs_url}) ({other.published.year})")

    with st.expander("Timing"):
        for stage, seconds in report.stage_timings.items():
            st.text(f"{stage:12} {seconds:6.2f}s")


if __name__ == "__main__":
    main()
