"""Streamlit front end -- a three-panel "notebook" workspace.

    Sources  |  Chat  |  Studio

Deliberately thin: NO business logic, NO API calls except through
agent_core.run_pipeline. All this file does is keep the conversation in
session_state and lay it out. Keep it that way.

Run with:  streamlit run app.py
"""

from __future__ import annotations

import html
import logging
from datetime import datetime

import streamlit as st

from agent_core import run_pipeline
from config import ConfigError, load_settings
from llm import LLMError
from models import Paper, Report
from tools import NoResultsError, PDFError

logging.basicConfig(level=logging.INFO)
for _noisy in ("httpx", "arxiv", "google_genai", "urllib3"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

st.set_page_config(
    page_title="Research Co-Pilot",
    page_icon=":material/auto_stories:",
    layout="wide",
    initial_sidebar_state="collapsed",
)

EXAMPLES = [
    ("Diffusion vs GANs", "How do diffusion models compare to GANs for image generation?"),
    ("Catastrophic forgetting", "What causes catastrophic forgetting in continual learning?"),
    ("Sparse MoE efficiency", "Do sparse mixture-of-experts models improve inference efficiency?"),
    ("RAG hallucinations", "Does retrieval-augmented generation reduce hallucinations in LLMs?"),
]

STAGE_LABELS = {
    "search": "Search arXiv",
    "select": "Select paper",
    "ingest": "Read & index",
    "retrieve": "Retrieve passages",
    "synthesize": "Write answer",
}

PANEL_HEIGHT = 720   # px -- each panel scrolls on its own, like a desktop app


# ---------------------------------------------------------------------------
# styling
# ---------------------------------------------------------------------------

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

:root {
  --bg: #f3f5f9;
  --panel: #ffffff;
  --ink: #1f2430;
  --muted: #667085;
  --line: #e4e7ee;
  --accent: #3a5ccc;
  --accent-soft: #eaf0ff;
  --chip: #f1f3f8;
}

html, body, [class*="css"], .stMarkdown, .stButton button, textarea, input {
  font-family: 'Inter', system-ui, -apple-system, 'Segoe UI', sans-serif;
}
.stApp { background: var(--bg); }
header[data-testid="stHeader"] { background: transparent; height: 0; }
#MainMenu, footer, [data-testid="stToolbar"] { visibility: hidden; }
.block-container { padding: 1.1rem 1.4rem 0.6rem; max-width: 100%; }

/* top bar */
.rc-topbar { display:flex; align-items:center; justify-content:space-between;
  margin: 0 0 .9rem; }
.rc-brand { display:flex; align-items:center; gap:.65rem; }
.rc-logo { width:34px; height:34px; border-radius:10px;
  background: linear-gradient(135deg,#3a5ccc,#7b5cd6); color:#fff;
  display:flex; align-items:center; justify-content:center; font-weight:700; }
.rc-title { font-size:1.15rem; font-weight:600; color:var(--ink); line-height:1.1; }
.rc-sub { font-size:.78rem; color:var(--muted); }
.rc-pill { font-size:.72rem; padding:.25rem .6rem; border-radius:999px;
  background:#fff4e0; color:#8a5a00; border:1px solid #f3dfb5; }

/* the three panels */
.st-key-panel_sources, .st-key-panel_chat, .st-key-panel_studio {
  background: var(--panel); border:1px solid var(--line) !important;
  border-radius: 18px !important; }
.rc-panel-head { display:flex; justify-content:space-between; align-items:center;
  font-weight:600; font-size:.95rem; color:var(--ink);
  padding-bottom:.55rem; margin-bottom:.6rem; border-bottom:1px solid var(--line); }
.rc-count { font-size:.72rem; font-weight:500; color:var(--muted);
  background:var(--chip); padding:.1rem .5rem; border-radius:999px; }
.rc-section { font-size:.7rem; letter-spacing:.06em; text-transform:uppercase;
  color:var(--muted); font-weight:600; margin:.9rem 0 .4rem; }

/* source cards */
.rc-src { border:1px solid var(--line); border-radius:12px; padding:.6rem .7rem;
  margin-bottom:.45rem; background:#fff; }
.rc-src.active { border-color:var(--accent); background:var(--accent-soft); }
.rc-src-title { font-size:.84rem; font-weight:600; color:var(--ink); line-height:1.3; }
.rc-src-meta { font-size:.72rem; color:var(--muted); margin-top:.2rem; }
.rc-src-meta a { color:var(--accent); text-decoration:none; }
.rc-doc-icon { display:inline-block; font-size:.62rem; font-weight:700; color:#b42318;
  background:#fee4e2; border-radius:4px; padding:0 .3rem; margin-right:.35rem; }
.rc-mini { font-size:.78rem; color:var(--ink); padding:.3rem 0;
  border-bottom:1px dashed var(--line); }
.rc-mini a { color:var(--ink); text-decoration:none; }
.rc-mini a:hover { color:var(--accent); }
.rc-mini span { color:var(--muted); font-size:.7rem; }

/* chat */
.rc-empty { text-align:center; padding:2.2rem 1rem 1rem; }
.rc-empty h2 { font-size:1.55rem; font-weight:600; color:var(--ink); margin:.4rem 0; }
.rc-empty p { color:var(--muted); font-size:.9rem; max-width:520px; margin:0 auto; }
.rc-steps { display:flex; gap:.5rem; justify-content:center; flex-wrap:wrap; margin:1.1rem 0 .4rem; }
.rc-step { font-size:.74rem; background:var(--chip); color:var(--ink);
  padding:.3rem .65rem; border-radius:999px; }
.rc-paperchip { display:inline-flex; align-items:center; gap:.35rem; font-size:.76rem;
  background:var(--accent-soft); color:var(--accent); padding:.25rem .65rem;
  border-radius:999px; margin-bottom:.5rem; font-weight:500; }
.rc-paperchip a { color:var(--accent); text-decoration:none; }
.rc-why { font-size:.78rem; color:var(--muted); border-left:3px solid var(--line);
  padding-left:.6rem; margin:.2rem 0 .7rem; }
.rc-meta { font-size:.72rem; color:var(--muted); margin-top:.4rem; }
[data-testid="stChatMessage"] { background:transparent; padding:.4rem 0; }
.stChatInput, [data-testid="stChatInput"] textarea { border-radius:16px; }

/* studio */
.rc-card { border:1px solid var(--line); border-radius:12px; padding:.7rem .8rem;
  margin-bottom:.55rem; background:#fff; }
.rc-card-title { font-weight:600; font-size:.84rem; color:var(--ink); margin-bottom:.25rem; }
.rc-card-body { font-size:.8rem; color:#344054; line-height:1.45; }
.rc-cite-head { font-size:.76rem; font-weight:600; color:var(--accent); }
.rc-bar-row { display:flex; align-items:center; gap:.5rem; font-size:.74rem;
  color:var(--ink); margin:.28rem 0; }
.rc-bar-label { width:110px; flex-shrink:0; color:var(--muted); }
.rc-track { flex:1; height:8px; border-radius:4px; background:var(--chip); }
.rc-bar { height:8px; border-radius:4px; background:linear-gradient(90deg,#3a5ccc,#7b5cd6); }
.rc-bar-val { width:48px; text-align:right; font-variant-numeric:tabular-nums; }
.rc-stat-grid { display:grid; grid-template-columns:1fr 1fr; gap:.45rem; margin-bottom:.6rem; }
.rc-stat { background:var(--chip); border-radius:10px; padding:.5rem .6rem; }
.rc-stat b { display:block; font-size:1.05rem; color:var(--ink); }
.rc-stat span { font-size:.68rem; color:var(--muted); text-transform:uppercase; letter-spacing:.04em; }

.stButton button { border-radius:12px; border:1px solid var(--line); font-size:.8rem; }
.stButton button[kind="primary"] { background:var(--accent); border-color:var(--accent); }
.stTabs [data-baseweb="tab"] { font-size:.8rem; padding:.3rem .55rem; }

@media (max-width: 900px) {
  .block-container { padding: .7rem .6rem; }
}
</style>
"""


# ---------------------------------------------------------------------------
# state
# ---------------------------------------------------------------------------

def init_state() -> None:
    ss = st.session_state
    ss.setdefault("messages", [])        # [{"role", "content", "report"?, "error"?}]
    ss.setdefault("sources", {})         # arxiv_id -> Paper, every paper the agent has read
    ss.setdefault("considered", {})      # arxiv_id -> Paper, runners-up across all turns
    ss.setdefault("notes", [])           # [{"title", "body", "saved"}]
    ss.setdefault("active_source", None)
    ss.setdefault("queued", None)        # question clicked from a suggestion button
    ss.setdefault("force_search", False)
    ss.setdefault("flash", None)         # one-shot toast that survives a rerun


def latest_report() -> Report | None:
    for msg in reversed(st.session_state.messages):
        if msg.get("report") is not None:
            return msg["report"]
    return None


def esc(text: str) -> str:
    return html.escape(str(text or ""))


# ---------------------------------------------------------------------------
# panels
# ---------------------------------------------------------------------------

def render_topbar() -> None:
    st.markdown(
        """
        <div class="rc-topbar">
          <div class="rc-brand">
            <div class="rc-logo">R</div>
            <div>
              <div class="rc-title">Academic Research Co-Pilot</div>
              <div class="rc-sub">Ask a question &middot; the agent finds, reads and cites one arXiv paper</div>
            </div>
          </div>
          <span class="rc-pill">Free-tier demo &middot; shared daily quota</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def source_card(paper: Paper, active: bool) -> str:
    return f"""
    <div class="rc-src {'active' if active else ''}">
      <div class="rc-src-title"><span class="rc-doc-icon">PDF</span>{esc(paper.title)}</div>
      <div class="rc-src-meta">{esc(paper.short_authors(2))} &middot; {paper.published.year}
        &middot; <a href="{esc(paper.abs_url)}" target="_blank">arXiv:{esc(paper.arxiv_id)}</a></div>
    </div>"""


def render_sources() -> None:
    ss = st.session_state
    sources: dict[str, Paper] = ss.sources
    st.markdown(
        f'<div class="rc-panel-head">Sources <span class="rc-count">{len(sources)}</span></div>',
        unsafe_allow_html=True,
    )

    if not sources:
        st.caption("No sources yet. Ask a question and the agent will pick the most "
                   "relevant arXiv paper and add it here.")
    else:
        for arxiv_id, paper in reversed(list(sources.items())):
            st.markdown(source_card(paper, arxiv_id == ss.active_source),
                        unsafe_allow_html=True)
            if st.button("View source guide", key=f"view_{arxiv_id}",
                         use_container_width=True):
                ss.active_source = arxiv_id
                st.rerun()

    if ss.active_source and ss.active_source in sources:
        paper = sources[ss.active_source]
        st.markdown('<div class="rc-section">Source guide</div>', unsafe_allow_html=True)
        with st.container(border=True):
            st.markdown(f"**{paper.title}**")
            st.caption(", ".join(paper.authors[:8]) + (" et al." if len(paper.authors) > 8 else ""))
            st.caption(f"Published {paper.published:%d %b %Y}")
            st.markdown(f"<div class='rc-card-body'>{esc(paper.abstract)}</div>",
                        unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            c1.link_button("Abstract", paper.abs_url, use_container_width=True)
            c2.link_button("PDF", paper.pdf_url, use_container_width=True)

    if ss.considered:
        st.markdown(
            f'<div class="rc-section">Also considered ({len(ss.considered)})</div>',
            unsafe_allow_html=True,
        )
        rows = "".join(
            f'<div class="rc-mini"><a href="{esc(p.abs_url)}" target="_blank">{esc(p.title)}</a> '
            f'<span>&middot; {p.published.year}</span></div>'
            for p in list(ss.considered.values())[:20]
        )
        st.markdown(rows, unsafe_allow_html=True)

    st.markdown('<div class="rc-section">Settings</div>', unsafe_allow_html=True)
    ss.force_search = st.toggle(
        "Always search fresh", value=ss.force_search,
        help="Ignore remembered question → paper matches and search arXiv again.",
    )
    if ss.messages and st.button("Clear notebook", icon=":material/delete_sweep:",
                                 use_container_width=True):
        for key in ("messages", "sources", "considered", "notes", "active_source"):
            del ss[key]
        st.rerun()


def render_answer(report: Report, idx: int) -> None:
    paper = report.paper
    st.markdown(
        f'<div class="rc-paperchip">📄 <a href="{esc(paper.abs_url)}" target="_blank">'
        f'{esc(paper.title[:90])}{"…" if len(paper.title) > 90 else ""}</a></div>',
        unsafe_allow_html=True,
    )
    if report.selection_reason:
        st.markdown(f'<div class="rc-why"><b>Why this paper:</b> {esc(report.selection_reason)}</div>',
                    unsafe_allow_html=True)

    st.markdown(report.answer)

    if report.citations:
        cols = st.columns(min(len(report.citations), 6))
        for i, citation in enumerate(report.citations[:12]):
            with cols[i % len(cols)]:
                with st.popover(f"[{citation.number}] p.{citation.page}", use_container_width=True):
                    st.markdown(f"**[{citation.number}] {citation.section}** · page {citation.page}")
                    st.caption(citation.excerpt)

    meta = f"{report.total_seconds:.1f}s · {len(report.chunks_used)} passages retrieved"
    if report.cache_hit:
        meta += " · paper served from cache"
    b1, b2, _ = st.columns([1.5, 1.2, 2.3])
    with b1:
        if st.button("Save to note", key=f"save_{idx}", icon=":material/push_pin:",
                     use_container_width=True):
            st.session_state.notes.append({
                "title": report.question,
                "body": report.answer,
                "source": f"{paper.title} ({paper.abs_url})",
                "saved": datetime.now().strftime("%d %b, %H:%M"),
            })
            st.session_state.flash = "Saved to notes"
            st.rerun()   # so the Studio panel's Notes tab updates immediately
    with b2:
        st.download_button("Export", data=report_markdown(report), key=f"dl_{idx}",
                           file_name=f"answer-{paper.arxiv_id}.md", mime="text/markdown",
                           icon=":material/download:", use_container_width=True)
    st.markdown(f'<div class="rc-meta">{meta}</div>', unsafe_allow_html=True)


def render_empty_state() -> None:
    st.markdown(
        """
        <div class="rc-empty">
          <div style="font-size:2.2rem">📚</div>
          <h2>What do you want to research?</h2>
          <p>Ask in plain language. The agent rewrites it into arXiv searches, picks
             the single best paper, reads it, and answers only from what it says —
             with numbered citations you can check.</p>
          <div class="rc-steps">
            <span class="rc-step">1 · Search</span><span class="rc-step">2 · Select</span>
            <span class="rc-step">3 · Read</span><span class="rc-step">4 · Retrieve</span>
            <span class="rc-step">5 · Cite</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(2)
    for i, (label, question) in enumerate(EXAMPLES):
        with cols[i % 2]:
            if st.button(f"**{label}**  \n{question}", key=f"ex_{i}",
                         use_container_width=True):
                st.session_state.queued = question
                st.rerun()


def run_question(question: str, settings) -> None:
    """Render one live turn, call the pipeline, store the result."""
    ss = st.session_state
    ss.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant", avatar=":material/auto_stories:"):
        status = st.status("Researching…", expanded=True)
        entry: dict = {"role": "assistant", "content": ""}
        try:
            report = run_pipeline(
                question, settings,
                on_stage=lambda key, message: status.write(
                    f"**{STAGE_LABELS.get(key, key)}** — {message}"),
                force_search=ss.force_search,
            )
        except NoResultsError as exc:
            status.update(label="No papers found", state="error")
            entry["error"] = str(exc)
        except PDFError as exc:
            status.update(label="Could not read the paper", state="error")
            entry["error"] = f"{exc}\n\nTry rephrasing so a different paper is selected."
        except LLMError as exc:
            status.update(label="Model call failed", state="error")
            if "429" in str(exc) or "RESOURCE_EXHAUSTED" in str(exc):
                entry["error"] = (
                    "**This demo has hit its free-tier API quota.** It runs on a free "
                    "Gemini key shared by everyone visiting, and resets every 24 hours. "
                    "Clone the repo and use your own free key to run it without limits."
                )
            else:
                entry["error"] = f"The language model call failed: {exc}"
        except Exception as exc:                   # noqa: BLE001 -- UI must never traceback
            status.update(label="Something went wrong", state="error")
            entry["error"] = f"{type(exc).__name__}: {exc}"
        else:
            status.update(label=f"Done in {report.total_seconds:.1f}s", state="complete",
                          expanded=False)
            entry["report"] = report
            entry["content"] = report.answer
            ss.sources[report.paper.arxiv_id] = report.paper
            ss.active_source = report.paper.arxiv_id
            for other in report.shortlist:
                if other.arxiv_id not in ss.sources:
                    ss.considered.setdefault(other.arxiv_id, other)
            ss.considered.pop(report.paper.arxiv_id, None)

    ss.messages.append(entry)
    st.rerun()   # redraw so the Sources and Studio panels pick up the new turn


def render_chat(settings) -> None:
    ss = st.session_state
    st.markdown('<div class="rc-panel-head">Chat</div>', unsafe_allow_html=True)

    history = st.container(height=PANEL_HEIGHT - 150, border=False)
    with history:
        if not ss.messages and not ss.queued:
            render_empty_state()
        for idx, msg in enumerate(ss.messages):
            if msg["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(msg["content"])
            else:
                with st.chat_message("assistant", avatar=":material/auto_stories:"):
                    if msg.get("error"):
                        st.warning(msg["error"])
                    else:
                        render_answer(msg["report"], idx)

    prompt = st.chat_input("Ask a research question…")
    question = (prompt or ss.queued or "").strip()
    if question:
        ss.queued = None
        with history:
            run_question(question, settings)


def report_markdown(report: Report) -> str:
    paper = report.paper
    lines = [f"# {report.question}", "",
             f"**Source:** [{paper.title}]({paper.abs_url}) — {paper.short_authors()}, "
             f"{paper.published.year}", "", report.answer, ""]
    if report.citations:
        lines += ["## Cited passages", ""]
        for c in report.citations:
            lines += [f"**[{c.number}]** {c.section}, page {c.page}", f"> {c.excerpt}", ""]
    return "\n".join(lines)


def notes_markdown() -> str:
    parts = ["# Research notebook", ""]
    for note in st.session_state.notes:
        parts += [f"## {note['title']}", f"*Saved {note['saved']} · {note['source']}*", "",
                  note["body"], ""]
    return "\n".join(parts)


def render_studio() -> None:
    ss = st.session_state
    report = latest_report()
    st.markdown('<div class="rc-panel-head">Studio</div>', unsafe_allow_html=True)

    tab_guide, tab_cites, tab_pass, tab_notes = st.tabs(
        ["Overview", "Citations", "Passages", f"Notes ({len(ss.notes)})"])

    with tab_guide:
        if report is None:
            st.caption("An overview of the latest answer appears here: the paper, "
                       "how long each stage took and what was retrieved.")
        else:
            paper = report.paper
            st.markdown(
                f"""
                <div class="rc-stat-grid">
                  <div class="rc-stat"><b>{report.total_seconds:.1f}s</b><span>Total time</span></div>
                  <div class="rc-stat"><b>{len(report.citations)}</b><span>Citations</span></div>
                  <div class="rc-stat"><b>{len(report.chunks_used)}</b><span>Passages</span></div>
                  <div class="rc-stat"><b>{len(report.shortlist) + 1}</b><span>Papers screened</span></div>
                </div>
                <div class="rc-card">
                  <div class="rc-card-title">{esc(paper.title)}</div>
                  <div class="rc-src-meta">{esc(paper.short_authors())} &middot; {paper.published.year}</div>
                  <div class="rc-card-body" style="margin-top:.4rem">{esc(paper.abstract[:420])}{'…' if len(paper.abstract) > 420 else ''}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.markdown('<div class="rc-section">Pipeline timing</div>', unsafe_allow_html=True)
            slowest = max(report.stage_timings.values() or [1]) or 1
            bars = "".join(
                f'<div class="rc-bar-row"><span class="rc-bar-label">{esc(STAGE_LABELS.get(k, k))}</span>'
                f'<div class="rc-track"><div class="rc-bar" style="width:{max(3, int(100 * v / slowest))}%"></div></div>'
                f'<span class="rc-bar-val">{v:.2f}s</span></div>'
                for k, v in report.stage_timings.items()
            )
            st.markdown(bars, unsafe_allow_html=True)
            if report.cache_hit:
                st.caption("Paper was already indexed, so download and chunking were skipped.")

            st.markdown('<div class="rc-section">Follow-up ideas</div>', unsafe_allow_html=True)
            followups = [
                f"What are the limitations of the approach in \"{paper.title[:60]}\"?",
                f"What datasets and metrics does \"{paper.title[:60]}\" use?",
            ]
            for i, q in enumerate(followups):
                if st.button(q, key=f"fu_{i}", use_container_width=True):
                    ss.queued = q
                    st.rerun()

    with tab_cites:
        if not report or not report.citations:
            st.caption("Numbered passages the answer relies on will appear here.")
        else:
            for c in report.citations:
                st.markdown(
                    f'<div class="rc-card"><div class="rc-cite-head">[{c.number}] {esc(c.section)} · page {c.page}</div>'
                    f'<div class="rc-card-body">{esc(c.excerpt)}</div></div>',
                    unsafe_allow_html=True,
                )

    with tab_pass:
        if not report:
            st.caption("Every passage retrieved from the paper, ranked by similarity.")
        else:
            for chunk in report.chunks_used:
                score = f"{chunk.score:.3f}" if chunk.score is not None else "–"
                with st.expander(f"{chunk.section} · p.{chunk.page} · sim {score}"):
                    st.caption(chunk.text)

    with tab_notes:
        if not ss.notes:
            st.caption("Pin any answer with **Save to note** to collect it here, "
                       "then export the whole notebook as Markdown.")
        else:
            st.download_button("Export notebook (.md)", data=notes_markdown(),
                               file_name="research-notebook.md", mime="text/markdown",
                               icon=":material/download:", use_container_width=True)
            for i, note in enumerate(reversed(ss.notes)):
                real = len(ss.notes) - 1 - i
                with st.expander(f"📌 {note['title'][:70]}"):
                    st.caption(f"Saved {note['saved']}")
                    st.markdown(note["body"])
                    if st.button("Delete", key=f"del_note_{real}", icon=":material/delete:"):
                        ss.notes.pop(real)
                        st.rerun()


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    init_state()
    if st.session_state.flash:
        st.toast(st.session_state.flash, icon=":material/push_pin:")
        st.session_state.flash = None
    render_topbar()

    try:
        settings = load_settings()
    except ConfigError as exc:
        st.error(f"**Configuration problem**\n\n{exc}")
        st.stop()

    left, center, right = st.columns([1, 2.1, 1.25], gap="small")
    with left:
        with st.container(height=PANEL_HEIGHT, key="panel_sources"):
            render_sources()
    with right:
        with st.container(height=PANEL_HEIGHT, key="panel_studio"):
            render_studio()
    with center:
        with st.container(height=PANEL_HEIGHT, key="panel_chat"):
            render_chat(settings)


if __name__ == "__main__":
    main()
