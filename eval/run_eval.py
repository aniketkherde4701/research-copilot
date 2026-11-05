"""Retrieval evaluation harness.

    python eval/run_eval.py            # run / resume
    python eval/run_eval.py --summary  # just print the numbers
    python eval/run_eval.py --reset    # start over

For each question it runs the full pipeline, shows you the paper, the answer and
the cited passages, and asks two yes/no judgements. Results save after every
question, so you can stop with Ctrl+C and pick up where you left off.

The two judgements are deliberately different things:
  SELECTION  -- did it choose a paper that could plausibly answer the question?
  FAITHFULNESS -- do the cited passages actually support what the answer claims?
Selection failures are visible; faithfulness failures are not, which is why the
second question is the one that matters.
"""

from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import load_settings
from agent_core import run_pipeline

ROOT = Path(__file__).parent
QUESTIONS = ROOT / "questions.txt"
RESULTS = ROOT / "results.json"

BOLD, DIM, GREEN, RED, YELLOW, RESET = "\033[1m", "\033[2m", "\033[92m", "\033[91m", "\033[93m", "\033[0m"

REFUSAL_HINTS = (
    "does not address", "do not address", "does not answer", "cannot answer",
    "no relevant passages", "not covered", "does not discuss", "is missing",
)


def load_questions() -> list[str]:
    lines = QUESTIONS.read_text(encoding="utf-8").splitlines()
    return [ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")]


def load_results() -> dict:
    if RESULTS.exists():
        try:
            return json.loads(RESULTS.read_text())
        except Exception:
            print(f"{YELLOW}results.json unreadable; starting fresh{RESET}")
    return {}


def save_results(data: dict) -> None:
    RESULTS.write_text(json.dumps(data, indent=2))


def ask(prompt: str, *, allow_partial: bool = False) -> str:
    """y = yes, p = partly, n = no, s = skip (not judged, left out of the score)."""
    keys = "y = yes   p = partly   n = no   s = skip (not counted)" if allow_partial \
        else "y = yes   n = no   s = skip (not counted)"
    valid = ("y", "p", "n", "s") if allow_partial else ("y", "n", "s")
    words = {"yes": "y", "partly": "p", "partial": "p", "partially": "p",
             "no": "n", "skip": "s"}
    while True:
        reply = input(f"  {BOLD}{prompt}{RESET}\n  {DIM}{keys}{RESET}\n  > ").strip().lower()
        reply = words.get(reply, reply)
        if reply in valid:
            return reply
        print(f"  {DIM}Type one of: {', '.join(valid)}{RESET}")


def show(report) -> bool:
    """Print the result. Returns True if it looks like a refusal."""
    p = report.paper
    print(f"\n  {BOLD}PAPER{RESET}  {p.title}")
    print(f"         {DIM}{p.short_authors()} - {p.published.year} - {p.abs_url}{RESET}")
    print(f"  {BOLD}WHY{RESET}    {report.selection_reason}")
    print(f"  {BOLD}CACHE{RESET}  {'hit - already indexed' if report.cache_hit else 'miss - freshly indexed'}")

    print(f"\n  {BOLD}ANSWER{RESET}")
    for line in report.answer.splitlines():
        print(f"    {line}")

    if report.citations:
        # Show the FULL chunk the model was given, not the truncated preview.
        # Judging faithfulness against a 300-char excerpt of a 2000-char chunk
        # makes everything look partly-supported regardless of the truth.
        full = {c.id: c.text for c in report.chunks_used}
        print(f"\n  {BOLD}CITED PASSAGES{RESET}  {DIM}(full text the model was given){RESET}")
        for c in report.citations:
            body = full.get(str(c.number), c.excerpt)
            print(f"\n    {BOLD}[{c.number}]{RESET} {DIM}{c.section}, page {c.page}"
                  f" - {len(body)} chars{RESET}")
            for line in _wrap(body, 88):
                print(f"        {line}")
    else:
        print(f"\n  {YELLOW}No citations in the answer.{RESET}")

    return any(h in report.answer.lower() for h in REFUSAL_HINTS)


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur); cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def run(settings) -> None:
    questions = load_questions()
    results = load_results()
    todo = [q for q in questions if q not in results]

    print(f"\n{BOLD}Retrieval evaluation{RESET}")
    print(f"{DIM}{len(results)} of {len(questions)} already judged; {len(todo)} to go.")
    print(f"Ctrl+C to stop - progress is saved after each question.{RESET}\n")

    for i, question in enumerate(todo, 1):
        print("=" * 92)
        print(f"{BOLD}[{len(results)+1}/{len(questions)}]{RESET}  {question}")
        print("=" * 92)

        started = time.monotonic()
        try:
            report = run_pipeline(question, settings)
        except Exception as exc:
            elapsed = time.monotonic() - started
            print(f"  {RED}FAILED after {elapsed:.1f}s: {type(exc).__name__}: {exc}{RESET}")
            results[question] = {"error": f"{type(exc).__name__}: {exc}",
                                 "seconds": round(elapsed, 1)}
            save_results(results)
            continue

        elapsed = time.monotonic() - started
        looks_refused = show(report)
        print(f"\n  {DIM}{elapsed:.1f}s - {len(report.chunks_used)} passages retrieved"
              f" - {len(report.citations)} cited{RESET}")
        if looks_refused:
            print(f"  {YELLOW}This reads like a refusal - judge whether refusing was right.{RESET}")

        print()
        selection = ask("Could this paper plausibly answer the question?")
        faithful = ask("Do the CITED PASSAGES above support what the answer claims? "
                       "(compare the answer's claims against the excerpt text -- "
                       "no need to open the PDF)", allow_partial=True)

        note = ""
        if faithful in ("p", "n"):
            note = input(f"  {DIM}Which claim did not hold? (one line, Enter to skip){RESET}\n  > ").strip()

        results[question] = {
            "arxiv_id": report.paper.arxiv_id,
            "title": report.paper.title,
            "url": report.paper.abs_url,
            "confidence": report.selection_reason and getattr(report, "confidence", None),
            "seconds": round(elapsed, 1),
            "cache_hit": report.cache_hit,
            "chunks": len(report.chunks_used),
            "citations": len(report.citations),
            "refused": looks_refused,
            "selection_ok": selection,
            "faithful": faithful,
            "note": note,
        }
        save_results(results)
        print(f"  {GREEN}saved{RESET}\n")

    summary()


def summary() -> None:
    results = load_results()
    if not results:
        print("No results yet.")
        return

    judged = {q: r for q, r in results.items() if "error" not in r}
    errors = {q: r for q, r in results.items() if "error" in r}

    def counts(field: str) -> dict:
        vals = [r[field] for r in judged.values() if r.get(field) in ("y", "p", "n")]
        return {"y": vals.count("y"), "p": vals.count("p"),
                "n": vals.count("n"), "total": len(vals)}

    sel = counts("selection_ok")
    fai = counts("faithful")

    cold = [r["seconds"] for r in judged.values() if not r.get("cache_hit")]
    warm = [r["seconds"] for r in judged.values() if r.get("cache_hit")]
    refused = sum(1 for r in judged.values() if r.get("refused"))
    no_cites = sum(1 for r in judged.values() if r.get("citations") == 0)

    print(f"\n{BOLD}{'=' * 60}")
    print(f"RESULTS - {len(judged)} questions judged, {len(errors)} errors")
    print(f"{'=' * 60}{RESET}\n")
    def pct(n: int, total: int) -> str:
        return f"{n / total * 100:.0f}%" if total else "-"

    print(f"  Paper selection sensible       {BOLD}{sel['y']}/{sel['total']}{RESET}"
          f"  ({pct(sel['y'], sel['total'])})")
    print(f"  Citations fully support answer {BOLD}{fai['y']}/{fai['total']}{RESET}"
          f"  ({pct(fai['y'], fai['total'])})")
    if fai["p"]:
        print(f"    {DIM}partly supported{RESET}           {fai['p']}/{fai['total']}"
              f"  {DIM}(some claims held, some did not){RESET}")
    if fai["n"]:
        print(f"    {RED}unsupported{RESET}                {fai['n']}/{fai['total']}"
              f"  {DIM}<- these are the ones to investigate{RESET}")
    print(f"  Answers that refused         {refused}/{len(judged)}")
    print(f"  Answers with no citation     {no_cites}/{len(judged)}")

    for label, xs in (("cold", cold), ("cached", warm)):
        if xs:
            med = statistics.median(xs)
            print(f"\n  Latency, {label:6} n={len(xs):2}  "
                  f"median {med:5.1f}s   min {min(xs):5.1f}s   max {max(xs):5.1f}s")

    notes = [(q, r["note"], r["faithful"]) for q, r in judged.items() if r.get("note")]
    if notes:
        print(f"\n  {BOLD}Where the grounding slipped{RESET}  {DIM}(cases worth investigating){RESET}")
        for q, n, f in notes:
            tag = "partly" if f == "p" else "unsupported"
            print(f"    {DIM}[{tag}]{RESET} {q[:54]}")
            print(f"           {n[:78]}")

    if errors:
        print(f"\n  {RED}Errors{RESET}")
        for q, r in errors.items():
            print(f"    {q[:58]:60} {r['error'][:40]}")

    print(f"\n{DIM}  Raw results: {RESULTS}{RESET}\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--summary", action="store_true", help="print numbers and exit")
    parser.add_argument("--reset", action="store_true", help="delete saved results")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    for noisy in ("httpx", "arxiv", "google_genai", "urllib3", "pinecone"):
        logging.getLogger(noisy).setLevel(logging.ERROR)

    if args.reset:
        RESULTS.unlink(missing_ok=True)
        print("Results cleared.")
        return 0
    if args.summary:
        summary()
        return 0

    try:
        run(load_settings())
    except KeyboardInterrupt:
        print(f"\n\n{YELLOW}Stopped. Progress saved - rerun to continue.{RESET}\n")
        summary()
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
