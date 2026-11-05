"""Preflight check. Run this BEFORE debugging anything else.

    python test_setup.py

Verifies Python version, env vars, Gemini generation, Gemini embeddings,
Pinecone connectivity, and arXiv access -- and prints a pass/fail table.

Built early on purpose: almost every confusing failure in phases 2-5 turns
out to be a bad key, a stale model ID, or a dimension mismatch. This tells
you which in ten seconds instead of an hour.
"""

from __future__ import annotations

import sys
import traceback
from dataclasses import dataclass

GREEN, RED, YELLOW, DIM, RESET = "\033[92m", "\033[91m", "\033[93m", "\033[2m", "\033[0m"


@dataclass
class Result:
    name: str
    ok: bool
    detail: str = ""
    warning: bool = False


results: list[Result] = []


def check(name: str, fn) -> bool:
    """Run one check, capture the failure rather than exploding."""
    try:
        detail = fn() or ""
        if isinstance(detail, tuple):          # (detail, is_warning)
            detail, warn = detail
            results.append(Result(name, True, detail, warning=warn))
        else:
            results.append(Result(name, True, detail))
        return True
    except Exception as exc:
        results.append(Result(name, False, f"{type(exc).__name__}: {exc}"))
        return False


# ---------------------------------------------------------------------
# checks
# ---------------------------------------------------------------------

def check_python():
    from config import MIN_PYTHON, check_python_version
    check_python_version()
    return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"


def check_imports():
    import arxiv, fitz, numpy, streamlit          # noqa: F401
    from google import genai                       # noqa: F401
    from pinecone import Pinecone                  # noqa: F401
    return "all packages importable"


def check_config():
    from config import load_settings
    s = load_settings()
    return f"index={s.pinecone_index_name}, embed={s.model_embedding}@{s.embedding_dim}d"


def check_gemini_models(settings, client):
    """Model IDs rotate. Catch a stale config.py here, not mid-pipeline."""
    available = set(client.list_models())
    wanted = {settings.model_triage, settings.model_synthesis, settings.model_embedding}
    missing = sorted(w for w in wanted if w not in available)
    if missing:
        sample = ", ".join(sorted(m for m in available if "flash" in m)[:4])
        return (f"NOT FOUND: {', '.join(missing)} -- update config.py. "
                f"Available e.g.: {sample}"), True
    return f"{len(wanted)} configured models all available"


def check_gemini_generate(settings, client):
    out = client.generate("Reply with exactly the word: pong",
                          model=settings.model_triage, temperature=0.0)
    if "pong" not in out.lower():
        raise RuntimeError(f"unexpected reply: {out[:60]!r}")
    return f"{settings.model_triage} responded"


def check_gemini_embed(settings, client):
    from llm import TASK_DOCUMENT
    vec = client.embed_one("semantic chunking of academic papers", task_type=TASK_DOCUMENT)
    if len(vec) != settings.embedding_dim:
        raise RuntimeError(
            f"dimension mismatch: got {len(vec)}, config says {settings.embedding_dim}. "
            f"Pinecone index must match -- fix config.py or recreate the index."
        )
    return f"{len(vec)}-dim vector returned"


def check_pinecone(settings):
    from pinecone import Pinecone, ServerlessSpec
    pc = Pinecone(api_key=settings.pinecone_api_key)
    existing = {i["name"] for i in pc.list_indexes()}

    if settings.pinecone_index_name not in existing:
        pc.create_index(
            name=settings.pinecone_index_name,
            dimension=settings.embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=settings.pinecone_cloud, region=settings.pinecone_region),
        )
        return f"created index '{settings.pinecone_index_name}' ({settings.embedding_dim}d, cosine)"

    desc = pc.describe_index(settings.pinecone_index_name)
    dim = desc["dimension"]
    if dim != settings.embedding_dim:
        raise RuntimeError(
            f"index '{settings.pinecone_index_name}' has dimension {dim}, "
            f"config expects {settings.embedding_dim}. Delete the index or change config.py."
        )
    stats = pc.Index(settings.pinecone_index_name).describe_index_stats()
    ns = len(stats.get("namespaces") or {})
    return f"index OK ({dim}d, {stats.get('total_vector_count', 0)} vectors, {ns} namespaces)"


def check_arxiv():
    import arxiv
    client = arxiv.Client()
    search = arxiv.Search(query="all:contrastive language image pretraining",
                          max_results=1, sort_by=arxiv.SortCriterion.Relevance)
    first = next(iter(client.results(search)), None)
    if first is None:
        raise RuntimeError("arXiv returned no results")
    return f'found "{first.title[:52]}..."'


# ---------------------------------------------------------------------
# runner
# ---------------------------------------------------------------------

def main() -> int:
    print(f"\n{DIM}Academic Research Co-Pilot -- preflight{RESET}\n")

    if not check("Python version", check_python):
        report(); return 1
    if not check("Dependencies", check_imports):
        print(f"\n{YELLOW}Install them: pip install -r requirements.txt{RESET}")
        report(); return 1
    if not check("Configuration", check_config):
        print(f"\n{YELLOW}Copy .env.example to .env and fill in your keys.{RESET}")
        report(); return 1

    from config import load_settings
    from llm import GeminiClient
    settings = load_settings()

    try:
        client = GeminiClient(settings)
    except Exception as exc:
        results.append(Result("Gemini client", False, str(exc)))
        report(); return 1

    check("Gemini model IDs", lambda: check_gemini_models(settings, client))
    check("Gemini generation", lambda: check_gemini_generate(settings, client))
    check("Gemini embeddings", lambda: check_gemini_embed(settings, client))
    check("Pinecone", lambda: check_pinecone(settings))
    check("arXiv", check_arxiv)

    return report()


def report() -> int:
    width = max(len(r.name) for r in results)
    print()
    for r in results:
        if not r.ok:
            mark, colour = "FAIL", RED
        elif r.warning:
            mark, colour = "WARN", YELLOW
        else:
            mark, colour = "PASS", GREEN
        print(f"  {colour}{mark}{RESET}  {r.name.ljust(width)}  {DIM}{r.detail}{RESET}")

    failed = [r for r in results if not r.ok]
    warned = [r for r in results if r.ok and r.warning]
    print()
    if failed:
        print(f"{RED}{len(failed)} check(s) failed.{RESET} Fix these before moving on.\n")
        return 1
    if warned:
        print(f"{YELLOW}All checks passed with {len(warned)} warning(s).{RESET}\n")
        return 0
    print(f"{GREEN}All checks passed. You're ready to build.{RESET}\n")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(130)
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
