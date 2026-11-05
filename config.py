"""Central configuration. Every tunable constant and model ID lives here.

Nothing else in the codebase should read os.environ directly -- that keeps
secrets in one place and makes model swaps a one-line change.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent
PROMPTS_DIR = PROJECT_ROOT / "prompts"
CACHE_DIR = PROJECT_ROOT / "cache"

MIN_PYTHON = (3, 11)


class ConfigError(RuntimeError):
    """Raised when the environment is not usable. Fail loud, fail early."""


@dataclass(frozen=True)
class Settings:
    # --- credentials -------------------------------------------------
    gemini_api_key: str
    pinecone_api_key: str

    # --- pinecone ----------------------------------------------------
    pinecone_index_name: str = "research-copilot"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # --- models ------------------------------------------------------
    # Gemini model IDs rotate. test_setup.py lists what is actually
    # available and warns if these are stale. Change them here only.
    model_triage: str = "gemini-3.5-flash-lite"   # cheap: query rewrite + paper selection
    model_synthesis: str = "gemini-3.5-flash"     # better: the final report
    model_embedding: str = "gemini-embedding-001" # still supports task_type (see README)
    embedding_dim: int = 768

    # --- retrieval ---------------------------------------------------
    arxiv_candidates: int = 30      # papers shortlisted before LLM triage
    arxiv_per_query: int = 25       # fetched per query before interleaving
    retrieval_top_k: int = 8        # chunks fed to synthesis

    # --- chunking ----------------------------------------------------
    chunk_min_tokens: int = 30
    chunk_target_tokens: int = 400
    chunk_max_tokens: int = 600
    chunk_breakpoint_percentile: int = 95   # cut where consecutive-sentence distance is unusually high
    max_chunks_per_paper: int = 40          # each chunk = 1 embed request on the free tier

    # --- free-tier budget --------------------------------------------
    # gemini-embedding-001 accepts ONE input per request, so every unit we
    # embed is a separate API call against a 100/min free-tier quota.
    embed_requests_per_minute: int = 50     # well under 100: retries and prior runs count too
    max_boundary_units: int = 25            # windows embedded to find topic cuts

    # --- misc --------------------------------------------------------
    request_timeout_s: int = 60
    max_retries: int = 5

    cache_dir: Path = field(default=CACHE_DIR)


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ConfigError(
            f"Missing required environment variable: {name}\n"
            f"  -> copy .env.example to .env and fill it in."
        )
    return value


def check_python_version() -> None:
    if sys.version_info < MIN_PYTHON:
        raise ConfigError(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ required, "
            f"found {sys.version_info.major}.{sys.version_info.minor}"
        )


def load_settings() -> Settings:
    """Load and validate configuration. Raises ConfigError with a readable message."""
    load_dotenv(PROJECT_ROOT / ".env")
    check_python_version()

    settings = Settings(
        gemini_api_key=_require("GEMINI_API_KEY"),
        pinecone_api_key=_require("PINECONE_API_KEY"),
        pinecone_index_name=os.environ.get("PINECONE_INDEX_NAME", "research-copilot"),
        pinecone_cloud=os.environ.get("PINECONE_CLOUD", "aws"),
        pinecone_region=os.environ.get("PINECONE_REGION", "us-east-1"),
    )
    settings.cache_dir.mkdir(exist_ok=True)
    return settings


def load_prompt(name: str) -> str:
    """Read a prompt template from prompts/. Keeps prompts out of the code."""
    path = PROMPTS_DIR / f"{name}.txt"
    if not path.exists():
        raise ConfigError(f"Prompt template not found: {path}")
    return path.read_text(encoding="utf-8")


if __name__ == "__main__":
    try:
        s = load_settings()
    except ConfigError as exc:
        print(f"FAIL: {exc}")
        raise SystemExit(1)
    print("Config loaded.")
    print(f"  index      : {s.pinecone_index_name} ({s.pinecone_cloud}/{s.pinecone_region})")
    print(f"  triage     : {s.model_triage}")
    print(f"  synthesis  : {s.model_synthesis}")
    print(f"  embedding  : {s.model_embedding} @ {s.embedding_dim}d")
    print(f"  keys       : gemini={'set' if s.gemini_api_key else 'MISSING'}, "
          f"pinecone={'set' if s.pinecone_api_key else 'MISSING'}")
