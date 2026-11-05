"""All Gemini calls funnel through here.

One place for the client, retries, and JSON parsing means the rest of the
codebase never touches the SDK directly -- so swapping model provider later
is a change to this file only.
"""

from __future__ import annotations

import json
import logging
import re
import time
from collections import deque
from typing import Any

from google import genai
from google.genai import types

from config import Settings

log = logging.getLogger(__name__)

# Document and query vectors are embedded ASYMMETRICALLY. A question and the
# passage that answers it are not paraphrases, so embedding them with different
# task types puts them closer together than symmetric embedding would.
TASK_DOCUMENT = "RETRIEVAL_DOCUMENT"
TASK_QUERY = "RETRIEVAL_QUERY"

# gemini-embedding-001 accepts exactly ONE input per request -- the SDK splits
# a list into N separate API calls. So "batch size" here is only about how many
# we hand the SDK at a time; the request count equals the number of texts, and
# that is what the free-tier quota counts. Hence the rate limiter below.
EMBED_BATCH_SIZE = 32

# "Please retry in 15.98s" / 'retryDelay': '15s'
_RETRY_AFTER = re.compile(r"(?:retry in|retryDelay['\"]?\s*:\s*['\"]?)\s*(\d+(?:\.\d+)?)s")


def _retry_after(exc: Exception) -> float | None:
    """Pull the server's own suggested wait out of a 429 message."""
    match = _RETRY_AFTER.search(str(exc))
    return float(match.group(1)) if match else None


class _RateLimiter:
    """Sliding-window limiter. Embedding is the only thing that gets near the
    free-tier ceiling, so this guards embed calls specifically."""

    def __init__(self, per_minute: int):
        self.per_minute = per_minute
        self._times: deque[float] = deque()

    def acquire(self) -> None:
        now = time.monotonic()
        while self._times and now - self._times[0] > 60.0:
            self._times.popleft()
        if len(self._times) >= self.per_minute:
            wait = 60.0 - (now - self._times[0]) + 0.1
            if wait > 0:
                log.info("Embedding rate limit reached; pausing %.1fs", wait)
                time.sleep(wait)
                return self.acquire()
        self._times.append(time.monotonic())


# Process-wide, deliberately. The quota is per API key, not per client object,
# and run_pipeline builds a fresh GeminiClient every run -- a per-instance
# limiter starts from zero each time and walks straight into the ceiling.
_SHARED_LIMITER: _RateLimiter | None = None


def _shared_limiter(per_minute: int) -> _RateLimiter:
    global _SHARED_LIMITER
    if _SHARED_LIMITER is None or _SHARED_LIMITER.per_minute != per_minute:
        _SHARED_LIMITER = _RateLimiter(per_minute)
    return _SHARED_LIMITER


class LLMError(RuntimeError):
    pass


class GeminiClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.client = genai.Client(api_key=settings.gemini_api_key)
        self._limiter = _shared_limiter(settings.embed_requests_per_minute)

    # ------------------------------------------------------------------
    # generation
    # ------------------------------------------------------------------
    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        schema: dict | None = None,
        temperature: float = 0.2,
    ) -> str | dict:
        """Generate text, or structured JSON when `schema` is given.

        Low temperature by default: this is an extraction and synthesis
        pipeline, not a creative one.
        """
        model = model or self.settings.model_synthesis
        config: dict[str, Any] = {
            "temperature": temperature,
            # We never pass tools, so the SDK's automatic function calling is
            # dead weight -- and it warns on every call. Turn it off.
            "automatic_function_calling": types.AutomaticFunctionCallingConfig(
                disable=True, maximum_remote_calls=None
            ),
        }
        if schema is not None:
            config["response_mime_type"] = "application/json"
            config["response_schema"] = schema

        response = self._with_retries(
            lambda: self.client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(**config),
            )
        )
        text = (response.text or "").strip()
        if schema is None:
            return text
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMError(f"Model returned invalid JSON: {text[:200]}") from exc

    # ------------------------------------------------------------------
    # embeddings
    # ------------------------------------------------------------------
    def embed(self, texts: list[str], *, task_type: str) -> list[list[float]]:
        """Embed a list of texts. Batches automatically.

        task_type must be TASK_DOCUMENT (indexing) or TASK_QUERY (searching).
        """
        if task_type not in (TASK_DOCUMENT, TASK_QUERY):
            raise ValueError(f"Unexpected task_type: {task_type}")
        if not texts:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start : start + EMBED_BATCH_SIZE]
            # One request per text, so reserve that many slots.
            for _ in batch:
                self._limiter.acquire()
            result = self._with_retries(
                lambda b=batch: self.client.models.embed_content(
                    model=self.settings.model_embedding,
                    contents=b,
                    config=types.EmbedContentConfig(
                        task_type=task_type,
                        output_dimensionality=self.settings.embedding_dim,
                    ),
                )
            )
            vectors.extend(e.values for e in result.embeddings)
        return vectors

    def embed_one(self, text: str, *, task_type: str) -> list[float]:
        return self.embed([text], task_type=task_type)[0]

    # ------------------------------------------------------------------
    # infrastructure
    # ------------------------------------------------------------------
    def list_models(self) -> list[str]:
        """Used by test_setup.py to catch stale model IDs in config.py."""
        return [m.name.removeprefix("models/") for m in self.client.models.list()]

    def _with_retries(self, fn):
        """Exponential backoff. Rate limits and transient 5xx are normal at
        free-tier quotas -- crashing on them would make the app feel broken."""
        delay = 1.0
        last: Exception | None = None
        for attempt in range(self.settings.max_retries):
            try:
                return fn()
            except Exception as exc:  # SDK raises a wide range; treat all as retryable once
                last = exc
                if attempt == self.settings.max_retries - 1:
                    break
                # A 429 tells us exactly how long to wait -- believe it rather
                # than guessing with backoff.
                suggested = _retry_after(exc)
                wait = suggested + 1.0 if suggested is not None else delay
                log.warning("Gemini call failed (attempt %d), retrying in %.1fs: %s",
                            attempt + 1, wait, str(exc)[:120])
                time.sleep(wait)
                delay *= 2
        raise LLMError(f"Gemini call failed after {self.settings.max_retries} attempts: {last}")
