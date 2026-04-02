"""Embeddings through the Portkey gateway.

Plain httpx rather than the SDK so batching, pacing and retries follow the gateway's limits:
- requests are capped by input count (≤ 2048) and by estimated tokens;
- requests are paced against the tokens-per-minute limit the gateway reports
  (x-ratelimit-limit-tokens), so bulk indexing waits instead of tripping 429s;
- 429/5xx and transport errors are retried, honouring Retry-After. A 429 without Retry-After
  backs off long enough for a one-minute rate window to roll over.
"""

import asyncio
import contextlib
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from typing import Protocol

import httpx

from app.config import Settings
from app.llm.portkey import portkey_headers
from app.llm.types import LLMError, routing_hint

log = logging.getLogger(__name__)

RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_AFTER_SECONDS = 60.0
RATE_LIMIT_BACKOFF_SECONDS = 5.0
RATE_WINDOW_SECONDS = 60.0
RATE_HEADROOM = 0.9  # stay under 90% of the reported limit


def estimate_tokens(text: str) -> int:
    return len(text) // 4 + 1


class TokenWindow:
    """Tokens sent in the last minute, checked against the limit the gateway reports."""

    def __init__(self, clock: Callable[[], float]) -> None:
        self.clock = clock
        self.limit: int | None = None
        self.sent: deque[tuple[float, int]] = deque()

    def observe(self, headers: httpx.Headers) -> None:
        with contextlib.suppress(TypeError, ValueError):
            self.limit = int(headers.get("x-ratelimit-limit-tokens"))

    def record(self, tokens: int) -> None:
        self.sent.append((self.clock(), tokens))

    def wait_time(self, tokens: int) -> float:
        now = self.clock()
        while self.sent and self.sent[0][0] <= now - RATE_WINDOW_SECONDS:
            self.sent.popleft()
        if self.limit is None:
            return 0.0
        budget = self.limit * RATE_HEADROOM
        used = sum(t for _, t in self.sent)
        if used + tokens <= budget:
            return 0.0
        for sent_at, sent_tokens in self.sent:  # wait until enough of the oldest requests age out
            used -= sent_tokens
            if used + tokens <= budget:
                return max(0.0, sent_at + RATE_WINDOW_SECONDS - now)
        return max(0.0, self.sent[-1][0] + RATE_WINDOW_SECONDS - now) if self.sent else 0.0


class EmbeddingError(LLMError):
    code = "embedding_failed"


class EmbeddingAuthError(EmbeddingError):
    code = "llm_auth_failed"


class Embedder(Protocol):
    model: str

    async def embed(self, texts: list[str]) -> list[list[float]]: ...


class PortkeyEmbedder:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        virtual_key: str | None,
        model: str,
        batch_size: int,
        max_batch_tokens: int = 30_000,
        client: httpx.AsyncClient | None = None,
        max_attempts: int = 5,
        backoff_base: float = 1.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.url = f"{base_url.rstrip('/')}/embeddings"
        self.model = model
        self.batch_size = min(batch_size, 2048)
        self.headers = {"Content-Type": "application/json", **portkey_headers(api_key, virtual_key)}
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
        self.max_batch_tokens = max_batch_tokens
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.sleep = sleep
        self.window = TokenWindow(clock)

    @classmethod
    def from_settings(cls, settings: Settings, client: httpx.AsyncClient | None = None) -> "PortkeyEmbedder":
        return cls(
            base_url=settings.portkey_base_url,
            api_key=settings.embedding_api_key,
            virtual_key=settings.embedding_virtual_key,
            model=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
            max_batch_tokens=settings.embedding_max_batch_tokens,
            client=client,
        )

    async def aclose(self) -> None:
        await self.client.aclose()

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError("Embedding inputs must be non-empty strings")
        vectors: list[list[float]] = []
        for batch in self._batches(texts):
            vectors.extend(await self._embed_batch(batch))
        if len(vectors) != len(texts):
            raise EmbeddingError(f"Embedding service returned {len(vectors)} vectors for {len(texts)} inputs")
        return vectors

    def _batches(self, texts: list[str]) -> list[list[str]]:
        batches: list[list[str]] = [[]]
        tokens = 0
        for text in texts:
            cost = estimate_tokens(text)
            current = batches[-1]
            if current and (len(current) >= self.batch_size or tokens + cost > self.max_batch_tokens):
                batches.append([])
                current, tokens = batches[-1], 0
            current.append(text)
            tokens += cost
        return batches

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": batch, "encoding_format": "float"}
        cost = sum(estimate_tokens(t) for t in batch)
        last_problem = "no attempt made"
        for attempt in range(self.max_attempts):
            is_last = attempt == self.max_attempts - 1
            if pause := self.window.wait_time(cost):
                log.info("Pacing embeddings to the gateway token limit", extra={"wait_s": round(pause, 1)})
                await self.sleep(pause)
            try:
                response = await self.client.post(self.url, headers=self.headers, json=payload)
            except httpx.TransportError as exc:
                last_problem = f"transport error {type(exc).__name__}"
                if is_last:
                    break
                await self._wait(attempt, None, last_problem)
                continue

            self.window.observe(response.headers)
            if response.status_code in RETRYABLE_STATUSES:
                last_problem = f"HTTP {response.status_code}"
                if is_last:
                    break
                await self._wait(
                    attempt,
                    response.headers.get("Retry-After"),
                    last_problem,
                    rate_limited=response.status_code == 429,
                )
                continue
            if response.status_code in (401, 403):
                raise EmbeddingAuthError(
                    "The Portkey gateway rejected the embedding credentials. "
                    "Check EMBEDDING_PORTKEY_API_KEY / PORTKEY_API_KEY."
                )
            if response.status_code >= 400:
                detail = _detail(response)
                raise EmbeddingError(
                    f"Embedding request failed (HTTP {response.status_code}): {detail}."
                    f"{routing_hint(detail, 'EMBEDDING_MODEL')}"
                )
            data = response.json().get("data")
            if not isinstance(data, list):
                raise EmbeddingError("Unexpected embeddings response: missing 'data'")
            self.window.record(cost)
            return [item["embedding"] for item in sorted(data, key=lambda item: item.get("index", 0))]

        raise EmbeddingError(f"Embedding request failed after {self.max_attempts} attempts ({last_problem})")

    async def _wait(
        self, attempt: int, retry_after: str | None, problem: str, rate_limited: bool = False
    ) -> None:
        if rate_limited:
            # Token limits are per minute; short retries just hit the same exhausted window.
            delay = min(MAX_RETRY_AFTER_SECONDS, RATE_LIMIT_BACKOFF_SECONDS * (2**attempt))
        else:
            delay = self.backoff_base * (2**attempt)
        if retry_after:
            with contextlib.suppress(ValueError):
                delay = min(float(retry_after), MAX_RETRY_AFTER_SECONDS)
        log.warning(
            "Embedding request retry",
            extra={
                "problem": problem,
                "delay_s": delay,
                "attempt": attempt + 1,
                "max_attempts": self.max_attempts,
            },
        )
        await self.sleep(delay)


def _detail(response: httpx.Response) -> str:
    try:
        body = response.json()
        error = body.get("error", body) if isinstance(body, dict) else body
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:300]
        return str(body)[:300]
    except ValueError:
        return response.text[:300]
