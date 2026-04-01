"""Embeddings through the Portkey gateway.

Plain httpx rather than the SDK so batching and retries follow the gateway's limits: at most
2048 inputs per request, and 429/5xx responses retried with Retry-After or exponential backoff
(document indexing sends sustained bursts that trip gateway rate limits).
"""

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Protocol

import httpx

from app.config import Settings
from app.llm.portkey import portkey_headers
from app.llm.types import LLMError, routing_hint

log = logging.getLogger(__name__)

RETRYABLE_STATUSES = frozenset({429, 500, 502, 503, 504})
MAX_RETRY_AFTER_SECONDS = 60.0


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
        client: httpx.AsyncClient | None = None,
        max_attempts: int = 5,
        backoff_base: float = 1.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self.url = f"{base_url.rstrip('/')}/embeddings"
        self.model = model
        self.batch_size = min(batch_size, 2048)
        self.headers = {"Content-Type": "application/json", **portkey_headers(api_key, virtual_key)}
        self.client = client or httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0))
        self.max_attempts = max_attempts
        self.backoff_base = backoff_base
        self.sleep = sleep

    @classmethod
    def from_settings(cls, settings: Settings, client: httpx.AsyncClient | None = None) -> "PortkeyEmbedder":
        return cls(
            base_url=settings.portkey_base_url,
            api_key=settings.embedding_api_key,
            virtual_key=settings.embedding_virtual_key,
            model=settings.embedding_model,
            batch_size=settings.embedding_batch_size,
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
        for start in range(0, len(texts), self.batch_size):
            vectors.extend(await self._embed_batch(texts[start : start + self.batch_size]))
        if len(vectors) != len(texts):
            raise EmbeddingError(f"Embedding service returned {len(vectors)} vectors for {len(texts)} inputs")
        return vectors

    async def _embed_batch(self, batch: list[str]) -> list[list[float]]:
        payload = {"model": self.model, "input": batch, "encoding_format": "float"}
        last_problem = "no attempt made"
        for attempt in range(self.max_attempts):
            is_last = attempt == self.max_attempts - 1
            try:
                response = await self.client.post(self.url, headers=self.headers, json=payload)
            except httpx.TransportError as exc:
                last_problem = f"transport error {type(exc).__name__}"
                if is_last:
                    break
                await self._wait(attempt, None, last_problem)
                continue

            if response.status_code in RETRYABLE_STATUSES:
                last_problem = f"HTTP {response.status_code}"
                if is_last:
                    break
                await self._wait(attempt, response.headers.get("Retry-After"), last_problem)
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
            return [item["embedding"] for item in sorted(data, key=lambda item: item.get("index", 0))]

        raise EmbeddingError(f"Embedding request failed after {self.max_attempts} attempts ({last_problem})")

    async def _wait(self, attempt: int, retry_after: str | None, problem: str) -> None:
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
