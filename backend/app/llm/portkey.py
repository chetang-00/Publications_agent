"""Chat completions through the Portkey gateway using the official openai SDK.

Portkey is OpenAI-compatible; the only difference is authentication. The key travels in
``x-portkey-api-key`` (plus an optional ``x-portkey-virtual-key``) and no ``Authorization``
header is sent — the same contract as the Open WebUI Portkey connection.
"""

import logging
from collections.abc import AsyncIterator
from typing import Any
from uuid import uuid4

import openai
from openai import AsyncOpenAI, Omit

from app.config import Settings
from app.llm.types import (
    LLMAuthError,
    LLMBadRequestError,
    LLMResult,
    LLMToolCall,
    LLMUnavailableError,
    TextDelta,
    routing_hint,
)

log = logging.getLogger(__name__)

PORTKEY_API_KEY_HEADER = "x-portkey-api-key"
PORTKEY_VIRTUAL_KEY_HEADER = "x-portkey-virtual-key"
PORTKEY_TRACE_ID_HEADER = "x-portkey-trace-id"

# The SDK insists on an api_key value; it is never sent because every request omits Authorization.
_UNUSED_API_KEY = "unused-portkey-authenticates-by-header"


def portkey_headers(api_key: str, virtual_key: str | None = None) -> dict[str, str]:
    headers = {PORTKEY_API_KEY_HEADER: api_key}
    if virtual_key:
        headers[PORTKEY_VIRTUAL_KEY_HEADER] = virtual_key
    return headers


def create_chat_client(settings: Settings, *, http_client: Any = None, max_retries: int = 3) -> AsyncOpenAI:
    return AsyncOpenAI(
        api_key=_UNUSED_API_KEY,
        base_url=settings.portkey_base_url,
        default_headers=portkey_headers(settings.chat_api_key, settings.chat_virtual_key),
        max_retries=max_retries,
        timeout=60.0,
        http_client=http_client,
    )


class OpenAIChatLLM:
    def __init__(self, settings: Settings, client: AsyncOpenAI | None = None) -> None:
        self.settings = settings
        self.model = settings.chat_model
        self.client = client or create_chat_client(settings)

    async def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        trace_id: str,
        tool_choice: str | None = None,
    ) -> AsyncIterator[TextDelta | LLMResult]:
        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "extra_headers": {"Authorization": Omit(), PORTKEY_TRACE_ID_HEADER: trace_id},
        }
        if self.settings.llm_temperature is not None:
            kwargs["temperature"] = self.settings.llm_temperature
        if tools:
            kwargs["tools"] = tools
            if tool_choice:
                kwargs["tool_choice"] = tool_choice
        if self.settings.llm_stream_usage:
            kwargs["stream_options"] = {"include_usage": True}

        content: list[str] = []
        calls: list[dict[str, Any]] = []
        by_index: dict[int, dict[str, Any]] = {}
        prompt_tokens = completion_tokens = 0
        finish_reason: str | None = None

        try:
            stream = await self.client.chat.completions.create(**kwargs)
            async for chunk in stream:
                if chunk.usage:
                    prompt_tokens = chunk.usage.prompt_tokens or 0
                    completion_tokens = chunk.usage.completion_tokens or 0
                for choice in chunk.choices:
                    delta = choice.delta
                    if delta and delta.content:
                        content.append(delta.content)
                        yield TextDelta(delta.content)
                    for fragment in (delta.tool_calls if delta else None) or []:
                        acc = by_index.get(fragment.index)
                        # Some non-OpenAI routes send every call with index 0; a new id is a new call.
                        if acc is None or (fragment.id and acc["id"] and fragment.id != acc["id"]):
                            acc = {"id": "", "name": "", "arguments": []}
                            calls.append(acc)
                            by_index[fragment.index] = acc
                        if fragment.id:
                            acc["id"] = fragment.id
                        if fragment.function:
                            name = fragment.function.name
                            if name and name != acc["name"]:  # names may stream in pieces or repeat whole
                                acc["name"] += name
                            if fragment.function.arguments:
                                acc["arguments"].append(fragment.function.arguments)
                    if choice.finish_reason:
                        finish_reason = choice.finish_reason
        except (openai.AuthenticationError, openai.PermissionDeniedError) as exc:
            raise LLMAuthError(
                "The Portkey gateway rejected the credentials. Check PORTKEY_API_KEY "
                "(and PORTKEY_VIRTUAL_KEY if your gateway uses virtual keys)."
            ) from exc
        except (openai.BadRequestError, openai.NotFoundError, openai.UnprocessableEntityError) as exc:
            detail = _error_text(exc)
            raise LLMBadRequestError(
                f"The model request was rejected: {detail}.{routing_hint(detail, 'CHAT_MODEL')}"
            ) from exc
        except openai.APIError as exc:
            log.warning("LLM request failed", extra={"error": _error_text(exc)})
            raise LLMUnavailableError(
                "The language model is temporarily unavailable. Please try again shortly."
            ) from exc

        tool_calls = [
            LLMToolCall(
                id=acc["id"] or f"call_{uuid4().hex[:24]}",
                name=acc["name"],
                arguments="".join(acc["arguments"]),
            )
            for acc in calls
        ]
        yield LLMResult(
            content="".join(content),
            tool_calls=tool_calls,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            finish_reason=finish_reason,
        )


def _error_text(exc: openai.APIError) -> str:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        error = body.get("error", body)
        if isinstance(error, dict) and error.get("message"):
            return str(error["message"])[:300]
    return str(exc.message)[:300]
