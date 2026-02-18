"""Provider-neutral types the agent loop uses to talk to a chat model."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class TextDelta:
    text: str


@dataclass(frozen=True)
class LLMToolCall:
    id: str
    name: str
    arguments: str  # raw JSON text exactly as the model produced it


@dataclass
class LLMResult:
    content: str
    tool_calls: list[LLMToolCall] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    finish_reason: str | None = None


class LLMError(Exception):
    code = "llm_error"

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class LLMAuthError(LLMError):
    code = "llm_auth_failed"


class LLMUnavailableError(LLMError):
    code = "llm_unavailable"


class LLMBadRequestError(LLMError):
    code = "llm_bad_request"


class ChatLLM(Protocol):
    model: str

    def stream(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None,
        trace_id: str,
        tool_choice: str | None = None,
    ) -> AsyncIterator[TextDelta | LLMResult]:
        """Yield TextDelta items as text arrives, then exactly one LLMResult.

        `tool_choice="none"` keeps the tool definitions in the request but forbids calling them.
        """
        ...
