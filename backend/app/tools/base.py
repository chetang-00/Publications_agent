"""Tool framework: Pydantic models in, JSON schemas out, every call validated both ways.

A tool is an async handler plus an `Args` model (what the LLM must send) and a `Result` model
(what goes back to the LLM). The registry renders the OpenAI `tools` list from the `Args`
models, validates the LLM's raw JSON arguments, runs the handler with a timeout, and validates
the result before it is serialised for the model.
"""

import asyncio
import json
import logging
import time
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from app.config import Settings
from app.db.session import Database
from app.llm.embeddings import Embedder
from app.rag.vectorstore import VectorStore

log = logging.getLogger(__name__)

MAX_TOOL_CONTENT_CHARS = 12_000
PREVIEW_CHARS = 500


class ToolArgs(BaseModel):
    """Base for tool argument models: unknown fields are an error, not silently dropped."""

    model_config = ConfigDict(extra="forbid")

    @model_validator(mode="before")
    @classmethod
    def _null_means_default(cls, data: Any) -> Any:
        # Models often send null for optional arguments; treat it as "not given" when the field has
        # a non-null default, instead of spending one of the run's invalid-call allowances.
        if not isinstance(data, dict):
            return data

        def has_default(name: str) -> bool:
            field = cls.model_fields.get(name)
            return (
                field is not None
                and not field.is_required()
                and (field.default_factory is not None or field.default is not None)
            )

        return {k: v for k, v in data.items() if not (v is None and has_default(k))}


class ToolError(Exception):
    """An expected failure whose message is safe and useful to show the LLM."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message


class InvalidToolArguments(Exception):
    def __init__(self, details: list[dict[str, str]]) -> None:
        super().__init__(f"{len(details)} invalid argument(s)")
        self.details = details


@dataclass
class ToolContext:
    db: Database
    store: VectorStore
    embedder: Embedder
    settings: Settings
    conversation_id: str | None = None
    run_id: str | None = None


@dataclass
class Tool[A: BaseModel, R: BaseModel]:
    name: str
    description: str
    args_model: type[A]
    result_model: type[R]
    handler: Callable[[A, ToolContext], Awaitable[R]]
    requires_approval: bool = False
    timeout_seconds: float | None = None
    # For approval-gated tools: extra facts shown on the approval card (e.g. the current value).
    approval_context: Callable[[A, ToolContext], Awaitable[dict[str, Any]]] | None = None

    def openai_schema(self) -> dict[str, Any]:
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": _strip_titles(self.args_model.model_json_schema()),
            },
        }


def _strip_titles(node: Any) -> Any:
    """Drop Pydantic's auto-generated `title` keys; they cost tokens and tell the model nothing."""
    if isinstance(node, list):
        return [_strip_titles(item) for item in node]
    if not isinstance(node, dict):
        return node
    cleaned: dict[str, Any] = {}
    for key, value in node.items():
        if key == "title" and isinstance(value, str):
            continue
        if key in ("properties", "$defs") and isinstance(value, dict):
            cleaned[key] = {name: _strip_titles(schema) for name, schema in value.items()}
        else:
            cleaned[key] = _strip_titles(value)
    return cleaned


@dataclass
class ToolOutcome:
    status: Literal["ok", "error", "timeout"]
    result: dict[str, Any] | None
    error: str | None
    duration_ms: int

    def content(self) -> str:
        """The JSON text sent back to the LLM as the tool message, capped in size."""
        payload = self.result if self.status == "ok" else {"status": self.status, "error": self.error}
        return _cap(json.dumps(payload, ensure_ascii=False, default=str))

    def preview(self) -> str:
        text = self.content() if self.status == "ok" else (self.error or self.status)
        return text if len(text) <= PREVIEW_CHARS else text[: PREVIEW_CHARS - 1] + "…"


def _cap(text: str) -> str:
    if len(text) <= MAX_TOOL_CONTENT_CHARS:
        return text
    cut = MAX_TOOL_CONTENT_CHARS - 400
    while True:
        capped = json.dumps(
            {
                "truncated": True,
                "note": "The result was too large and has been cut. Narrow the query (filters, limit) for complete data.",
                "partial": text[:cut],
            },
            ensure_ascii=False,
        )
        if len(capped) <= MAX_TOOL_CONTENT_CHARS:
            return capped
        cut = int(cut * 0.8)


def invalid_arguments_content(details: list[dict[str, str]]) -> str:
    return json.dumps(
        {
            "error": "invalid_arguments",
            "details": details,
            "hint": "Fix the arguments so they match the tool's JSON schema, then call the tool again.",
        }
    )


def unknown_tool_content(name: str, available: list[str]) -> str:
    return json.dumps({"error": "unknown_tool", "tool": name, "available_tools": available})


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = ()) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            self.register(tool)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def tools(self) -> list[Tool]:
        return list(self._tools.values())

    def schemas(self) -> list[dict[str, Any]]:
        return [tool.openai_schema() for tool in self._tools.values()]

    def parse_args(self, tool: Tool, raw: str) -> BaseModel:
        try:
            return tool.args_model.model_validate_json(raw.strip() or "{}")
        except ValidationError as exc:
            raise InvalidToolArguments(
                [
                    {
                        "loc": ".".join(str(part) for part in err["loc"]) or "(root)",
                        "msg": err["msg"],
                        "type": err["type"],
                    }
                    for err in exc.errors()
                ]
            ) from None

    async def execute(self, tool: Tool, args: BaseModel, ctx: ToolContext) -> ToolOutcome:
        timeout = tool.timeout_seconds or ctx.settings.tool_timeout_seconds
        started = time.perf_counter()

        def elapsed() -> int:
            return int((time.perf_counter() - started) * 1000)

        try:
            async with asyncio.timeout(timeout):
                raw_result = await tool.handler(args, ctx)
        except TimeoutError:
            return ToolOutcome("timeout", None, f"Tool '{tool.name}' timed out after {timeout:g}s", elapsed())
        except ToolError as exc:
            return ToolOutcome("error", None, exc.message, elapsed())
        except Exception:
            log.exception("Tool failed", extra={"tool": tool.name})
            return ToolOutcome("error", None, f"Tool '{tool.name}' failed unexpectedly.", elapsed())

        try:
            data = raw_result.model_dump() if isinstance(raw_result, BaseModel) else raw_result
            result = tool.result_model.model_validate(data).model_dump(mode="json")
        except ValidationError:
            log.exception("Tool returned an invalid result", extra={"tool": tool.name})
            return ToolOutcome("error", None, f"Tool '{tool.name}' produced an invalid result.", elapsed())
        return ToolOutcome("ok", result, None, elapsed())
