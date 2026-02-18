"""Events streamed to the browser over Server-Sent Events while an agent run executes."""

from typing import Annotated, Any, Literal

from pydantic import BaseModel, Field, TypeAdapter

ToolCallStatus = Literal["ok", "invalid_arguments", "error", "timeout", "rejected", "unknown_tool"]


class Citation(BaseModel):
    kind: Literal["publication", "document"]
    id: str  # publication id, or document id
    marker: str  # "pub:6" or "doc:<id>:<chunk>"
    title: str | None = None
    filename: str | None = None
    page: int | None = None
    chunk_index: int | None = None


class RunStarted(BaseModel):
    type: Literal["run_started"] = "run_started"
    run_id: str
    conversation_id: str
    user_message_id: int | None = None


class TokenDelta(BaseModel):
    type: Literal["token"] = "token"
    text: str
    step: int


class ToolCallStarted(BaseModel):
    type: Literal["tool_call_started"] = "tool_call_started"
    tool_call_id: str
    name: str
    arguments: dict[str, Any] | None  # None when the model's JSON could not be parsed
    step: int


class ToolCallFinished(BaseModel):
    type: Literal["tool_call_finished"] = "tool_call_finished"
    tool_call_id: str
    name: str
    status: ToolCallStatus
    result_preview: str
    duration_ms: int | None = None
    step: int


class ApprovalRequired(BaseModel):
    type: Literal["approval_required"] = "approval_required"
    run_id: str
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    context: dict[str, Any]


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int


class MessageCompleted(BaseModel):
    type: Literal["message_completed"] = "message_completed"
    run_id: str
    message_id: int
    content: str
    citations: list[Citation]
    usage: dict[str, int]
    steps: int


class RunError(BaseModel):
    type: Literal["error"] = "error"
    code: str
    message: str
    run_id: str | None = None


AgentEvent = Annotated[
    RunStarted
    | TokenDelta
    | ToolCallStarted
    | ToolCallFinished
    | ApprovalRequired
    | MessageCompleted
    | RunError,
    Field(discriminator="type"),
]
agent_event_adapter: TypeAdapter[AgentEvent] = TypeAdapter(AgentEvent)


def to_sse(event: BaseModel) -> str:
    return f"event: {event.type}\ndata: {event.model_dump_json()}\n\n"  # type: ignore[attr-defined]
