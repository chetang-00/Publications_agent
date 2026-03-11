"""Request and response models for the HTTP API."""

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agent.events import Citation


class ApiModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


# ── conversations ──


class CreateConversationIn(BaseModel):
    title: str | None = Field(None, max_length=200)


class ConversationOut(ApiModel):
    id: str
    title: str
    created_at: datetime
    updated_at: datetime


class SendMessageIn(BaseModel):
    content: str = Field(min_length=1, max_length=8000)

    @field_validator("content")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("message must not be blank")
        return value


class MessageOut(ApiModel):
    id: int
    run_id: str | None
    role: Literal["user", "assistant"]
    content: str
    citations: list[Citation] = []
    created_at: datetime


class PendingApprovalOut(BaseModel):
    run_id: str
    tool_call_id: str
    name: str
    arguments: dict[str, Any]
    context: dict[str, Any]


class RunOut(ApiModel):
    id: str
    status: str
    steps: int
    model: str
    prompt_tokens: int
    completion_tokens: int
    error_code: str | None
    error: str | None
    started_at: datetime
    finished_at: datetime | None
    pending_approval: PendingApprovalOut | None = None


class ToolCallOut(BaseModel):
    run_id: str
    call_id: str
    step: int
    name: str
    arguments: dict[str, Any] | None
    status: str
    result_preview: str | None
    error: str | None
    duration_ms: int | None


class DocumentOut(ApiModel):
    id: str
    filename: str
    content_type: str
    size_bytes: int
    status: str
    error: str | None
    page_count: int | None
    chunk_count: int | None
    created_at: datetime


class DocumentChunkOut(BaseModel):
    document_id: str
    filename: str
    chunk_index: int
    page: int | None
    text: str


class ConversationDetailOut(BaseModel):
    conversation: ConversationOut
    messages: list[MessageOut]
    runs: list[RunOut]
    tool_calls: list[ToolCallOut]
    documents: list[DocumentOut]


class AttachDocumentsIn(BaseModel):
    document_ids: list[str] = Field(max_length=50)


class AttachDocumentsOut(BaseModel):
    document_ids: list[str]


# ── runs ──


class ApprovalIn(BaseModel):
    tool_call_id: str = Field(min_length=1, max_length=128)
    approved: bool
    note: str | None = Field(None, max_length=500)


# ── misc ──


class ToolInfoOut(BaseModel):
    name: str
    description: str
    requires_approval: bool
    parameters: dict[str, Any]


class HealthOut(BaseModel):
    status: Literal["ok"] = "ok"


class CheckOut(BaseModel):
    ok: bool
    detail: str


class ReadinessOut(BaseModel):
    status: Literal["ready", "not_ready"]
    checks: dict[str, CheckOut]
