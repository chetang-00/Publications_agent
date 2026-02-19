from fastapi import APIRouter, Response
from fastapi.responses import StreamingResponse

from app.agent.loop import RunInProgress
from app.api.deps import ContainerDep
from app.api.errors import ApiError, not_found
from app.api.schemas import (
    AttachDocumentsIn,
    AttachDocumentsOut,
    ConversationDetailOut,
    ConversationOut,
    CreateConversationIn,
    DocumentOut,
    MessageOut,
    PendingApprovalOut,
    RunOut,
    SendMessageIn,
    ToolCallOut,
)
from app.container import Container
from app.db import repo
from app.db.models import AgentRun, ToolCallRecord
from app.tools.base import ToolContext, ToolError

router = APIRouter(prefix="/conversations", tags=["conversations"])

SSE_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}


def sse_response(stream) -> StreamingResponse:
    return StreamingResponse(stream, media_type="text/event-stream", headers=SSE_HEADERS)


def _preview(record: ToolCallRecord) -> str | None:
    import json

    if record.result is not None:
        text = json.dumps(record.result, ensure_ascii=False, default=str)
        return text if len(text) <= 500 else text[:499] + "…"
    return record.error


async def _pending_approval(c: Container, run: AgentRun) -> PendingApprovalOut | None:
    if run.status != "awaiting_approval":
        return None
    pending = await repo.pending_tool_call(c.db, run.id)
    if pending is None:
        return None
    tool = c.registry.get(pending.name)
    context: dict = {}
    if tool is not None and tool.approval_context is not None:
        ctx = ToolContext(
            db=c.db,
            store=c.store,
            embedder=c.embedder,
            settings=c.settings,
            conversation_id=run.conversation_id,
            run_id=run.id,
        )
        try:
            context = await tool.approval_context(
                tool.args_model.model_validate(pending.arguments or {}), ctx
            )
        except ToolError as exc:
            context = {"error": exc.message}
    return PendingApprovalOut(
        run_id=run.id,
        tool_call_id=pending.call_id,
        name=pending.name,
        arguments=pending.arguments or {},
        context=context,
    )


@router.post("", status_code=201, response_model=ConversationOut)
async def create_conversation(body: CreateConversationIn, c: ContainerDep) -> ConversationOut:
    return ConversationOut.model_validate(await repo.create_conversation(c.db, body.title))


@router.get("", response_model=list[ConversationOut])
async def list_conversations(c: ContainerDep) -> list[ConversationOut]:
    return [ConversationOut.model_validate(conv) for conv in await repo.list_conversations(c.db)]


@router.get("/{conversation_id}", response_model=ConversationDetailOut)
async def get_conversation(conversation_id: str, c: ContainerDep) -> ConversationDetailOut:
    conversation = await repo.get_conversation(c.db, conversation_id)
    if conversation is None:
        raise not_found("Conversation")
    messages = [
        m
        for m in await repo.conversation_messages(c.db, conversation_id)
        if m.role == "user" or (m.role == "assistant" and not m.tool_calls)
    ]
    runs = await repo.conversation_runs(c.db, conversation_id)
    run_out = []
    for run in runs:
        out = RunOut.model_validate(run)
        out.pending_approval = await _pending_approval(c, run)
        run_out.append(out)
    tool_calls = await repo.run_tool_calls(c.db, [r.id for r in runs])
    return ConversationDetailOut(
        conversation=ConversationOut.model_validate(conversation),
        messages=[
            MessageOut(
                id=m.id,
                run_id=m.run_id,
                role=m.role,  # type: ignore[arg-type]
                content=m.content,
                citations=m.citations or [],
                created_at=m.created_at,
            )
            for m in messages
        ],
        runs=run_out,
        tool_calls=[
            ToolCallOut(
                run_id=t.run_id,
                call_id=t.call_id,
                step=t.step,
                name=t.name,
                arguments=t.arguments,
                status=t.status,
                result_preview=_preview(t),
                error=t.error,
                duration_ms=t.duration_ms,
            )
            for t in tool_calls
        ],
        documents=[
            DocumentOut.model_validate(d) for d in await repo.attached_documents(c.db, conversation_id)
        ],
    )


@router.delete("/{conversation_id}", status_code=204)
async def delete_conversation(conversation_id: str, c: ContainerDep) -> Response:
    if c.runs.is_busy(conversation_id):
        raise ApiError(409, "run_in_progress", "Wait for the current answer to finish before deleting.")
    if not await repo.delete_conversation(c.db, conversation_id):
        raise not_found("Conversation")
    return Response(status_code=204)


@router.post("/{conversation_id}/documents", response_model=AttachDocumentsOut)
async def attach_documents(
    conversation_id: str, body: AttachDocumentsIn, c: ContainerDep
) -> AttachDocumentsOut:
    if await repo.get_conversation(c.db, conversation_id) is None:
        raise not_found("Conversation")
    return AttachDocumentsOut(
        document_ids=await repo.set_attached_documents(c.db, conversation_id, body.document_ids)
    )


@router.post(
    "/{conversation_id}/messages",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "Server-Sent Events of the agent run"}
    },
)
async def send_message(conversation_id: str, body: SendMessageIn, c: ContainerDep) -> StreamingResponse:
    if await repo.get_conversation(c.db, conversation_id) is None:
        raise not_found("Conversation")
    try:
        stream = c.runs.launch(
            conversation_id, lambda emit: c.runner.start(conversation_id, body.content, emit)
        )
    except RunInProgress as exc:
        raise ApiError(409, "run_in_progress", str(exc)) from exc
    return sse_response(stream)
