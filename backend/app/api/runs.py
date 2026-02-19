from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.agent.loop import RunInProgress
from app.api.conversations import sse_response
from app.api.deps import ContainerDep
from app.api.errors import ApiError, not_found
from app.api.schemas import ApprovalIn
from app.db import repo

router = APIRouter(prefix="/runs", tags=["runs"])


@router.post(
    "/{run_id}/approvals",
    response_class=StreamingResponse,
    responses={
        200: {"content": {"text/event-stream": {}}, "description": "Server-Sent Events of the resumed run"}
    },
)
async def decide_approval(run_id: str, body: ApprovalIn, c: ContainerDep) -> StreamingResponse:
    run = await repo.get_run(c.db, run_id)
    if run is None:
        raise not_found("Run")
    pending = await repo.pending_tool_call(c.db, run_id)
    if run.status != "awaiting_approval" or pending is None or pending.call_id != body.tool_call_id:
        raise ApiError(409, "nothing_pending", "There is no change with that id waiting for approval.")
    try:
        stream = c.runs.launch(
            run.conversation_id,
            lambda emit: c.runner.resume(run_id, body.tool_call_id, body.approved, body.note, emit),
        )
    except RunInProgress as exc:
        raise ApiError(409, "run_in_progress", str(exc)) from exc
    return sse_response(stream)
