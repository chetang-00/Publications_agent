"""The agent loop.

The model decides every step: answer, call one or more tools, retry with different arguments, or
stop. This code runs what it asks for, enforces the limits (steps, invalid calls, timeouts), pauses
for human approval before any write, verifies citations, and records everything in SQLite so a run
can be inspected or resumed after approval.

    loop:
        stream model response (forward text tokens)
        no tool calls        → verify citations, store answer, done
        tool calls           → validate each (Pydantic), run reads in parallel,
                               pause for approval on a write, feed results back, repeat
        step limit reached   → one last call with tools disabled
"""

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from pydantic import BaseModel

from app.agent import citations as citation_checks
from app.agent.events import (
    ApprovalRequired,
    MessageCompleted,
    RunError,
    RunStarted,
    TokenDelta,
    ToolCallFinished,
    ToolCallStarted,
)
from app.agent.prompts import STEP_LIMIT_NOTE, build_system_prompt
from app.config import Settings
from app.db import repo
from app.db.models import Message
from app.db.session import Database
from app.llm.embeddings import Embedder
from app.llm.types import ChatLLM, LLMError, LLMResult, LLMToolCall, LLMUnavailableError, TextDelta
from app.logging import run_id_var
from app.rag.vectorstore import VectorStore
from app.tools.base import (
    InvalidToolArguments,
    Tool,
    ToolContext,
    ToolError,
    ToolRegistry,
    invalid_arguments_content,
    unknown_tool_content,
)

log = logging.getLogger(__name__)

Emit = Callable[[BaseModel], Awaitable[None]]

MAX_INVALID_TOOL_CALLS = 3
MAX_RAW_ARGUMENT_CHARS = 20_000
FALLBACK_ANSWER = "I couldn't produce an answer this time. Please rephrase the question or try again."
SUPERSEDED_NOTE = "Superseded: the user sent a new message before approving or rejecting this change."
INTERRUPTED_NOTE = "This answer was interrupted by an error and could not finish."
GENERIC_FAILURE = "Something went wrong while answering. Please try again."


class RunInProgress(Exception):
    """The conversation already has a run that is still producing an answer."""


class NothingPending(Exception):
    """There is no tool call awaiting approval with that id."""


@dataclass
class _RunState:
    run_id: str
    conversation_id: str
    step: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    invalid_tool_calls: int = 0


def _parse_object(text: str) -> dict[str, Any] | None:
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        return None
    return value if isinstance(value, dict) else None


def _to_openai(message: Message) -> dict[str, Any]:
    if message.role == "assistant" and message.tool_calls:
        return {"role": "assistant", "content": message.content or None, "tool_calls": message.tool_calls}
    if message.role == "tool":
        return {"role": "tool", "tool_call_id": message.tool_call_id, "content": message.content}
    return {"role": message.role, "content": message.content}


class AgentRunner:
    def __init__(
        self,
        *,
        llm: ChatLLM,
        registry: ToolRegistry,
        db: Database,
        store: VectorStore,
        embedder: Embedder,
        settings: Settings,
        today: Callable[[], date] = date.today,
    ) -> None:
        self.llm = llm
        self.registry = registry
        self.db = db
        self.store = store
        self.embedder = embedder
        self.settings = settings
        self.today = today

    # ── public API ───────────────────────────────────────────────────────────

    async def start(self, conversation_id: str, content: str, emit: Emit) -> str:
        active = await repo.active_run(self.db, conversation_id)
        if active is not None:
            if active.status == "running":
                # RunManager allows one live run per conversation, so a 'running' row seen here was
                # left behind by a failure and can never finish. Close it instead of blocking.
                await repo.update_run(
                    self.db, active.id, status="failed", error_code="interrupted", error=INTERRUPTED_NOTE
                )
            else:
                await self._cancel_pending(active)

        run = await repo.create_run(self.db, conversation_id, self.llm.model)
        state = _RunState(run_id=run.id, conversation_id=conversation_id)
        try:
            user_message = await repo.add_message(
                self.db, conversation_id=conversation_id, run_id=run.id, role="user", content=content
            )
            await repo.title_from_first_message(self.db, conversation_id, content)
            await emit(
                RunStarted(run_id=run.id, conversation_id=conversation_id, user_message_id=user_message.id)
            )
            tracker = await self._tracker(state)
        except Exception:
            log.exception("Agent run failed to start", extra={"run_id": run.id})
            await self._fail(state, "internal_error", GENERIC_FAILURE, emit)
            return run.id
        await self._loop(state, tracker, emit)
        return run.id

    async def resume(
        self, run_id: str, tool_call_id: str, approved: bool, note: str | None, emit: Emit
    ) -> None:
        run = await repo.get_run(self.db, run_id)
        if run is None or run.status != "awaiting_approval":
            raise NothingPending("This run is not waiting for an approval.")
        pending = await repo.pending_tool_call(self.db, run_id)
        if pending is None or pending.call_id != tool_call_id:
            raise NothingPending("No tool call with that id is waiting for approval.")

        await repo.update_run(self.db, run_id, status="running")
        state = _RunState(
            run_id=run.id,
            conversation_id=run.conversation_id,
            step=run.steps,
            prompt_tokens=run.prompt_tokens,
            completion_tokens=run.completion_tokens,
            invalid_tool_calls=run.invalid_tool_calls,
        )
        token = run_id_var.set(run.id)
        try:
            tracker = await self._tracker(state)
            await self._apply_decision(state, pending, approved, note, tracker, emit)
        except Exception:
            log.exception("Applying the approval decision failed")
            await self._fail(state, "internal_error", GENERIC_FAILURE, emit)
            return
        finally:
            run_id_var.reset(token)
        await self._loop(state, tracker, emit)

    async def _apply_decision(
        self,
        state: _RunState,
        pending: Any,
        approved: bool,
        note: str | None,
        tracker: citation_checks.CitationTracker,
        emit: Emit,
    ) -> None:
        tool = self.registry.get(pending.name)
        if approved and tool is not None:
            args = tool.args_model.model_validate(pending.arguments or {})
            outcome = await self.registry.execute(tool, args, self._ctx(state))
            status, content, preview = outcome.status, outcome.content(), outcome.preview()
            result, error, duration = outcome.result, outcome.error, outcome.duration_ms
        else:
            reason = (note or "").strip() or "The user rejected this change."
            status, content, preview = (
                "rejected",
                json.dumps({"status": "rejected", "note": reason}),
                f"Rejected: {reason}",
            )
            result, error, duration = None, reason, None
        log.info("Approval decided", extra={"tool": pending.name, "approved": approved, "status": status})

        await repo.update_tool_call(
            self.db,
            state.run_id,
            pending.step,
            pending.call_id,
            status=status,
            result=result,
            error=error,
            duration_ms=duration,
        )
        await repo.add_message(
            self.db,
            conversation_id=state.conversation_id,
            run_id=state.run_id,
            role="tool",
            tool_call_id=pending.call_id,
            content=content,
        )
        if status == "ok":
            tracker.record_tool_result(pending.name, result)
        await emit(
            ToolCallFinished(
                tool_call_id=pending.call_id,
                name=pending.name,
                status=status,
                result_preview=preview,
                duration_ms=duration,
                step=pending.step,
            )
        )

    # ── the loop ─────────────────────────────────────────────────────────────

    async def _loop(self, state: _RunState, tracker: citation_checks.CitationTracker, emit: Emit) -> None:
        token = run_id_var.set(state.run_id)
        ctx = self._ctx(state)
        try:
            while True:
                if state.step >= self.settings.agent_max_steps:
                    await self._answer_without_tools(state, tracker, emit)
                    return
                state.step += 1
                result = await self._call_model(state, emit)
                if not result.tool_calls:
                    await self._finish(state, result.content, tracker, emit)
                    return

                await repo.add_message(
                    self.db,
                    conversation_id=state.conversation_id,
                    run_id=state.run_id,
                    role="assistant",
                    content=result.content,
                    tool_calls=[
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {"name": c.name, "arguments": c.arguments},
                        }
                        for c in result.tool_calls
                    ],
                )
                paused = await self._run_tool_calls(state, result.tool_calls, ctx, tracker, emit)
                await self._save_progress(state)
                if paused:
                    return
                if state.invalid_tool_calls > MAX_INVALID_TOOL_CALLS:
                    await self._fail(
                        state,
                        "too_many_invalid_tool_calls",
                        "The assistant kept calling tools with invalid arguments, so the answer was stopped. "
                        "Try rephrasing the question.",
                        emit,
                    )
                    return
        except LLMError as exc:
            log.warning("Model call failed", extra={"code": exc.code})
            await self._fail(state, exc.code, exc.message, emit)
        except Exception:
            log.exception("Agent run failed")
            await self._fail(
                state, "internal_error", "Something went wrong while answering. Please try again.", emit
            )
        finally:
            run_id_var.reset(token)

    async def _call_model(
        self,
        state: _RunState,
        emit: Emit,
        tool_choice: str | None = None,
        extra_messages: Sequence[dict[str, Any]] = (),
    ) -> LLMResult:
        messages = [*await self._build_messages(state), *extra_messages]
        result: LLMResult | None = None
        async for item in self.llm.stream(
            messages, self.registry.schemas(), trace_id=state.run_id, tool_choice=tool_choice
        ):
            if isinstance(item, TextDelta):
                await emit(TokenDelta(text=item.text, step=state.step))
            elif isinstance(item, LLMResult):
                result = item
        if result is None:
            raise LLMUnavailableError("The language model returned no response.")
        state.prompt_tokens += result.prompt_tokens
        state.completion_tokens += result.completion_tokens
        return result

    async def _answer_without_tools(
        self, state: _RunState, tracker: citation_checks.CitationTracker, emit: Emit
    ) -> None:
        state.step += 1
        log.info("Step limit reached; requesting final answer", extra={"steps": state.step - 1})
        result = await self._call_model(
            state, emit, tool_choice="none", extra_messages=[{"role": "system", "content": STEP_LIMIT_NOTE}]
        )
        await self._finish(state, result.content, tracker, emit)

    # ── tool calls ───────────────────────────────────────────────────────────

    async def _run_tool_calls(
        self,
        state: _RunState,
        calls: list[LLMToolCall],
        ctx: ToolContext,
        tracker: citation_checks.CitationTracker,
        emit: Emit,
    ) -> bool:
        """Run one step's tool calls. Returns True when the run paused for approval."""
        tool_messages: dict[str, str] = {}
        reads: list[tuple[LLMToolCall, Tool, BaseModel]] = []
        writes: list[tuple[LLMToolCall, Tool, BaseModel]] = []

        for c in calls:
            await emit(
                ToolCallStarted(
                    tool_call_id=c.id, name=c.name, arguments=_parse_object(c.arguments), step=state.step
                )
            )
            tool = self.registry.get(c.name)
            if tool is None:
                tool_messages[c.id] = unknown_tool_content(c.name, self.registry.names())
                await self._record(state, c, "unknown_tool", error=f"Unknown tool '{c.name}'")
                await self._finished(emit, state, c, "unknown_tool", f"Unknown tool '{c.name}'")
                continue
            try:
                args = self.registry.parse_args(tool, c.arguments)
            except InvalidToolArguments as exc:
                state.invalid_tool_calls += 1
                tool_messages[c.id] = invalid_arguments_content(exc.details)
                summary = "; ".join(f"{d['loc']}: {d['msg']}" for d in exc.details)
                await self._record(state, c, "invalid_arguments", error=summary)
                await self._finished(emit, state, c, "invalid_arguments", summary)
                continue
            (writes if tool.requires_approval else reads).append((c, tool, args))

        outcomes = await asyncio.gather(*(self.registry.execute(tool, args, ctx) for _, tool, args in reads))
        for (c, tool, args), outcome in zip(reads, outcomes, strict=True):
            tool_messages[c.id] = outcome.content()
            await self._record(
                state,
                c,
                outcome.status,
                arguments=args.model_dump(mode="json"),
                result=outcome.result,
                error=outcome.error,
                duration_ms=outcome.duration_ms,
            )
            if outcome.status == "ok":
                tracker.record_tool_result(tool.name, outcome.result)
            log.info(
                "Tool finished",
                extra={"tool": tool.name, "status": outcome.status, "duration_ms": outcome.duration_ms},
            )
            await self._finished(emit, state, c, outcome.status, outcome.preview(), outcome.duration_ms)

        pending: tuple[LLMToolCall, Tool, BaseModel, dict[str, Any]] | None = None
        for c, tool, args in writes:
            try:
                if pending is not None:
                    raise ToolError(
                        "Only one change can be approved at a time. Propose this change again after the "
                        "current one has been approved or rejected."
                    )
                context = await tool.approval_context(args, ctx) if tool.approval_context else {}
            except ToolError as exc:
                tool_messages[c.id] = json.dumps({"status": "error", "error": exc.message})
                await self._record(
                    state, c, "error", arguments=args.model_dump(mode="json"), error=exc.message
                )
                await self._finished(emit, state, c, "error", exc.message)
                continue
            pending = (c, tool, args, context)

        # Tool messages are stored in the model's call order; the pending call's message is added on resume.
        for c in calls:
            if c.id in tool_messages:
                await repo.add_message(
                    self.db,
                    conversation_id=state.conversation_id,
                    run_id=state.run_id,
                    role="tool",
                    tool_call_id=c.id,
                    content=tool_messages[c.id],
                )

        if pending is None:
            return False
        c, tool, args, context = pending
        arguments = args.model_dump(mode="json")
        await self._record(state, c, "awaiting_approval", arguments=arguments)
        await self._save_progress(state, status="awaiting_approval")
        log.info("Awaiting approval", extra={"tool": tool.name})
        await emit(
            ApprovalRequired(
                run_id=state.run_id, tool_call_id=c.id, name=tool.name, arguments=arguments, context=context
            )
        )
        return True

    async def _record(
        self,
        state: _RunState,
        c: LLMToolCall,
        status: str,
        *,
        arguments: dict[str, Any] | None = None,
        result: dict[str, Any] | None = None,
        error: str | None = None,
        duration_ms: int | None = None,
    ) -> None:
        await repo.record_tool_call(
            self.db,
            run_id=state.run_id,
            call_id=c.id,
            step=state.step,
            name=c.name,
            raw_arguments=c.arguments[:MAX_RAW_ARGUMENT_CHARS],
            arguments=arguments if arguments is not None else _parse_object(c.arguments),
            status=status,
            result=result,
            error=error,
            duration_ms=duration_ms,
        )

    @staticmethod
    async def _finished(
        emit: Emit,
        state: _RunState,
        c: LLMToolCall,
        status: str,
        preview: str,
        duration_ms: int | None = None,
    ) -> None:
        await emit(
            ToolCallFinished(
                tool_call_id=c.id,
                name=c.name,
                status=status,  # type: ignore[arg-type]
                result_preview=preview[:500],
                duration_ms=duration_ms,
                step=state.step,
            )
        )

    # ── endings ──────────────────────────────────────────────────────────────

    async def _finish(
        self, state: _RunState, content: str, tracker: citation_checks.CitationTracker, emit: Emit
    ) -> None:
        cleaned, citations, unverified = tracker.finalize(content)
        if unverified:
            log.warning("Removed unverified citations", extra={"markers": unverified})
        if not cleaned:
            cleaned = FALLBACK_ANSWER
        message = await repo.add_message(
            self.db,
            conversation_id=state.conversation_id,
            run_id=state.run_id,
            role="assistant",
            content=cleaned,
            citations=[c.model_dump() for c in citations],
        )
        await self._save_progress(state, status="completed")
        await emit(
            MessageCompleted(
                run_id=state.run_id,
                message_id=message.id,
                content=cleaned,
                citations=citations,
                usage={"prompt_tokens": state.prompt_tokens, "completion_tokens": state.completion_tokens},
                steps=state.step,
            )
        )

    async def _fail(self, state: _RunState, code: str, message: str, emit: Emit) -> None:
        await self._save_progress(state, status="failed", error_code=code, error=message)
        await emit(RunError(code=code, message=message, run_id=state.run_id))

    async def _cancel_pending(self, run: Any) -> None:
        pending = await repo.pending_tool_call(self.db, run.id)
        if pending is not None:
            await repo.update_tool_call(
                self.db, run.id, pending.step, pending.call_id, status="rejected", error=SUPERSEDED_NOTE
            )
            await repo.add_message(
                self.db,
                conversation_id=run.conversation_id,
                run_id=run.id,
                role="tool",
                tool_call_id=pending.call_id,
                content=json.dumps({"status": "rejected", "note": SUPERSEDED_NOTE}),
            )
        await repo.update_run(
            self.db, run.id, status="cancelled", error_code="superseded", error=SUPERSEDED_NOTE
        )

    # ── helpers ──────────────────────────────────────────────────────────────

    async def _save_progress(self, state: _RunState, **values: Any) -> None:
        await repo.update_run(
            self.db,
            state.run_id,
            steps=state.step,
            prompt_tokens=state.prompt_tokens,
            completion_tokens=state.completion_tokens,
            invalid_tool_calls=state.invalid_tool_calls,
            **values,
        )

    async def _tracker(self, state: _RunState) -> citation_checks.CitationTracker:
        tracker = citation_checks.CitationTracker()
        tracker.record_citations(await repo.earlier_citations(self.db, state.conversation_id))
        for record in await repo.run_tool_calls(self.db, [state.run_id]):
            if record.status == "ok":
                tracker.record_tool_result(record.name, record.result)
        return tracker

    async def _build_messages(self, state: _RunState) -> list[dict[str, Any]]:
        attached = await repo.attached_documents(self.db, state.conversation_id)
        system = build_system_prompt([(d.id, d.filename, d.status) for d in attached], self.today())
        history = await repo.history_messages(self.db, state.conversation_id, state.run_id)
        current = await repo.run_messages(self.db, state.run_id)
        return [{"role": "system", "content": system}, *(_to_openai(m) for m in [*history, *current])]

    def _ctx(self, state: _RunState) -> ToolContext:
        return ToolContext(
            db=self.db,
            store=self.store,
            embedder=self.embedder,
            settings=self.settings,
            conversation_id=state.conversation_id,
            run_id=state.run_id,
        )
