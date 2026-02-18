import asyncio
import json
import time

import pytest
from pydantic import BaseModel
from sqlalchemy import select

from app.agent.loop import AgentRunner, NothingPending, RunInProgress
from app.db import repo
from app.db.models import (
    AgentRun,
    ClusterLabelChange,
    ConversationDocument,
    Document,
    Message,
    Publication,
    ToolCallRecord,
)
from app.llm.types import LLMAuthError, LLMUnavailableError
from app.tools import build_registry
from app.tools.base import Tool, ToolArgs, ToolRegistry
from tests.fakes import FakeLLM, FakeTurn, call


class Events(list):
    async def __call__(self, event) -> None:
        self.append(event)

    def types(self) -> list[str]:
        return [e.type for e in self]

    def of(self, kind: str) -> list:
        return [e for e in self if e.type == kind]


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
def make_runner(seeded, store, embedder, settings, llm):
    def make(registry: ToolRegistry | None = None) -> AgentRunner:
        return AgentRunner(
            llm=llm,
            registry=registry or build_registry(),
            db=seeded,
            store=store,
            embedder=embedder,
            settings=settings,
        )

    return make


@pytest.fixture
def runner(make_runner) -> AgentRunner:
    return make_runner()


@pytest.fixture
async def conversation(seeded) -> str:
    return (await repo.create_conversation(seeded)).id


async def get_run(db, run_id) -> AgentRun:
    async with db.sessionmaker() as s:
        return await s.get(AgentRun, run_id)


async def tool_records(db, run_id) -> list[ToolCallRecord]:
    async with db.sessionmaker() as s:
        return list(
            (
                await s.execute(
                    select(ToolCallRecord).where(ToolCallRecord.run_id == run_id).order_by(ToolCallRecord.id)
                )
            ).scalars()
        )


async def label_of(db, pub_id) -> str | None:
    async with db.sessionmaker() as s:
        return (await s.get(Publication, pub_id)).cluster_label


# ── extra tools for timing tests ──


class NoArgs(ToolArgs):
    pass


class Value(BaseModel):
    value: int


def sleeper(name: str, seconds: float, timeout: float | None = None) -> Tool:
    async def handler(args, ctx):
        await asyncio.sleep(seconds)
        return Value(value=1)

    return Tool(
        name=name,
        description=name,
        args_model=NoArgs,
        result_model=Value,
        handler=handler,
        timeout_seconds=timeout,
    )


# ── 1. plain answer ──


async def test_answer_without_tools(runner, llm, conversation, seeded):
    llm.add(FakeTurn(text="CRISPR is a genome editing technique."))
    events = Events()
    run_id = await runner.start(conversation, "What is CRISPR?", events)

    assert events.types() == ["run_started", "token", "token", "message_completed"]
    done = events.of("message_completed")[0]
    assert done.content == "CRISPR is a genome editing technique."
    assert done.citations == []
    assert done.usage == {"prompt_tokens": 10, "completion_tokens": 5}
    run = await get_run(seeded, run_id)
    assert (run.status, run.steps, run.prompt_tokens) == ("completed", 1, 10)

    request = llm.requests[0]
    assert request["trace_id"] == run_id
    assert request["messages"][0]["role"] == "system"
    assert request["messages"][-1] == {"role": "user", "content": "What is CRISPR?"}
    assert {t["function"]["name"] for t in request["tools"]} >= {"search_publications", "search_documents"}


async def test_conversation_title_comes_from_first_message(runner, llm, conversation, seeded):
    llm.add(FakeTurn(text="Hi."))
    await runner.start(conversation, "  Which journals publish most CRISPR work?  ", Events())
    convo = await repo.get_conversation(seeded, conversation)
    assert convo.title == "Which journals publish most CRISPR work?"


# ── 2. single tool call ──


async def test_single_tool_call_then_answer(runner, llm, conversation, seeded):
    llm.add(
        FakeTurn(tool_calls=[call("c1", "get_publication", {"publication_id": 6})]),
        FakeTurn(text="It studies double strand break repair in yeast [pub:6]."),
    )
    events = Events()
    run_id = await runner.start(conversation, "Tell me about paper 6", events)

    assert events.types() == [
        "run_started",
        "tool_call_started",
        "tool_call_finished",
        "token",
        "token",
        "message_completed",
    ]
    started, finished = events.of("tool_call_started")[0], events.of("tool_call_finished")[0]
    assert (started.tool_call_id, started.name, started.arguments) == (
        "c1",
        "get_publication",
        {"publication_id": 6},
    )
    assert (finished.status, finished.name) == ("ok", "get_publication")
    assert "yeast" in finished.result_preview
    citation = events.of("message_completed")[0].citations[0]
    assert (citation.kind, citation.id, citation.title) == (
        "publication",
        "6",
        "DNA double strand break repair in yeast",
    )

    second_request = llm.requests[1]["messages"]
    assert second_request[-2]["tool_calls"][0]["function"]["name"] == "get_publication"
    assert second_request[-1]["role"] == "tool" and second_request[-1]["tool_call_id"] == "c1"
    assert "double strand break" in second_request[-1]["content"]

    (record,) = await tool_records(seeded, run_id)
    assert (record.status, record.step, record.arguments) == ("ok", 1, {"publication_id": 6})
    assert record.result["title"] == "DNA double strand break repair in yeast"


# ── 3. parallel tools ──


async def test_parallel_tool_calls_run_concurrently(make_runner, llm, conversation):
    runner = make_runner(ToolRegistry([sleeper("slow_a", 0.3), sleeper("slow_b", 0.3)]))
    llm.add(FakeTurn(tool_calls=[call("a", "slow_a", {}), call("b", "slow_b", {})]), FakeTurn(text="done"))
    started = time.perf_counter()
    events = Events()
    await runner.start(conversation, "go", events)
    assert time.perf_counter() - started < 0.55
    assert [e.status for e in events.of("tool_call_finished")] == ["ok", "ok"]


# ── 4. multi-step chain ──


async def test_multi_step_chain(runner, llm, conversation, seeded):
    llm.add(
        FakeTurn(tool_calls=[call("c1", "resolve_author", {"name": "Rajesh Ranganath"})]),
        FakeTurn(
            tool_calls=[
                call(
                    "c2",
                    "publication_stats",
                    {"group_by": "keyword", "author": "Ranganath R.", "year_to": 2019},
                ),
                call(
                    "c3",
                    "publication_stats",
                    {"group_by": "keyword", "author": "Ranganath R.", "year_from": 2020},
                ),
            ]
        ),
        FakeTurn(
            tool_calls=[
                call("c4", "search_publications", {"query": "diffusion models", "authors": ["Ranganath R."]})
            ]
        ),
        FakeTurn(text="Earlier work focused on causal inference [pub:3]; recent work on diffusion [pub:4]."),
    )
    events = Events()
    run_id = await runner.start(conversation, "How did Ranganath's topics shift?", events)
    assert [e.name for e in events.of("tool_call_started")] == [
        "resolve_author",
        "publication_stats",
        "publication_stats",
        "search_publications",
    ]
    assert all(e.status == "ok" for e in events.of("tool_call_finished"))
    assert {c.id for c in events.of("message_completed")[0].citations} == {"3", "4"}
    assert (await get_run(seeded, run_id)).steps == 4


# ── 5/6/16. invalid arguments ──


async def test_invalid_arguments_are_returned_and_corrected(runner, llm, conversation):
    llm.add(
        FakeTurn(tool_calls=[call("c1", "filter_publications", {"limit": 500})]),
        FakeTurn(tool_calls=[call("c2", "filter_publications", {"limit": 5})]),
        FakeTurn(text="Here are five papers."),
    )
    events = Events()
    await runner.start(conversation, "list papers", events)
    assert [e.status for e in events.of("tool_call_finished")] == ["invalid_arguments", "ok"]
    tool_message = llm.requests[1]["messages"][-1]
    content = json.loads(tool_message["content"])
    assert content["error"] == "invalid_arguments"
    assert content["details"][0]["loc"] == "limit"
    assert events.types()[-1] == "message_completed"


async def test_non_json_arguments_are_invalid_arguments(runner, llm, conversation):
    llm.add(
        FakeTurn(tool_calls=[call("c1", "get_publication", '{"publication_id": 6')]),
        FakeTurn(text="Sorry, let me answer directly."),
    )
    events = Events()
    await runner.start(conversation, "paper 6?", events)
    finished = events.of("tool_call_finished")[0]
    assert finished.status == "invalid_arguments"
    assert events.of("tool_call_started")[0].arguments is None
    assert events.types()[-1] == "message_completed"


async def test_too_many_invalid_calls_fail_the_run(runner, llm, conversation, seeded):
    for i in range(4):
        llm.add(FakeTurn(tool_calls=[call(f"c{i}", "get_publication", {"publication_id": 0})]))
    events = Events()
    run_id = await runner.start(conversation, "x", events)
    error = events.of("error")[0]
    assert error.code == "too_many_invalid_tool_calls"
    run = await get_run(seeded, run_id)
    assert (run.status, run.error_code) == ("failed", "too_many_invalid_tool_calls")


# ── 7. unknown tool ──


async def test_unknown_tool_is_reported_to_the_model(runner, llm, conversation):
    llm.add(FakeTurn(tool_calls=[call("c1", "delete_everything", {})]), FakeTurn(text="I can't do that."))
    events = Events()
    await runner.start(conversation, "delete all", events)
    assert events.of("tool_call_finished")[0].status == "unknown_tool"
    content = json.loads(llm.requests[1]["messages"][-1]["content"])
    assert content["error"] == "unknown_tool"
    assert "search_publications" in content["available_tools"]
    assert events.types()[-1] == "message_completed"


# ── 8. timeout ──


async def test_tool_timeout_does_not_stop_the_run(make_runner, llm, conversation):
    runner = make_runner(ToolRegistry([sleeper("stuck", 5, timeout=0.05)]))
    llm.add(FakeTurn(tool_calls=[call("c1", "stuck", {})]), FakeTurn(text="The tool timed out."))
    events = Events()
    await runner.start(conversation, "go", events)
    assert events.of("tool_call_finished")[0].status == "timeout"
    assert events.types()[-1] == "message_completed"


# ── 9. step limit ──


async def test_step_limit_forces_a_final_answer(runner, llm, conversation, settings, seeded):
    settings.agent_max_steps = 2
    llm.add(
        FakeTurn(tool_calls=[call("c1", "filter_publications", {"keyword": "CRISPR"})]),
        FakeTurn(tool_calls=[call("c2", "filter_publications", {"keyword": "DNA"})]),
        FakeTurn(text="Based on what I found: three CRISPR papers [pub:8]."),
    )
    events = Events()
    run_id = await runner.start(conversation, "x", events)
    final_request = llm.requests[2]
    assert final_request["tool_choice"] == "none"
    assert "step limit" in final_request["messages"][-1]["content"].lower()
    assert events.of("message_completed")[0].citations[0].id == "8"
    assert (await get_run(seeded, run_id)).status == "completed"


# ── 10-14. approval flow ──


async def request_label_change(runner, llm, conversation, events=None):
    llm.add(
        FakeTurn(
            text="I'll propose a label change.",
            tool_calls=[
                call(
                    "w1",
                    "update_cluster_label",
                    {"publication_id": 11, "new_label": "Astrophysics", "reason": "about galaxies"},
                )
            ],
        )
    )
    events = events if events is not None else Events()
    run_id = await runner.start(conversation, "Label paper 11 as Astrophysics", events)
    return run_id, events


async def test_write_tool_pauses_for_approval(runner, llm, conversation, seeded):
    run_id, events = await request_label_change(runner, llm, conversation)
    assert events.types()[-1] == "approval_required"
    approval = events.of("approval_required")[0]
    assert (approval.run_id, approval.tool_call_id, approval.name) == (run_id, "w1", "update_cluster_label")
    assert approval.arguments == {
        "publication_id": 11,
        "new_label": "Astrophysics",
        "reason": "about galaxies",
    }
    assert approval.context["current_label"] is None
    assert approval.context["title"] == "Galaxy formation in dark matter halos"
    assert "message_completed" not in events.types()
    assert (await get_run(seeded, run_id)).status == "awaiting_approval"
    assert await label_of(seeded, 11) is None
    (record,) = await tool_records(seeded, run_id)
    assert record.status == "awaiting_approval"


async def test_approve_executes_and_resumes(runner, llm, conversation, seeded):
    run_id, _ = await request_label_change(runner, llm, conversation)
    llm.add(FakeTurn(text="Done: paper 11 is now labelled Astrophysics [pub:11]."))
    events = Events()
    await runner.resume(run_id, "w1", approved=True, note=None, emit=events)
    assert events.types()[0] == "tool_call_finished"
    assert events.of("tool_call_finished")[0].status == "ok"
    assert events.types()[-1] == "message_completed"
    assert events.of("message_completed")[0].citations[0].id == "11"
    assert await label_of(seeded, 11) == "Astrophysics"
    async with seeded.sessionmaker() as s:
        change = (await s.execute(select(ClusterLabelChange))).scalar_one()
    assert change.run_id == run_id
    assert (await get_run(seeded, run_id)).status == "completed"


async def test_reject_leaves_data_unchanged(runner, llm, conversation, seeded):
    run_id, _ = await request_label_change(runner, llm, conversation)
    llm.add(FakeTurn(text="Understood, I left the label unchanged."))
    events = Events()
    await runner.resume(run_id, "w1", approved=False, note="Use 'Astronomy' instead", emit=events)
    assert events.of("tool_call_finished")[0].status == "rejected"
    tool_message = json.loads(llm.requests[1]["messages"][-1]["content"])
    assert tool_message["status"] == "rejected"
    assert "Astronomy" in tool_message["note"]
    assert await label_of(seeded, 11) is None
    (record,) = await tool_records(seeded, run_id)
    assert record.status == "rejected"


async def test_new_message_while_awaiting_approval_cancels_the_pending_change(
    runner, llm, conversation, seeded
):
    old_run, _ = await request_label_change(runner, llm, conversation)
    llm.add(FakeTurn(text="Sure, what else?"))
    events = Events()
    await runner.start(conversation, "Actually, never mind.", events)
    assert events.types()[-1] == "message_completed"
    assert (await get_run(seeded, old_run)).status == "cancelled"
    (record,) = await tool_records(seeded, old_run)
    assert record.status == "rejected"
    assert await label_of(seeded, 11) is None
    roles = [m["role"] for m in llm.requests[-1]["messages"]]
    assert "tool" not in roles


async def test_read_tools_run_before_pausing_for_a_write(runner, llm, conversation):
    llm.add(
        FakeTurn(
            tool_calls=[
                call("r1", "get_publication", {"publication_id": 11}),
                call(
                    "w1",
                    "update_cluster_label",
                    {"publication_id": 11, "new_label": "Astrophysics", "reason": "galaxies"},
                ),
            ]
        )
    )
    events = Events()
    await runner.start(conversation, "check and relabel 11", events)
    assert events.types() == [
        "run_started",
        "tool_call_started",
        "tool_call_started",
        "tool_call_finished",
        "approval_required",
    ]
    assert events.of("tool_call_finished")[0].tool_call_id == "r1"


async def test_only_one_write_is_approved_at_a_time(runner, llm, conversation):
    llm.add(
        FakeTurn(
            tool_calls=[
                call(
                    "w1", "update_cluster_label", {"publication_id": 11, "new_label": "A", "reason": "first"}
                ),
                call(
                    "w2", "update_cluster_label", {"publication_id": 12, "new_label": "B", "reason": "second"}
                ),
            ]
        )
    )
    events = Events()
    await runner.start(conversation, "relabel both", events)
    second = events.of("tool_call_finished")[0]
    assert (second.tool_call_id, second.status) == ("w2", "error")
    assert events.of("approval_required")[0].tool_call_id == "w1"


async def test_approval_context_error_does_not_pause(runner, llm, conversation):
    llm.add(
        FakeTurn(
            tool_calls=[
                call(
                    "w1", "update_cluster_label", {"publication_id": 999, "new_label": "X", "reason": "test"}
                )
            ]
        ),
        FakeTurn(text="That paper does not exist."),
    )
    events = Events()
    await runner.start(conversation, "relabel 999", events)
    assert "approval_required" not in events.types()
    assert events.of("tool_call_finished")[0].status == "error"
    assert events.types()[-1] == "message_completed"


async def test_resume_requires_a_matching_pending_call(runner, llm, conversation):
    run_id, _ = await request_label_change(runner, llm, conversation)
    with pytest.raises(NothingPending):
        await runner.resume(run_id, "not-the-call", approved=True, note=None, emit=Events())
    with pytest.raises(NothingPending):
        await runner.resume("no-such-run", "w1", approved=True, note=None, emit=Events())


async def test_start_refuses_while_a_run_is_in_progress(runner, conversation, seeded):
    await repo.create_run(seeded, conversation, model="m")
    with pytest.raises(RunInProgress):
        await runner.start(conversation, "hello", Events())


# ── 15. LLM failures ──


@pytest.mark.parametrize(
    ("error", "code"),
    [(LLMAuthError("bad key"), "llm_auth_failed"), (LLMUnavailableError("down"), "llm_unavailable")],
)
async def test_llm_errors_fail_the_run_with_a_code(runner, llm, conversation, seeded, error, code):
    llm.add(FakeTurn(error=error))
    events = Events()
    run_id = await runner.start(conversation, "hi", events)
    assert events.types() == ["run_started", "error"]
    assert events.of("error")[0].code == code
    run = await get_run(seeded, run_id)
    assert (run.status, run.error_code) == ("failed", code)


async def test_unexpected_exception_is_reported_without_internals(runner, llm, conversation):
    llm.add(FakeTurn(error=RuntimeError("secret stack detail")))
    events = Events()
    await runner.start(conversation, "hi", events)
    error = events.of("error")[0]
    assert error.code == "internal_error"
    assert "secret" not in error.message


# ── answers and citations ──


async def test_empty_completion_produces_fallback(runner, llm, conversation, seeded):
    llm.add(FakeTurn(text=""))
    events = Events()
    await runner.start(conversation, "hmm", events)
    done = events.of("message_completed")[0]
    assert "couldn't produce an answer" in done.content
    async with seeded.sessionmaker() as s:
        last = (await s.execute(select(Message).order_by(Message.id.desc()).limit(1))).scalar_one()
    assert last.content == done.content


async def test_unverified_citations_are_stripped(runner, llm, conversation):
    llm.add(FakeTurn(text="A famous paper [pub:999] says so."))
    events = Events()
    await runner.start(conversation, "x", events)
    done = events.of("message_completed")[0]
    assert done.content == "A famous paper says so."
    assert done.citations == []


async def test_history_keeps_answers_but_drops_earlier_tool_chatter(runner, llm, conversation):
    llm.add(
        FakeTurn(tool_calls=[call("c1", "get_publication", {"publication_id": 6})]),
        FakeTurn(text="Paper 6 is about yeast [pub:6]."),
        FakeTurn(text="It was published in Cell [pub:6]."),
    )
    await runner.start(conversation, "What is paper 6?", Events())
    events = Events()
    await runner.start(conversation, "Where was it published?", events)
    messages = llm.requests[-1]["messages"]
    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[2]["content"] == "Paper 6 is about yeast [pub:6]."
    # The citation from the earlier answer is still trusted.
    assert events.of("message_completed")[0].citations[0].id == "6"


async def test_attached_documents_are_named_in_the_system_prompt(runner, llm, conversation, seeded):
    async with seeded.sessionmaker() as s:
        s.add(
            Document(
                id="d1",
                filename="trial.pdf",
                content_type="application/pdf",
                size_bytes=1,
                sha256="a" * 64,
                status="ready",
            )
        )
        await s.flush()
        s.add(ConversationDocument(conversation_id=conversation, document_id="d1"))
        await s.commit()
    llm.add(FakeTurn(text="ok"))
    await runner.start(conversation, "summarise my document", Events())
    system = llm.requests[0]["messages"][0]["content"]
    assert "trial.pdf" in system and "d1" in system
