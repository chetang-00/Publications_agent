import asyncio

from tests.api.conftest import new_conversation, parse_sse
from tests.fakes import FakeTurn, call


async def test_conversation_crud(client):
    conv_id = await new_conversation(client)
    listed = (await client.get("/api/conversations")).json()
    assert [c["id"] for c in listed] == [conv_id]
    assert listed[0]["title"] == "New conversation"

    detail = (await client.get(f"/api/conversations/{conv_id}")).json()
    assert detail["conversation"]["id"] == conv_id
    assert detail["messages"] == [] and detail["runs"] == [] and detail["documents"] == []

    assert (await client.delete(f"/api/conversations/{conv_id}")).status_code == 204
    assert (await client.get(f"/api/conversations/{conv_id}")).status_code == 404
    assert (await client.delete(f"/api/conversations/{conv_id}")).status_code == 404


async def test_message_streams_a_validated_event_sequence(client, llm):
    llm.add(
        FakeTurn(tool_calls=[call("c1", "get_publication", {"publication_id": 6})]),
        FakeTurn(text="It is about yeast DNA repair [pub:6]."),
    )
    conv_id = await new_conversation(client)
    response = await client.post(
        f"/api/conversations/{conv_id}/messages", json={"content": "What is paper 6?"}
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["cache-control"] == "no-cache"
    events = parse_sse(response.text)
    assert [e.type for e in events] == [
        "run_started",
        "tool_call_started",
        "tool_call_finished",
        "token",
        "token",
        "message_completed",
    ]
    assert events[-1].citations[0].title == "DNA double strand break repair in yeast"

    detail = (await client.get(f"/api/conversations/{conv_id}")).json()
    assert detail["conversation"]["title"] == "What is paper 6?"
    assert [(m["role"], m["content"]) for m in detail["messages"]] == [
        ("user", "What is paper 6?"),
        ("assistant", "It is about yeast DNA repair [pub:6]."),
    ]
    assert detail["messages"][1]["citations"][0]["id"] == "6"
    (run,) = detail["runs"]
    assert (run["status"], run["steps"], run["pending_approval"]) == ("completed", 2, None)
    (tool_call,) = detail["tool_calls"]
    assert (tool_call["name"], tool_call["status"], tool_call["run_id"]) == (
        "get_publication",
        "ok",
        run["id"],
    )
    assert "yeast" in tool_call["result_preview"]


async def test_message_to_unknown_conversation_is_404(client):
    response = await client.post("/api/conversations/missing/messages", json={"content": "hi"})
    assert response.status_code == 404


async def test_blank_message_is_rejected(client):
    conv_id = await new_conversation(client)
    response = await client.post(f"/api/conversations/{conv_id}/messages", json={"content": "   "})
    assert response.status_code == 422
    too_long = await client.post(f"/api/conversations/{conv_id}/messages", json={"content": "x" * 8001})
    assert too_long.status_code == 422


async def test_concurrent_message_returns_409(client, llm):
    llm.add(FakeTurn(text="slow answer", delay=0.3))
    conv_id = await new_conversation(client)
    first, second = await asyncio.gather(
        client.post(f"/api/conversations/{conv_id}/messages", json={"content": "one"}),
        client.post(f"/api/conversations/{conv_id}/messages", json={"content": "two"}),
    )
    statuses = sorted([first.status_code, second.status_code])
    assert statuses == [200, 409]
    rejected = first if first.status_code == 409 else second
    assert rejected.json()["error"]["code"] == "run_in_progress"
    detail = (await client.get(f"/api/conversations/{conv_id}")).json()
    assert [m["role"] for m in detail["messages"]] == ["user", "assistant"]


async def test_cannot_delete_a_conversation_while_it_is_answering(client, llm):
    llm.add(FakeTurn(text="slow", delay=0.3))
    conv_id = await new_conversation(client)
    sending = asyncio.create_task(
        client.post(f"/api/conversations/{conv_id}/messages", json={"content": "hi"})
    )
    await asyncio.sleep(0.1)
    assert (await client.delete(f"/api/conversations/{conv_id}")).status_code == 409
    await sending


async def test_llm_failure_is_streamed_as_error_event(client, llm):
    from app.llm.types import LLMAuthError

    llm.add(FakeTurn(error=LLMAuthError("Check PORTKEY_API_KEY")))
    conv_id = await new_conversation(client)
    events = parse_sse(
        (await client.post(f"/api/conversations/{conv_id}/messages", json={"content": "hi"})).text
    )
    assert [e.type for e in events] == ["run_started", "error"]
    assert events[-1].code == "llm_auth_failed"
    run = (await client.get(f"/api/conversations/{conv_id}")).json()["runs"][0]
    assert (run["status"], run["error_code"]) == ("failed", "llm_auth_failed")
