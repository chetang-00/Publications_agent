from tests.api.conftest import new_conversation, parse_sse
from tests.fakes import FakeTurn, call


async def ask_for_label_change(client, llm) -> tuple[str, str]:
    llm.add(
        FakeTurn(
            tool_calls=[
                call(
                    "w1",
                    "update_cluster_label",
                    {"publication_id": 11, "new_label": "Astrophysics", "reason": "galaxies"},
                )
            ]
        )
    )
    conv_id = await new_conversation(client)
    events = parse_sse(
        (await client.post(f"/api/conversations/{conv_id}/messages", json={"content": "relabel 11"})).text
    )
    assert events[-1].type == "approval_required"
    return conv_id, events[-1].run_id


async def test_pending_approval_is_visible_after_reload(client, llm):
    conv_id, run_id = await ask_for_label_change(client, llm)
    run = (await client.get(f"/api/conversations/{conv_id}")).json()["runs"][0]
    assert run["status"] == "awaiting_approval"
    pending = run["pending_approval"]
    assert (pending["run_id"], pending["tool_call_id"], pending["name"]) == (
        run_id,
        "w1",
        "update_cluster_label",
    )
    assert pending["context"]["title"] == "Galaxy formation in dark matter halos"
    assert pending["arguments"]["new_label"] == "Astrophysics"


async def test_approve_over_http(client, llm):
    _, run_id = await ask_for_label_change(client, llm)
    llm.add(FakeTurn(text="Updated [pub:11]."))
    response = await client.post(
        f"/api/runs/{run_id}/approvals", json={"tool_call_id": "w1", "approved": True}
    )
    assert response.status_code == 200
    events = parse_sse(response.text)
    assert events[0].type == "tool_call_finished"
    assert events[0].status == "ok"
    assert events[-1].type == "message_completed"
    assert (await client.get("/api/publications/11")).json()["cluster_label"] == "Astrophysics"


async def test_reject_over_http(client, llm):
    _, run_id = await ask_for_label_change(client, llm)
    llm.add(FakeTurn(text="Okay, unchanged."))
    response = await client.post(
        f"/api/runs/{run_id}/approvals", json={"tool_call_id": "w1", "approved": False, "note": "wrong label"}
    )
    events = parse_sse(response.text)
    assert events[0].status == "rejected"
    assert (await client.get("/api/publications/11")).json()["cluster_label"] is None


async def test_approval_conflicts(client, llm):
    _, run_id = await ask_for_label_change(client, llm)
    wrong_call = await client.post(
        f"/api/runs/{run_id}/approvals", json={"tool_call_id": "nope", "approved": True}
    )
    assert wrong_call.status_code == 409
    assert wrong_call.json()["error"]["code"] == "nothing_pending"
    llm.add(FakeTurn(text="done"))
    await client.post(f"/api/runs/{run_id}/approvals", json={"tool_call_id": "w1", "approved": True})
    again = await client.post(f"/api/runs/{run_id}/approvals", json={"tool_call_id": "w1", "approved": True})
    assert again.status_code == 409
    unknown = await client.post("/api/runs/missing/approvals", json={"tool_call_id": "w1", "approved": True})
    assert unknown.status_code == 404
