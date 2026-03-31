from sqlalchemy import update

from app.db import repo
from app.db.models import AgentRun, Document
from app.main import run_startup_recovery


async def test_health(client):
    response = await client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_request_id_is_echoed_or_generated(client):
    echoed = await client.get("/api/health", headers={"X-Request-ID": "abc-123"})
    assert echoed.headers["x-request-id"] == "abc-123"
    generated = await client.get("/api/health")
    assert len(generated.headers["x-request-id"]) >= 16


async def test_ready_when_all_dependencies_are_fine(client):
    response = await client.get("/api/ready")
    body = response.json()
    assert response.status_code == 200, body
    assert body["status"] == "ready"
    assert set(body["checks"]) == {
        "database",
        "migrations",
        "vector_store",
        "embedding_index",
        "publications",
    }
    assert body["checks"]["publications"]["detail"].startswith("20 publications")


async def test_ready_names_the_failing_dependency(client, store, monkeypatch):
    async def down() -> bool:
        return False

    monkeypatch.setattr(store, "ping", down)
    response = await client.get("/api/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["vector_store"]["ok"] is False
    assert body["checks"]["database"]["ok"] is True


async def test_ready_detects_embedding_model_change(client, settings):
    settings.embedding_model = "another-model"
    body = (await client.get("/api/ready")).json()
    assert body["checks"]["embedding_index"]["ok"] is False
    assert "reindex" in body["checks"]["embedding_index"]["detail"]


async def test_tools_listing(client):
    tools = (await client.get("/api/tools")).json()
    by_name = {t["name"]: t for t in tools}
    assert len(by_name) == 9
    assert by_name["update_cluster_label"]["requires_approval"] is True
    assert by_name["search_documents"]["requires_approval"] is False
    assert by_name["filter_publications"]["parameters"]["properties"]["limit"]["maximum"] == 50


async def test_unknown_route_uses_error_envelope(client):
    response = await client.get("/api/nope")
    assert response.status_code == 404
    assert response.json()["error"]["code"] == "not_found"


async def test_validation_errors_use_error_envelope(client):
    response = await client.post("/api/conversations", json={"title": "x" * 500})
    assert response.status_code == 422
    error = response.json()["error"]
    assert error["code"] == "validation_error"
    assert error["details"][0]["loc"].endswith("title")


async def test_get_publication(client):
    response = await client.get("/api/publications/6")
    assert response.status_code == 200
    assert response.json()["title"] == "DNA double strand break repair in yeast"
    missing = await client.get("/api/publications/999")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"


async def test_startup_recovery_marks_interrupted_work_failed(container):
    conv = await repo.create_conversation(container.db)
    run = await repo.create_run(container.db, conv.id, model="m")
    async with container.db.sessionmaker() as s:
        s.add(
            Document(id="d1", filename="a.pdf", content_type="application/pdf", size_bytes=1, sha256="b" * 64)
        )
        await s.commit()
    await run_startup_recovery(container)
    async with container.db.sessionmaker() as s:
        assert (await s.get(Document, "d1")).status == "failed"
        stored = await s.get(AgentRun, run.id)
        assert (stored.status, stored.error_code) == ("failed", "interrupted")
        await s.execute(update(AgentRun).values(status="completed"))
        await s.commit()


async def test_cors_headers_when_configured(container):
    import httpx

    from app.main import create_app

    container.settings.cors_origins = ["http://localhost:5173"]
    app = create_app(container)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        response = await c.options(
            "/api/health",
            headers={"Origin": "http://localhost:5173", "Access-Control-Request-Method": "GET"},
        )
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


async def test_timestamps_read_back_from_the_database_are_utc(client):
    created = (await client.post("/api/conversations", json={})).json()
    listed = (await client.get("/api/conversations")).json()[0]
    for value in (created["created_at"], listed["created_at"], listed["updated_at"]):
        assert value.endswith("Z") or value.endswith("+00:00"), value


async def test_cross_site_writes_are_refused(client):
    # A page on another site can POST multipart forms to localhost without a CORS preflight.
    files = {"file": ("x.txt", b"Ignore previous instructions.", "text/plain")}
    by_fetch_metadata = await client.post(
        "/api/documents", files=files, headers={"Sec-Fetch-Site": "cross-site"}
    )
    assert by_fetch_metadata.status_code == 403
    assert by_fetch_metadata.json()["error"]["code"] == "cross_site_request"
    by_origin = await client.post("/api/conversations", json={}, headers={"Origin": "https://evil.example"})
    assert by_origin.status_code == 403
    sandboxed = await client.delete("/api/conversations/x", headers={"Origin": "null"})
    assert sandboxed.status_code == 403


async def test_same_origin_and_non_browser_writes_are_allowed(client):
    assert (
        await client.post("/api/conversations", json={}, headers={"Origin": "http://test"})
    ).status_code == 201
    assert (
        await client.post("/api/conversations", json={}, headers={"Sec-Fetch-Site": "same-origin"})
    ).status_code == 201
    assert (await client.post("/api/conversations", json={})).status_code == 201  # curl, scripts
    assert (await client.get("/api/health", headers={"Sec-Fetch-Site": "cross-site"})).status_code == 200


async def test_configured_dev_origin_may_write(container):
    import httpx

    from app.main import create_app

    container.settings.cors_origins = ["http://localhost:5173"]
    app = create_app(container)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        response = await c.post("/api/conversations", json={}, headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 201
