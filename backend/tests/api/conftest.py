import json
from collections.abc import AsyncIterator

import httpx
import pytest

from app.agent.events import agent_event_adapter
from app.agent.loop import AgentRunner
from app.agent.streaming import RunManager
from app.container import Container
from app.main import create_app
from app.rag.ingest import IngestionService
from app.tasks import TaskSet
from app.tools import build_registry
from tests.fakes import FakeLLM


@pytest.fixture
def llm() -> FakeLLM:
    return FakeLLM()


@pytest.fixture
async def container(seeded, store, embedder, settings, llm) -> AsyncIterator[Container]:
    settings.embedding_model = embedder.model  # matches what the seeded index was built with
    registry = build_registry()
    tasks = TaskSet()
    runner = AgentRunner(
        llm=llm, registry=registry, db=seeded, store=store, embedder=embedder, settings=settings
    )
    yield Container(
        settings=settings,
        db=seeded,
        store=store,
        embedder=embedder,
        llm=llm,
        registry=registry,
        ingestion=IngestionService(seeded, store, embedder, settings.upload_dir, settings.max_upload_bytes),
        runner=runner,
        runs=RunManager(tasks, keepalive_seconds=0.5),
        tasks=tasks,
    )
    await tasks.wait_all()


@pytest.fixture
async def client(container) -> AsyncIterator[httpx.AsyncClient]:
    app = create_app(container)
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as c:
        yield c


def parse_sse(body: str) -> list:
    """Parse an SSE body into validated event models (comments/pings are skipped)."""
    events = []
    for block in body.split("\n\n"):
        data_lines = [line[len("data: ") :] for line in block.split("\n") if line.startswith("data: ")]
        if data_lines:
            events.append(agent_event_adapter.validate_python(json.loads("\n".join(data_lines))))
    return events


async def new_conversation(client: httpx.AsyncClient) -> str:
    response = await client.post("/api/conversations", json={})
    assert response.status_code == 201
    return response.json()["id"]
