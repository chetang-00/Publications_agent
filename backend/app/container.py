"""Wires the application's services together. Tests build a Container with fakes instead."""

from dataclasses import dataclass

from app.agent.loop import AgentRunner
from app.agent.streaming import RunManager
from app.config import Settings
from app.db.session import Database
from app.llm.embeddings import Embedder, PortkeyEmbedder
from app.llm.portkey import OpenAIChatLLM
from app.llm.types import ChatLLM
from app.rag.ingest import IngestionService
from app.rag.vectorstore import VectorStore
from app.tasks import TaskSet
from app.tools import build_registry
from app.tools.base import ToolRegistry


@dataclass
class Container:
    settings: Settings
    db: Database
    store: VectorStore
    embedder: Embedder
    llm: ChatLLM
    registry: ToolRegistry
    ingestion: IngestionService
    runner: AgentRunner
    runs: RunManager
    tasks: TaskSet

    async def aclose(self) -> None:
        await self.tasks.shutdown()
        close = getattr(self.embedder, "aclose", None)
        if close is not None:
            await close()
        await self.store.close()
        await self.db.dispose()


def build_container(settings: Settings) -> Container:
    db = Database(settings.database_url)
    store = VectorStore.from_url(settings.qdrant_url)
    embedder = PortkeyEmbedder.from_settings(settings)
    llm = OpenAIChatLLM(settings)
    registry = build_registry()
    tasks = TaskSet()
    return Container(
        settings=settings,
        db=db,
        store=store,
        embedder=embedder,
        llm=llm,
        registry=registry,
        ingestion=IngestionService(db, store, embedder, settings.upload_dir, settings.max_upload_bytes),
        runner=AgentRunner(
            llm=llm, registry=registry, db=db, store=store, embedder=embedder, settings=settings
        ),
        runs=RunManager(tasks),
        tasks=tasks,
    )
