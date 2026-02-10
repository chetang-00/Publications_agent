from collections.abc import AsyncIterator
from pathlib import Path

import pytest
from qdrant_client import AsyncQdrantClient

from app.config import Settings
from app.db.session import Database, run_migrations
from app.rag.vectorstore import VectorStore
from app.seed.publications import index_publications, load_publications, read_csv
from tests.fakes import FakeEmbedder

SETTINGS_ENV_VARS = (
    "PORTKEY_BASE_URL",
    "PORTKEY_API_KEY",
    "PORTKEY_VIRTUAL_KEY",
    "EMBEDDING_PORTKEY_API_KEY",
    "EMBEDDING_PORTKEY_VIRTUAL_KEY",
    "CHAT_MODEL",
    "EMBEDDING_MODEL",
    "EMBEDDING_BATCH_SIZE",
    "QDRANT_URL",
    "DATABASE_URL",
    "UPLOAD_DIR",
    "MAX_UPLOAD_MB",
    "AGENT_MAX_STEPS",
    "TOOL_TIMEOUT_SECONDS",
    "LOG_LEVEL",
    "CORS_ORIGINS",
)


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tests never read the developer's shell environment or .env file."""
    for name in SETTINGS_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


@pytest.fixture
def settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        portkey_api_key="test-portkey-key",
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'app.db'}",
        qdrant_url=":memory:",
        upload_dir=tmp_path / "uploads",
        embedding_batch_size=8,
    )


@pytest.fixture
async def db(settings: Settings) -> AsyncIterator[Database]:
    run_migrations(settings.database_url)
    database = Database(settings.database_url)
    yield database
    await database.dispose()


FIXTURES = Path(__file__).parent / "fixtures"
SAMPLE_CSV = FIXTURES / "publications_sample.csv"


@pytest.fixture
def embedder() -> FakeEmbedder:
    return FakeEmbedder()


@pytest.fixture
async def store() -> AsyncIterator[VectorStore]:
    vector_store = VectorStore(AsyncQdrantClient(location=":memory:"))
    yield vector_store
    await vector_store.close()


@pytest.fixture
async def seeded(db: Database, store: VectorStore, embedder: FakeEmbedder) -> Database:
    """Database and vector store loaded with the 20 synthetic sample publications."""
    result = read_csv(SAMPLE_CSV)
    await load_publications(db, result.rows)
    await index_publications(db, store, embedder, batch_size=8)
    return db
