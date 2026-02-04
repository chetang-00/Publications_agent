"""Database engine/session setup and Alembic helpers."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine, event
from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

BACKEND_DIR = Path(__file__).resolve().parents[2]
ALEMBIC_INI = BACKEND_DIR / "alembic.ini"


def sync_url(url: str) -> str:
    """Alembic and the read-only SQL tool use the plain sqlite3 driver."""
    return url.replace("+aiosqlite", "")


def sqlite_file(url: str) -> Path:
    database = make_url(url).database
    if not database or database == ":memory:":
        raise ValueError(f"DATABASE_URL must point to a SQLite file, got {url!r}")
    return Path(database)


def _set_sqlite_pragmas(dbapi_connection, _record) -> None:
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA foreign_keys=ON")
    cursor.execute("PRAGMA busy_timeout=5000")
    cursor.execute("PRAGMA synchronous=NORMAL")
    cursor.close()


class Database:
    def __init__(self, url: str) -> None:
        self.url = url
        self.sqlite_path = sqlite_file(url)
        self.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine = create_async_engine(url, connect_args={"timeout": 30})
        event.listen(self.engine.sync_engine, "connect", _set_sqlite_pragmas)
        self.sessionmaker: async_sessionmaker[AsyncSession] = async_sessionmaker(
            self.engine, expire_on_commit=False
        )

    async def dispose(self) -> None:
        await self.engine.dispose()


def _alembic_config(url: str) -> Config:
    config = Config(str(ALEMBIC_INI))
    config.set_main_option("script_location", str(BACKEND_DIR / "alembic"))
    config.set_main_option("sqlalchemy.url", sync_url(url))
    return config


def run_migrations(url: str) -> None:
    sqlite_file(url).parent.mkdir(parents=True, exist_ok=True)
    command.upgrade(_alembic_config(url), "head")


def migrations_at_head(url: str) -> bool:
    path = sqlite_file(url)
    if not path.exists():
        return False
    head = ScriptDirectory.from_config(_alembic_config(url)).get_current_head()
    engine = create_engine(sync_url(url), poolclass=NullPool)
    try:
        with engine.connect() as conn:
            current = MigrationContext.configure(conn).get_current_revision()
    finally:
        engine.dispose()
    return current == head
