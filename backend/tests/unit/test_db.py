from alembic.autogenerate import compare_metadata
from alembic.migration import MigrationContext
from sqlalchemy import create_engine, func, select, text

from app.db.models import Base, Conversation, Message
from app.db.session import Database, migrations_at_head, run_migrations, sync_url


def test_migrations_match_models(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'm.db'}"
    run_migrations(url)
    engine = create_engine(sync_url(url))
    with engine.connect() as conn:
        diffs = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    engine.dispose()
    assert diffs == []


def test_migrations_at_head(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'h.db'}"
    assert migrations_at_head(url) is False
    run_migrations(url)
    assert migrations_at_head(url) is True


def test_run_migrations_creates_parent_directory(tmp_path):
    url = f"sqlite+aiosqlite:///{tmp_path / 'nested' / 'dir' / 'x.db'}"
    run_migrations(url)
    assert (tmp_path / "nested" / "dir" / "x.db").exists()


def test_sqlite_path(tmp_path):
    database = Database(f"sqlite+aiosqlite:///{tmp_path / 'p.db'}")
    assert database.sqlite_path == tmp_path / "p.db"


async def test_connections_use_wal_and_foreign_keys(db: Database):
    async with db.engine.connect() as conn:
        assert (await conn.execute(text("PRAGMA journal_mode"))).scalar() == "wal"
        assert (await conn.execute(text("PRAGMA foreign_keys"))).scalar() == 1


async def test_deleting_conversation_cascades_to_messages(db: Database):
    async with db.sessionmaker() as session:
        conv = Conversation(title="t")
        session.add(conv)
        await session.flush()
        session.add(Message(conversation_id=conv.id, role="user", content="hi"))
        await session.commit()
        await session.execute(Conversation.__table__.delete().where(Conversation.id == conv.id))
        await session.commit()
        remaining = (await session.execute(select(func.count()).select_from(Message))).scalar()
    assert remaining == 0
