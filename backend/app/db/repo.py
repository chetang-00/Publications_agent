"""Persistence for conversations, messages, agent runs and tool calls."""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import delete, func, select, update

from app.db.models import (
    AgentRun,
    Conversation,
    ConversationDocument,
    Document,
    Message,
    ToolCallRecord,
)
from app.db.session import Database

DEFAULT_TITLE = "New conversation"
HISTORY_LIMIT = 20
ACTIVE_STATUSES = ("running", "awaiting_approval")


# ── conversations ──


async def create_conversation(db: Database, title: str | None = None) -> Conversation:
    async with db.sessionmaker() as s:
        conv = Conversation(title=(title or DEFAULT_TITLE)[:200])
        s.add(conv)
        await s.commit()
        return conv


async def get_conversation(db: Database, conversation_id: str) -> Conversation | None:
    async with db.sessionmaker() as s:
        return await s.get(Conversation, conversation_id)


async def list_conversations(db: Database, limit: int = 200) -> list[Conversation]:
    async with db.sessionmaker() as s:
        rows = await s.execute(select(Conversation).order_by(Conversation.updated_at.desc()).limit(limit))
        return list(rows.scalars())


async def delete_conversation(db: Database, conversation_id: str) -> bool:
    async with db.sessionmaker() as s:
        result = await s.execute(delete(Conversation).where(Conversation.id == conversation_id))
        await s.commit()
        return bool(result.rowcount)


async def title_from_first_message(db: Database, conversation_id: str, content: str) -> None:
    title = " ".join(content.split())[:80] or DEFAULT_TITLE
    async with db.sessionmaker() as s:
        await s.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id, Conversation.title == DEFAULT_TITLE)
            .values(title=title)
        )
        await s.execute(
            update(Conversation)
            .where(Conversation.id == conversation_id)
            .values(updated_at=datetime.now(UTC))
        )
        await s.commit()


async def attached_documents(db: Database, conversation_id: str) -> list[Document]:
    async with db.sessionmaker() as s:
        rows = await s.execute(
            select(Document)
            .join(ConversationDocument, ConversationDocument.document_id == Document.id)
            .where(ConversationDocument.conversation_id == conversation_id)
            .order_by(ConversationDocument.created_at)
        )
        return list(rows.scalars())


async def set_attached_documents(
    db: Database, conversation_id: str, document_ids: Sequence[str]
) -> list[str]:
    """Replace the conversation's attachments; unknown ids are ignored. Returns the attached ids."""
    async with db.sessionmaker() as s:
        known = list(
            (await s.execute(select(Document.id).where(Document.id.in_(list(document_ids))))).scalars()
        )
        await s.execute(
            delete(ConversationDocument).where(ConversationDocument.conversation_id == conversation_id)
        )
        s.add_all(
            ConversationDocument(conversation_id=conversation_id, document_id=doc_id) for doc_id in known
        )
        await s.commit()
        return known


# ── messages ──


async def add_message(db: Database, **fields: Any) -> Message:
    async with db.sessionmaker() as s:
        message = Message(**fields)
        s.add(message)
        await s.commit()
        return message


async def history_messages(db: Database, conversation_id: str, current_run_id: str) -> list[Message]:
    """Earlier user questions and final answers (no tool chatter), most recent HISTORY_LIMIT."""
    async with db.sessionmaker() as s:
        rows = await s.execute(
            select(Message)
            .where(
                Message.conversation_id == conversation_id,
                (Message.run_id != current_run_id) | Message.run_id.is_(None),
                (Message.role == "user") | ((Message.role == "assistant") & Message.tool_calls.is_(None)),
            )
            .order_by(Message.id.desc())
            .limit(HISTORY_LIMIT)
        )
        return list(reversed(list(rows.scalars())))


async def run_messages(db: Database, run_id: str) -> list[Message]:
    async with db.sessionmaker() as s:
        rows = await s.execute(select(Message).where(Message.run_id == run_id).order_by(Message.id))
        return list(rows.scalars())


async def earlier_citations(db: Database, conversation_id: str) -> list[dict[str, Any]]:
    async with db.sessionmaker() as s:
        rows = await s.execute(
            select(Message.citations).where(
                Message.conversation_id == conversation_id,
                Message.role == "assistant",
                Message.citations.is_not(None),
            )
        )
        return [c for citations in rows.scalars() for c in (citations or [])]


async def conversation_messages(db: Database, conversation_id: str) -> list[Message]:
    async with db.sessionmaker() as s:
        rows = await s.execute(
            select(Message).where(Message.conversation_id == conversation_id).order_by(Message.id)
        )
        return list(rows.scalars())


# ── runs ──


async def create_run(db: Database, conversation_id: str, model: str) -> AgentRun:
    async with db.sessionmaker() as s:
        run = AgentRun(conversation_id=conversation_id, model=model, status="running")
        s.add(run)
        await s.commit()
        return run


async def get_run(db: Database, run_id: str) -> AgentRun | None:
    async with db.sessionmaker() as s:
        return await s.get(AgentRun, run_id)


async def update_run(db: Database, run_id: str, **values: Any) -> None:
    if values.get("status") in ("completed", "failed", "cancelled"):
        values.setdefault("finished_at", datetime.now(UTC))
    async with db.sessionmaker() as s:
        await s.execute(update(AgentRun).where(AgentRun.id == run_id).values(**values))
        await s.commit()


async def active_run(db: Database, conversation_id: str) -> AgentRun | None:
    async with db.sessionmaker() as s:
        rows = await s.execute(
            select(AgentRun)
            .where(AgentRun.conversation_id == conversation_id, AgentRun.status.in_(ACTIVE_STATUSES))
            .order_by(AgentRun.started_at.desc())
            .limit(1)
        )
        return rows.scalar_one_or_none()


async def conversation_runs(db: Database, conversation_id: str) -> list[AgentRun]:
    async with db.sessionmaker() as s:
        rows = await s.execute(
            select(AgentRun).where(AgentRun.conversation_id == conversation_id).order_by(AgentRun.started_at)
        )
        return list(rows.scalars())


async def fail_interrupted_runs(db: Database) -> int:
    """Runs left 'running' by a crash or restart can never finish; mark them failed."""
    async with db.sessionmaker() as s:
        result = await s.execute(
            update(AgentRun)
            .where(AgentRun.status == "running")
            .values(
                status="failed",
                error_code="interrupted",
                error="The server restarted while this answer was being generated.",
                finished_at=datetime.now(UTC),
            )
        )
        await s.commit()
        return result.rowcount or 0


# ── tool calls ──


async def record_tool_call(db: Database, **fields: Any) -> None:
    async with db.sessionmaker() as s:
        s.add(ToolCallRecord(**fields))
        await s.commit()


async def update_tool_call(db: Database, run_id: str, step: int, call_id: str, **values: Any) -> None:
    async with db.sessionmaker() as s:
        await s.execute(
            update(ToolCallRecord)
            .where(
                ToolCallRecord.run_id == run_id,
                ToolCallRecord.step == step,
                ToolCallRecord.call_id == call_id,
            )
            .values(**values)
        )
        await s.commit()


async def pending_tool_call(db: Database, run_id: str) -> ToolCallRecord | None:
    async with db.sessionmaker() as s:
        rows = await s.execute(
            select(ToolCallRecord).where(
                ToolCallRecord.run_id == run_id, ToolCallRecord.status == "awaiting_approval"
            )
        )
        return rows.scalars().first()


async def run_tool_calls(db: Database, run_ids: Sequence[str]) -> list[ToolCallRecord]:
    if not run_ids:
        return []
    async with db.sessionmaker() as s:
        rows = await s.execute(
            select(ToolCallRecord).where(ToolCallRecord.run_id.in_(list(run_ids))).order_by(ToolCallRecord.id)
        )
        return list(rows.scalars())


async def count_messages(db: Database, conversation_id: str) -> int:
    async with db.sessionmaker() as s:
        return (
            await s.execute(
                select(func.count()).select_from(Message).where(Message.conversation_id == conversation_id)
            )
        ).scalar_one()
