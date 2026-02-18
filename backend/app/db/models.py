"""SQLAlchemy models. One SQLite database holds publications, documents, conversations and agent runs."""

from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from sqlalchemy import JSON, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


def utcnow() -> datetime:
    return datetime.now(UTC)


def new_id() -> str:
    return str(uuid4())


class Base(DeclarativeBase):
    type_annotation_map = {  # noqa: RUF012 (SQLAlchemy reads this class attribute)
        datetime: DateTime(timezone=True),
        dict[str, Any]: JSON,
        list[Any]: JSON,
    }


# ── Publications ─────────────────────────────────────────────────────────────


class Publication(Base):
    __tablename__ = "publications"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    eid: Mapped[str] = mapped_column(String(64), unique=True)
    title: Mapped[str] = mapped_column(Text)
    year: Mapped[int | None] = mapped_column(index=True)
    authors: Mapped[str | None] = mapped_column(Text)
    author_full_names: Mapped[str | None] = mapped_column(Text)
    source_title: Mapped[str | None] = mapped_column(Text, index=True)
    publisher: Mapped[str | None] = mapped_column(Text)
    document_type: Mapped[str | None] = mapped_column(String(64))
    doi: Mapped[str | None] = mapped_column(String(255))
    link: Mapped[str | None] = mapped_column(Text)
    cited_by: Mapped[int | None]
    open_access: Mapped[str | None] = mapped_column(Text)
    affiliations: Mapped[str | None] = mapped_column(Text)
    authors_with_affiliations: Mapped[str | None] = mapped_column(Text)
    abstract: Mapped[str | None] = mapped_column(Text)
    author_keywords: Mapped[str | None] = mapped_column(Text)
    index_keywords: Mapped[str | None] = mapped_column(Text)
    cluster_label: Mapped[str | None] = mapped_column(String(200), index=True)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)
    vector_indexed_at: Mapped[datetime | None]


class PublicationAuthor(Base):
    __tablename__ = "publication_authors"

    publication_id: Mapped[int] = mapped_column(
        ForeignKey("publications.id", ondelete="CASCADE"), primary_key=True
    )
    position: Mapped[int] = mapped_column(primary_key=True)
    author: Mapped[str] = mapped_column(String(255))
    author_norm: Mapped[str] = mapped_column(String(255), index=True)
    full_name: Mapped[str | None] = mapped_column(String(255))


class PublicationKeyword(Base):
    __tablename__ = "publication_keywords"
    __table_args__ = (Index("ix_publication_keywords_publication_id", "publication_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    publication_id: Mapped[int] = mapped_column(ForeignKey("publications.id", ondelete="CASCADE"))
    keyword: Mapped[str] = mapped_column(String(500))
    keyword_norm: Mapped[str] = mapped_column(String(500), index=True)
    kind: Mapped[str] = mapped_column(String(10))  # "author" | "index"


class ClusterLabelChange(Base):
    __tablename__ = "cluster_label_changes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    publication_id: Mapped[int] = mapped_column(ForeignKey("publications.id", ondelete="CASCADE"), index=True)
    old_label: Mapped[str | None] = mapped_column(String(200))
    new_label: Mapped[str] = mapped_column(String(200))
    reason: Mapped[str] = mapped_column(Text)
    run_id: Mapped[str | None] = mapped_column(String(36))
    changed_at: Mapped[datetime] = mapped_column(default=utcnow)


# ── Documents ────────────────────────────────────────────────────────────────


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    filename: Mapped[str] = mapped_column(String(255))
    content_type: Mapped[str] = mapped_column(String(100))
    size_bytes: Mapped[int]
    sha256: Mapped[str] = mapped_column(String(64), unique=True)
    page_count: Mapped[int | None]
    chunk_count: Mapped[int | None]
    status: Mapped[str] = mapped_column(String(16), default="processing")  # processing | ready | failed
    error: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"
    __table_args__ = (UniqueConstraint("document_id", "chunk_index"),)

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int]
    page: Mapped[int | None]
    text: Mapped[str] = mapped_column(Text)
    token_count: Mapped[int]


# ── Conversations and agent runs ─────────────────────────────────────────────


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    title: Mapped[str] = mapped_column(String(200), default="New conversation")
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(default=utcnow, onupdate=utcnow)


class ConversationDocument(Base):
    __tablename__ = "conversation_documents"

    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), primary_key=True
    )
    document_id: Mapped[str] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class AgentRun(Base):
    __tablename__ = "agent_runs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=new_id)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    # running | awaiting_approval | completed | failed
    status: Mapped[str] = mapped_column(String(24), default="running")
    steps: Mapped[int] = mapped_column(default=0)
    invalid_tool_calls: Mapped[int] = mapped_column(default=0)
    model: Mapped[str] = mapped_column(String(200))
    prompt_tokens: Mapped[int] = mapped_column(default=0)
    completion_tokens: Mapped[int] = mapped_column(default=0)
    error_code: Mapped[str | None] = mapped_column(String(64))
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(default=utcnow)
    finished_at: Mapped[datetime | None]


class Message(Base):
    __tablename__ = "messages"

    # Integer key: messages are ordered by insertion, which timestamps cannot guarantee.
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    conversation_id: Mapped[str] = mapped_column(
        ForeignKey("conversations.id", ondelete="CASCADE"), index=True
    )
    run_id: Mapped[str | None] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16))  # user | assistant | tool
    content: Mapped[str] = mapped_column(Text, default="")
    tool_calls: Mapped[list[Any] | None]
    tool_call_id: Mapped[str | None] = mapped_column(String(128))
    citations: Mapped[list[Any] | None]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class ToolCallRecord(Base):
    __tablename__ = "tool_calls"
    __table_args__ = (UniqueConstraint("run_id", "step", "call_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[str] = mapped_column(ForeignKey("agent_runs.id", ondelete="CASCADE"), index=True)
    call_id: Mapped[str] = mapped_column(String(128))
    step: Mapped[int]
    name: Mapped[str] = mapped_column(String(100))
    raw_arguments: Mapped[str] = mapped_column(Text)
    arguments: Mapped[dict[str, Any] | None]
    # ok | invalid_arguments | error | timeout | awaiting_approval | rejected
    status: Mapped[str] = mapped_column(String(24))
    result: Mapped[dict[str, Any] | None]
    error: Mapped[str | None] = mapped_column(Text)
    duration_ms: Mapped[int | None]
    created_at: Mapped[datetime] = mapped_column(default=utcnow)


class AppMeta(Base):
    """Small key/value store, e.g. which embedding model and dimension the vectors were built with."""

    __tablename__ = "app_meta"

    key: Mapped[str] = mapped_column(String(100), primary_key=True)
    value: Mapped[str] = mapped_column(Text)
