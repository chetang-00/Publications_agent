"""Tools over the user's uploaded documents."""

from pydantic import BaseModel, Field
from sqlalchemy import select

from app.db.models import ConversationDocument, Document, DocumentChunk
from app.llm.types import LLMError
from app.rag.vectorstore import EmbeddingDimensionMismatch
from app.tools.base import Tool, ToolArgs, ToolContext, ToolError

CHUNK_TEXT_CHARS = 1500


async def _attached_ids(ctx: ToolContext) -> list[str]:
    if not ctx.conversation_id:
        return []
    async with ctx.db.sessionmaker() as session:
        return list(
            (
                await session.execute(
                    select(ConversationDocument.document_id).where(
                        ConversationDocument.conversation_id == ctx.conversation_id
                    )
                )
            ).scalars()
        )


# ── list_documents ──


class ListDocumentsArgs(ToolArgs):
    conversation_only: bool = Field(False, description="Only documents attached to this conversation.")


class DocumentInfo(BaseModel):
    id: str
    filename: str
    status: str
    page_count: int | None
    chunk_count: int | None
    attached: bool
    error: str | None
    created_at: str


class ListDocumentsResult(BaseModel):
    documents: list[DocumentInfo]


async def list_documents(args: ListDocumentsArgs, ctx: ToolContext) -> ListDocumentsResult:
    attached = set(await _attached_ids(ctx))
    async with ctx.db.sessionmaker() as session:
        docs = (await session.execute(select(Document).order_by(Document.created_at.desc()))).scalars().all()
    return ListDocumentsResult(
        documents=[
            DocumentInfo(
                id=d.id,
                filename=d.filename,
                status=d.status,
                page_count=d.page_count,
                chunk_count=d.chunk_count,
                attached=d.id in attached,
                error=d.error,
                created_at=d.created_at.isoformat(),
            )
            for d in docs
            if not args.conversation_only or d.id in attached
        ]
    )


# ── search_documents ──


class SearchDocumentsArgs(ToolArgs):
    query: str = Field(min_length=2, max_length=500, description="What to look for, in natural language.")
    document_ids: list[str] = Field(
        default_factory=list,
        max_length=20,
        description="Documents to search. Omit to search the documents attached to this conversation, "
        "or all documents when none are attached.",
    )
    top_k: int = Field(6, ge=1, le=10, description="Number of passages to return.")


class ChunkHit(BaseModel):
    cite: str  # ready-made citation marker, e.g. "[doc:<document_id>:<chunk_index>]"
    document_id: str
    filename: str
    chunk_index: int
    page: int | None
    score: float
    text: str


class SearchDocumentsResult(BaseModel):
    query: str
    searched_documents: int
    chunks: list[ChunkHit]


async def search_documents(args: SearchDocumentsArgs, ctx: ToolContext) -> SearchDocumentsResult:
    scope = args.document_ids or await _attached_ids(ctx)
    async with ctx.db.sessionmaker() as session:
        stmt = select(Document.id).where(Document.status == "ready")
        if scope:
            stmt = stmt.where(Document.id.in_(scope))
        ready = list((await session.execute(stmt)).scalars())
    if not ready:
        if scope:
            raise ToolError(
                "None of the requested documents are ready to search (still processing or failed)."
            )
        raise ToolError("No processed documents are available. Upload documents on the Documents page first.")

    try:
        vector = (await ctx.embedder.embed([args.query]))[0]
        hits = await ctx.store.search_chunks(vector, ready, args.top_k)
    except EmbeddingDimensionMismatch as exc:
        raise ToolError(str(exc)) from exc
    except LLMError as exc:
        raise ToolError(f"Document search is temporarily unavailable ({exc.message}).") from exc

    keys = [(h.payload["document_id"], h.payload["chunk_index"]) for h in hits]
    async with ctx.db.sessionmaker() as session:
        rows = (
            await session.execute(
                select(DocumentChunk).where(DocumentChunk.document_id.in_({doc_id for doc_id, _ in keys}))
            )
        ).scalars()
        texts = {(c.document_id, c.chunk_index): c.text for c in rows}
    chunks = [
        ChunkHit(
            cite=f"[doc:{h.payload['document_id']}:{h.payload['chunk_index']}]",
            document_id=h.payload["document_id"],
            filename=h.payload.get("filename", ""),
            chunk_index=h.payload["chunk_index"],
            page=h.payload.get("page"),
            score=round(h.score, 4),
            text=texts[key][:CHUNK_TEXT_CHARS],
        )
        for h, key in zip(hits, keys, strict=True)
        if key in texts
    ]
    return SearchDocumentsResult(query=args.query, searched_documents=len(ready), chunks=chunks)


TOOLS: list[Tool] = [
    Tool(
        name="list_documents",
        description="List the user's uploaded documents with their processing status and whether each is attached to this conversation.",
        args_model=ListDocumentsArgs,
        result_model=ListDocumentsResult,
        handler=list_documents,
    ),
    Tool(
        name="search_documents",
        description=(
            "Semantic search over the user's uploaded documents (PDF, DOCX, TXT, MD). Returns the most relevant "
            "passages with document id, filename, page and chunk index for citation. Use for any question about "
            "the user's own files."
        ),
        args_model=SearchDocumentsArgs,
        result_model=SearchDocumentsResult,
        handler=search_documents,
    ),
]
