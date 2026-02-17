"""Document ingestion: store the upload, then parse → chunk → embed → index, recording status."""

import asyncio
import hashlib
import logging
import os
import re
import tempfile
from pathlib import Path
from uuid import uuid4

from sqlalchemy import delete, select, update

from app.db.models import Document, DocumentChunk, new_id
from app.db.session import Database
from app.llm.embeddings import Embedder
from app.llm.types import LLMError
from app.rag.chunking import chunk_pages
from app.rag.parsing import CONTENT_TYPES, KIND_BY_CONTENT_TYPE, ParseError, detect_kind, parse_document
from app.rag.vectorstore import DOCUMENT_CHUNKS, ChunkPoint, EmbeddingDimensionMismatch, VectorStore

log = logging.getLogger(__name__)

_UNSAFE_FILENAME_CHARS = re.compile(r"[\x00-\x1f\x7f/\\]")


class FileTooLarge(Exception):
    pass


def sanitize_filename(filename: str) -> str:
    name = filename.replace("\\", "/").rsplit("/", 1)[-1]
    name = _UNSAFE_FILENAME_CHARS.sub("", name).strip().lstrip(".") or "upload"
    if len(name) > 200:
        stem, dot, suffix = name.rpartition(".")
        name = f"{stem[: 200 - len(suffix) - 1]}.{suffix}" if dot else name[:200]
    return name


class IngestionService:
    def __init__(
        self, db: Database, store: VectorStore, embedder: Embedder, upload_dir: Path, max_bytes: int
    ) -> None:
        self.db = db
        self.store = store
        self.embedder = embedder
        self.upload_dir = Path(upload_dir)
        self.max_bytes = max_bytes

    def file_path(self, doc: Document) -> Path:
        kind = KIND_BY_CONTENT_TYPE[doc.content_type]
        return self.upload_dir / f"{doc.id}.{kind}"

    async def save_upload(self, filename: str, data: bytes) -> tuple[Document, bool]:
        """Store an upload. Returns (document, needs_processing)."""
        name = sanitize_filename(filename)
        kind = detect_kind(name, data[:16])
        if len(data) > self.max_bytes:
            raise FileTooLarge(f"The file is larger than the {self.max_bytes // (1024 * 1024)} MB limit.")
        digest = hashlib.sha256(data).hexdigest()

        async with self.db.sessionmaker() as session:
            existing = (
                await session.execute(select(Document).where(Document.sha256 == digest))
            ).scalar_one_or_none()
            if existing is not None:
                if existing.status != "failed":
                    return existing, False
                existing.status, existing.error = "processing", None
                await session.commit()
                return existing, True

            doc = Document(
                id=new_id(),
                filename=name,
                content_type=CONTENT_TYPES[kind],
                size_bytes=len(data),
                sha256=digest,
                status="processing",
            )
            await asyncio.to_thread(self._write_atomically, self.file_path(doc), data)
            session.add(doc)
            await session.commit()
            return doc, True

    @staticmethod
    def _write_atomically(path: Path, data: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".upload-")
        try:
            with os.fdopen(fd, "wb") as handle:
                handle.write(data)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise

    async def process(self, document_id: str) -> None:
        async with self.db.sessionmaker() as session:
            doc = await session.get(Document, document_id)
        if doc is None or doc.status != "processing":
            return
        try:
            parsed = await asyncio.to_thread(
                parse_document, self.file_path(doc), KIND_BY_CONTENT_TYPE[doc.content_type]
            )
            chunks = chunk_pages(parsed.pages)
            vectors = await self.embedder.embed([c.text for c in chunks])
            await self.store.delete_document(doc.id)
            await self.store.upsert_chunks(
                [
                    ChunkPoint(
                        id=str(uuid4()),
                        vector=vector,
                        payload={
                            "document_id": doc.id,
                            "chunk_index": chunk.index,
                            "page": chunk.page,
                            "filename": doc.filename,
                        },
                    )
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ]
            )
            async with self.db.sessionmaker() as session:
                await session.execute(delete(DocumentChunk).where(DocumentChunk.document_id == doc.id))
                session.add_all(
                    DocumentChunk(
                        document_id=doc.id,
                        chunk_index=c.index,
                        page=c.page,
                        text=c.text,
                        token_count=c.token_count,
                    )
                    for c in chunks
                )
                await session.execute(
                    update(Document)
                    .where(Document.id == doc.id)
                    .values(status="ready", error=None, page_count=parsed.page_count, chunk_count=len(chunks))
                )
                await session.commit()
            log.info("Document ready", extra={"document_id": doc.id, "chunks": len(chunks)})
        except ParseError as exc:
            await self._fail(doc.id, str(exc))
        except EmbeddingDimensionMismatch as exc:
            await self._fail(doc.id, str(exc))
        except LLMError as exc:
            await self._fail(doc.id, f"Embedding failed: {exc.message}")
        except Exception:
            log.exception("Document processing failed", extra={"document_id": doc.id})
            await self._fail(doc.id, "Unexpected error while processing the document.")

    async def _fail(self, document_id: str, reason: str) -> None:
        log.warning("Document failed", extra={"document_id": document_id, "reason": reason})
        async with self.db.sessionmaker() as session:
            await session.execute(
                update(Document).where(Document.id == document_id).values(status="failed", error=reason)
            )
            await session.commit()
        await self.store.delete_document(document_id)

    async def delete(self, document_id: str) -> bool:
        async with self.db.sessionmaker() as session:
            doc = await session.get(Document, document_id)
            if doc is None:
                return False
            path = self.file_path(doc)
            await session.delete(doc)
            await session.commit()
        await self.store.delete_document(document_id)
        path.unlink(missing_ok=True)
        return True

    async def mark_interrupted(self) -> int:
        async with self.db.sessionmaker() as session:
            result = await session.execute(
                update(Document)
                .where(Document.status == "processing")
                .values(
                    status="failed", error="Processing was interrupted by a restart. Upload the file again."
                )
            )
            await session.commit()
            return result.rowcount or 0

    async def reindex_all(self) -> int:
        """Rebuild document vectors from the stored chunk text (no re-parsing)."""
        await self.store.reset_collection(DOCUMENT_CHUNKS)
        total = 0
        async with self.db.sessionmaker() as session:
            docs = (await session.execute(select(Document).where(Document.status == "ready"))).scalars().all()
        for doc in docs:
            async with self.db.sessionmaker() as session:
                chunks = (
                    (
                        await session.execute(
                            select(DocumentChunk)
                            .where(DocumentChunk.document_id == doc.id)
                            .order_by(DocumentChunk.chunk_index)
                        )
                    )
                    .scalars()
                    .all()
                )
            if not chunks:
                continue
            vectors = await self.embedder.embed([c.text for c in chunks])
            await self.store.upsert_chunks(
                [
                    ChunkPoint(
                        id=str(uuid4()),
                        vector=vector,
                        payload={
                            "document_id": doc.id,
                            "chunk_index": c.chunk_index,
                            "page": c.page,
                            "filename": doc.filename,
                        },
                    )
                    for c, vector in zip(chunks, vectors, strict=True)
                ]
            )
            total += len(chunks)
        return total
