import pytest
from sqlalchemy import func, select

from app.db.models import Document, DocumentChunk
from app.llm.embeddings import EmbeddingError
from app.rag.ingest import FileTooLarge, IngestionService
from app.rag.parsing import UnsupportedFile
from app.rag.vectorstore import DOCUMENT_CHUNKS
from tests.doc_helpers import make_pdf
from tests.fakes import fake_vector


@pytest.fixture
def service(db, store, embedder, settings) -> IngestionService:
    return IngestionService(db, store, embedder, settings.upload_dir, max_bytes=settings.max_upload_bytes)


@pytest.fixture
def pdf_bytes(tmp_path) -> bytes:
    path = make_pdf(
        tmp_path / "trial.pdf",
        [
            "Methods. We enrolled 120 patients in a randomized trial.",
            "Results. Survival improved by 20 percent.",
        ],
    )
    return path.read_bytes()


async def get_doc(db, doc_id) -> Document:
    async with db.sessionmaker() as s:
        return await s.get(Document, doc_id)


async def test_save_upload_stores_file_and_row(service, pdf_bytes, settings):
    doc, created = await service.save_upload("trial.pdf", pdf_bytes)
    assert created is True
    assert doc.status == "processing"
    assert doc.content_type == "application/pdf"
    assert doc.size_bytes == len(pdf_bytes)
    assert service.file_path(doc).read_bytes() == pdf_bytes
    assert service.file_path(doc).parent == settings.upload_dir


async def test_duplicate_upload_returns_existing_document(service, pdf_bytes):
    first, _ = await service.save_upload("trial.pdf", pdf_bytes)
    second, created = await service.save_upload("copy.pdf", pdf_bytes)
    assert created is False
    assert second.id == first.id


async def test_duplicate_of_failed_document_is_retried(service, pdf_bytes, embedder):
    doc, _ = await service.save_upload("trial.pdf", pdf_bytes)
    embedder.fail_with = EmbeddingError("down")
    await service.process(doc.id)
    embedder.fail_with = None
    again, created = await service.save_upload("trial.pdf", pdf_bytes)
    assert (again.id, created, again.status, again.error) == (doc.id, True, "processing", None)


async def test_filename_is_sanitised(service):
    doc, _ = await service.save_upload("../../etc/notes\x00.txt", b"Some harmless notes about the study.")
    assert doc.filename == "notes.txt"


async def test_rejects_wrong_signature_and_oversize(service, settings):
    with pytest.raises(UnsupportedFile):
        await service.save_upload("fake.pdf", b"\x89PNG\r\n\x1a\n")
    small = type(service)(service.db, service.store, service.embedder, settings.upload_dir, max_bytes=10)
    with pytest.raises(FileTooLarge):
        await small.save_upload("big.txt", b"x" * 11)


async def test_process_makes_document_searchable(service, pdf_bytes, db, store):
    doc, _ = await service.save_upload("trial.pdf", pdf_bytes)
    await service.process(doc.id)
    doc = await get_doc(db, doc.id)
    assert (doc.status, doc.page_count, doc.chunk_count, doc.error) == ("ready", 2, 2, None)
    async with db.sessionmaker() as s:
        chunks = (await s.execute(select(DocumentChunk).order_by(DocumentChunk.chunk_index))).scalars().all()
    assert [(c.chunk_index, c.page) for c in chunks] == [(0, 1), (1, 2)]
    hits = await store.search_chunks(fake_vector("survival improved"), [doc.id], limit=1)
    assert hits[0].payload == {"document_id": doc.id, "chunk_index": 1, "page": 2, "filename": "trial.pdf"}


async def test_scanned_pdf_fails_with_reason(service, tmp_path, db):
    data = make_pdf(tmp_path / "scan.pdf", ["", ""]).read_bytes()
    doc, _ = await service.save_upload("scan.pdf", data)
    await service.process(doc.id)
    doc = await get_doc(db, doc.id)
    assert doc.status == "failed"
    assert "No extractable text" in doc.error


async def test_embedding_failure_marks_document_failed(service, pdf_bytes, embedder, db):
    doc, _ = await service.save_upload("trial.pdf", pdf_bytes)
    embedder.fail_with = EmbeddingError("gateway down")
    await service.process(doc.id)
    doc = await get_doc(db, doc.id)
    assert doc.status == "failed"
    assert "Embedding failed" in doc.error


async def test_reprocessing_replaces_chunks(service, pdf_bytes, db, store):
    doc, _ = await service.save_upload("trial.pdf", pdf_bytes)
    await service.process(doc.id)
    async with db.sessionmaker() as s:
        (await s.get(Document, doc.id)).status = "processing"
        await s.commit()
    await service.process(doc.id)
    async with db.sessionmaker() as s:
        assert (await s.execute(select(func.count()).select_from(DocumentChunk))).scalar_one() == 2
    assert await store.count(DOCUMENT_CHUNKS) == 2


async def test_delete_removes_rows_vectors_and_file(service, pdf_bytes, db, store):
    doc, _ = await service.save_upload("trial.pdf", pdf_bytes)
    await service.process(doc.id)
    path = service.file_path(doc)
    assert await service.delete(doc.id) is True
    assert await get_doc(db, doc.id) is None
    assert not path.exists()
    assert await store.count(DOCUMENT_CHUNKS) == 0
    async with db.sessionmaker() as s:
        assert (await s.execute(select(func.count()).select_from(DocumentChunk))).scalar_one() == 0
    assert await service.delete(doc.id) is False


async def test_mark_interrupted(service, pdf_bytes, db):
    doc, _ = await service.save_upload("trial.pdf", pdf_bytes)
    assert await service.mark_interrupted() == 1
    doc = await get_doc(db, doc.id)
    assert doc.status == "failed"
    assert "restart" in doc.error


async def test_reindex_rebuilds_vectors_from_stored_chunks(service, pdf_bytes, store):
    doc, _ = await service.save_upload("trial.pdf", pdf_bytes)
    await service.process(doc.id)
    await store.reset_collection(DOCUMENT_CHUNKS)
    assert await service.reindex_all() == 2
    assert await store.count(DOCUMENT_CHUNKS) == 2
