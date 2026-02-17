import json

import pytest

from app.db.models import Conversation, ConversationDocument
from app.rag.ingest import IngestionService
from app.tools import build_registry
from app.tools.base import ToolContext
from tests.doc_helpers import make_pdf


@pytest.fixture
async def docs(db, store, embedder, settings, tmp_path):
    """Three documents: two ready (one attached to a conversation), one failed."""
    service = IngestionService(db, store, embedder, settings.upload_dir, max_bytes=settings.max_upload_bytes)
    trial = make_pdf(
        tmp_path / "trial.pdf", ["Methods. We enrolled 120 patients.", "Results. Survival improved."]
    )
    guide = make_pdf(tmp_path / "guide.pdf", ["Lab safety guide. Wear goggles near lasers."])
    scan = make_pdf(tmp_path / "scan.pdf", [""])
    ids = {}
    for name, path in (("trial", trial), ("guide", guide), ("scan", scan)):
        doc, _ = await service.save_upload(f"{name}.pdf", path.read_bytes())
        await service.process(doc.id)
        ids[name] = doc.id
    async with db.sessionmaker() as s:
        conv = Conversation(title="t")
        s.add(conv)
        await s.flush()
        s.add(ConversationDocument(conversation_id=conv.id, document_id=ids["trial"]))
        await s.commit()
        ids["conversation"] = conv.id
    return ids


async def call(ctx: ToolContext, name: str, **arguments):
    registry = build_registry()
    tool = registry.get(name)
    return await registry.execute(tool, registry.parse_args(tool, json.dumps(arguments)), ctx)


@pytest.fixture
def ctx(db, store, embedder, settings, docs) -> ToolContext:
    return ToolContext(
        db=db, store=store, embedder=embedder, settings=settings, conversation_id=docs["conversation"]
    )


async def test_list_documents_with_attachment_flags(ctx, docs):
    out = await call(ctx, "list_documents")
    by_name = {d["filename"]: d for d in out.result["documents"]}
    assert by_name["trial.pdf"]["attached"] is True
    assert by_name["guide.pdf"]["attached"] is False
    assert by_name["scan.pdf"]["status"] == "failed"
    assert "No extractable text" in by_name["scan.pdf"]["error"]


async def test_list_only_conversation_documents(ctx):
    out = await call(ctx, "list_documents", conversation_only=True)
    assert [d["filename"] for d in out.result["documents"]] == ["trial.pdf"]


async def test_search_defaults_to_attached_documents(ctx):
    out = await call(ctx, "search_documents", query="lasers goggles safety")
    assert {c["filename"] for c in out.result["chunks"]} == {"trial.pdf"}
    assert out.result["searched_documents"] == 1


async def test_search_explicit_documents_with_page_numbers(ctx, docs):
    out = await call(
        ctx, "search_documents", query="survival improved", document_ids=[docs["trial"]], top_k=1
    )
    chunk = out.result["chunks"][0]
    assert (chunk["document_id"], chunk["chunk_index"], chunk["page"]) == (docs["trial"], 1, 2)
    assert "Survival improved" in chunk["text"]


async def test_search_all_ready_documents_without_conversation(db, store, embedder, settings, docs):
    ctx = ToolContext(db=db, store=store, embedder=embedder, settings=settings)
    out = await call(ctx, "search_documents", query="lasers goggles")
    assert out.result["chunks"][0]["filename"] == "guide.pdf"
    assert out.result["searched_documents"] == 2


async def test_search_only_failed_documents_is_an_error(ctx, docs):
    out = await call(ctx, "search_documents", query="anything", document_ids=[docs["scan"]])
    assert out.status == "error"
    assert "ready" in out.error


async def test_search_with_no_documents_explains_upload(db, store, embedder, settings):
    ctx = ToolContext(db=db, store=store, embedder=embedder, settings=settings)
    out = await call(ctx, "search_documents", query="anything")
    assert out.status == "error"
    assert "Upload" in out.error
