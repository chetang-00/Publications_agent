from app.db.models import Document
from app.rag.vectorstore import DOCUMENT_CHUNKS
from tests.api.conftest import new_conversation
from tests.doc_helpers import make_pdf


def pdf_upload(tmp_path, name="trial.pdf", pages=None):
    path = make_pdf(
        tmp_path / name, pages or ["Methods. We enrolled 120 patients.", "Results. Survival improved."]
    )
    return {"file": (name, path.read_bytes(), "application/pdf")}


async def test_upload_processes_in_background(client, container, tmp_path):
    response = await client.post("/api/documents", files=pdf_upload(tmp_path))
    assert response.status_code == 202
    doc = response.json()
    assert (doc["status"], doc["filename"], doc["content_type"]) == (
        "processing",
        "trial.pdf",
        "application/pdf",
    )
    await container.tasks.wait_all()
    ready = (await client.get(f"/api/documents/{doc['id']}")).json()
    assert (ready["status"], ready["page_count"], ready["chunk_count"]) == ("ready", 2, 2)
    assert [d["id"] for d in (await client.get("/api/documents")).json()] == [doc["id"]]


async def test_duplicate_upload_returns_existing_document(client, container, tmp_path):
    data = pdf_upload(tmp_path)["file"][1]  # identical bytes uploaded under two names
    first = (
        await client.post("/api/documents", files={"file": ("trial.pdf", data, "application/pdf")})
    ).json()
    await container.tasks.wait_all()
    second = await client.post("/api/documents", files={"file": ("copy.pdf", data, "application/pdf")})
    assert second.status_code == 200
    assert second.json()["id"] == first["id"]


async def test_scanned_pdf_ends_failed_with_reason(client, container, tmp_path):
    doc = (await client.post("/api/documents", files=pdf_upload(tmp_path, "scan.pdf", ["", ""]))).json()
    await container.tasks.wait_all()
    failed = (await client.get(f"/api/documents/{doc['id']}")).json()
    assert failed["status"] == "failed"
    assert "No extractable text" in failed["error"]


async def test_upload_rejections(client, settings):
    bad_ext = await client.post(
        "/api/documents", files={"file": ("slides.pptx", b"PK\x03\x04", "application/octet-stream")}
    )
    assert (bad_ext.status_code, bad_ext.json()["error"]["code"]) == (400, "unsupported_file")
    wrong_sig = await client.post(
        "/api/documents", files={"file": ("photo.pdf", b"\x89PNG\r\n\x1a\n", "application/pdf")}
    )
    assert wrong_sig.status_code == 400
    empty = await client.post("/api/documents", files={"file": ("empty.txt", b"", "text/plain")})
    assert empty.status_code == 400
    settings.max_upload_mb = 1
    big = await client.post(
        "/api/documents", files={"file": ("big.txt", b"a" * (1024 * 1024 + 10), "text/plain")}
    )
    assert (big.status_code, big.json()["error"]["code"]) == (413, "file_too_large")
    missing = await client.post("/api/documents", data={"x": "1"})
    assert missing.status_code == 422


async def test_delete_document(client, container, store, tmp_path):
    doc = (await client.post("/api/documents", files=pdf_upload(tmp_path))).json()
    await container.tasks.wait_all()
    assert (await client.delete(f"/api/documents/{doc['id']}")).status_code == 204
    assert (await client.get(f"/api/documents/{doc['id']}")).status_code == 404
    assert await store.count(DOCUMENT_CHUNKS) == 0
    assert (await client.delete(f"/api/documents/{doc['id']}")).status_code == 404


async def test_cannot_delete_a_document_while_processing(client, container):
    async with container.db.sessionmaker() as s:
        s.add(
            Document(
                id="busy", filename="a.pdf", content_type="application/pdf", size_bytes=1, sha256="c" * 64
            )
        )
        await s.commit()
    response = await client.delete("/api/documents/busy")
    assert (response.status_code, response.json()["error"]["code"]) == (409, "document_processing")


async def test_attach_documents_to_conversation(client, container, tmp_path):
    doc = (await client.post("/api/documents", files=pdf_upload(tmp_path))).json()
    await container.tasks.wait_all()
    conv_id = await new_conversation(client)
    response = await client.post(
        f"/api/conversations/{conv_id}/documents", json={"document_ids": [doc["id"], "unknown-id"]}
    )
    assert response.status_code == 200
    assert response.json() == {"document_ids": [doc["id"]]}
    detail = (await client.get(f"/api/conversations/{conv_id}")).json()
    assert [d["id"] for d in detail["documents"]] == [doc["id"]]
    cleared = await client.post(f"/api/conversations/{conv_id}/documents", json={"document_ids": []})
    assert cleared.json() == {"document_ids": []}
