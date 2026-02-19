from typing import Annotated

from fastapi import APIRouter, File, Response, UploadFile
from fastapi.responses import JSONResponse
from sqlalchemy import select

from app.api.deps import ContainerDep
from app.api.errors import ApiError, not_found
from app.api.schemas import DocumentOut
from app.db.models import Document
from app.rag.ingest import FileTooLarge
from app.rag.parsing import UnsupportedFile

router = APIRouter(prefix="/documents", tags=["documents"])

READ_CHUNK = 1024 * 1024


@router.post("", status_code=202, response_model=DocumentOut, responses={200: {"model": DocumentOut}})
async def upload_document(c: ContainerDep, file: Annotated[UploadFile, File()]) -> JSONResponse:
    limit = c.settings.max_upload_bytes
    data = bytearray()
    while chunk := await file.read(READ_CHUNK):
        data.extend(chunk)
        if len(data) > limit:
            raise ApiError(
                413, "file_too_large", f"The file is larger than the {c.settings.max_upload_mb} MB limit."
            )
    try:
        doc, needs_processing = await c.ingestion.save_upload(file.filename or "upload", bytes(data))
    except UnsupportedFile as exc:
        raise ApiError(400, "unsupported_file", str(exc)) from exc
    except FileTooLarge as exc:
        raise ApiError(413, "file_too_large", str(exc)) from exc
    if needs_processing:
        c.tasks.spawn(c.ingestion.process(doc.id), name=f"ingest:{doc.id}")
    body = DocumentOut.model_validate(doc).model_dump(mode="json")
    return JSONResponse(body, status_code=202 if needs_processing else 200)


@router.get("", response_model=list[DocumentOut])
async def list_documents(c: ContainerDep) -> list[DocumentOut]:
    async with c.db.sessionmaker() as s:
        docs = (await s.execute(select(Document).order_by(Document.created_at.desc()))).scalars()
        return [DocumentOut.model_validate(d) for d in docs]


@router.get("/{document_id}", response_model=DocumentOut)
async def get_document(document_id: str, c: ContainerDep) -> DocumentOut:
    async with c.db.sessionmaker() as s:
        doc = await s.get(Document, document_id)
    if doc is None:
        raise not_found("Document")
    return DocumentOut.model_validate(doc)


@router.delete("/{document_id}", status_code=204)
async def delete_document(document_id: str, c: ContainerDep) -> Response:
    async with c.db.sessionmaker() as s:
        doc = await s.get(Document, document_id)
    if doc is None:
        raise not_found("Document")
    if doc.status == "processing":
        raise ApiError(
            409, "document_processing", "The document is still being processed; delete it when it finishes."
        )
    await c.ingestion.delete(document_id)
    return Response(status_code=204)
