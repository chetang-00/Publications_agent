import asyncio

from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import func, select, text

from app.api.deps import ContainerDep
from app.api.schemas import CheckOut, HealthOut, ReadinessOut
from app.db.models import AppMeta, Publication
from app.db.session import migrations_at_head
from app.rag.vectorstore import PUBLICATIONS

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthOut)
async def health() -> HealthOut:
    """Liveness: the process is up."""
    return HealthOut()


@router.get("/ready", response_model=ReadinessOut, responses={503: {"model": ReadinessOut}})
async def ready(c: ContainerDep) -> JSONResponse:
    """Readiness: every dependency the agent needs, each reported by name."""
    checks: dict[str, CheckOut] = {}

    try:
        async with c.db.sessionmaker() as s:
            await s.execute(text("SELECT 1"))
            meta = dict((await s.execute(select(AppMeta.key, AppMeta.value))).all())
            total = (await s.execute(select(func.count(Publication.id)))).scalar_one()
            indexed = (
                await s.execute(
                    select(func.count(Publication.id)).where(Publication.vector_indexed_at.is_not(None))
                )
            ).scalar_one()
        checks["database"] = CheckOut(ok=True, detail="reachable")
    except Exception as exc:
        meta, total, indexed = {}, 0, 0
        checks["database"] = CheckOut(ok=False, detail=f"unreachable: {type(exc).__name__}")

    at_head = await asyncio.to_thread(migrations_at_head, c.settings.database_url)
    checks["migrations"] = CheckOut(
        ok=at_head, detail="up to date" if at_head else "pending; run `python -m app.cli migrate`"
    )

    store_ok = await c.store.ping()
    checks["vector_store"] = CheckOut(
        ok=store_ok, detail="reachable" if store_ok else f"cannot reach {c.settings.qdrant_url}"
    )

    built_with = meta.get("embedding_model")
    if built_with and built_with != c.settings.embedding_model:
        checks["embedding_index"] = CheckOut(
            ok=False,
            detail=f"vectors were built with '{built_with}' but EMBEDDING_MODEL is '{c.settings.embedding_model}'; run `make reindex`",
        )
    else:
        checks["embedding_index"] = CheckOut(ok=True, detail=f"model {c.settings.embedding_model}")

    vectors = await c.store.count(PUBLICATIONS) if store_ok else 0
    checks["publications"] = CheckOut(
        ok=total > 0 and indexed == total,
        detail=f"{total} publications, {indexed} indexed, {vectors} vectors"
        + ("" if total else "; run `make seed`"),
    )

    is_ready = all(check.ok for check in checks.values())
    body = ReadinessOut(status="ready" if is_ready else "not_ready", checks=checks)
    return JSONResponse(body.model_dump(), status_code=200 if is_ready else 503)
