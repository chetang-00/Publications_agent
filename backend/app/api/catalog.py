"""Read-only endpoints for publications (citation drawer) and the tool catalogue."""

from fastapi import APIRouter

from app.api.deps import ContainerDep
from app.api.errors import not_found
from app.api.schemas import ToolInfoOut
from app.tools.base import ToolContext, ToolError
from app.tools.publications import GetPublicationArgs, PublicationDetail, get_publication

router = APIRouter(tags=["catalog"])


@router.get("/publications/{publication_id}", response_model=PublicationDetail)
async def read_publication(publication_id: int, c: ContainerDep) -> PublicationDetail:
    ctx = ToolContext(db=c.db, store=c.store, embedder=c.embedder, settings=c.settings)
    try:
        return await get_publication(GetPublicationArgs(publication_id=max(publication_id, 1)), ctx)
    except ToolError as exc:
        raise not_found("Publication") from exc


@router.get("/tools", response_model=list[ToolInfoOut])
async def list_tools(c: ContainerDep) -> list[ToolInfoOut]:
    return [
        ToolInfoOut(
            name=tool.name,
            description=tool.description,
            requires_approval=tool.requires_approval,
            parameters=tool.openai_schema()["function"]["parameters"],
        )
        for tool in c.registry.tools()
    ]
