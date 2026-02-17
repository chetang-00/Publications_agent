"""The agent's tools. `build_registry()` is the single place tools are assembled."""

from app.tools.base import ToolRegistry


def build_registry() -> ToolRegistry:
    from app.tools import documents, publications, sql

    return ToolRegistry([*publications.TOOLS, *sql.TOOLS, *documents.TOOLS])
