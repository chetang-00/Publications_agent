"""Citation checking: an answer may only cite ids that a tool actually returned.

The model writes markers like [pub:6] or [doc:<uuid>:2]. At the end of a run every marker is
checked against what tools returned in this run (plus citations already verified in earlier
answers of the conversation). Verified markers stay in the text and become structured citations;
anything else is removed, so a hallucinated reference never reaches the user.
"""

import re
from typing import Any

from app.agent.events import Citation

_MARKER = re.compile(r"\[\s*(?:pub:\s*(?P<pub>\d+)|doc:\s*(?P<doc>[0-9a-fA-F-]{36}):\s*(?P<chunk>\d+))\s*\]")


class CitationTracker:
    def __init__(self) -> None:
        self.publications: dict[int, str | None] = {}
        self.chunks: dict[tuple[str, int], tuple[str | None, int | None]] = {}

    def _pub(self, pub_id: Any, title: Any = None) -> None:
        if isinstance(pub_id, int) or (isinstance(pub_id, str) and pub_id.isdigit()):
            self.publications.setdefault(int(pub_id), title if isinstance(title, str) else None)

    def record_tool_result(self, tool_name: str, result: dict[str, Any] | None) -> None:
        if not isinstance(result, dict):
            return
        items = result.get("items")
        if isinstance(items, list):  # search_publications, filter_publications
            for item in items:
                if isinstance(item, dict):
                    self._pub(item.get("id"), item.get("title"))
        if tool_name == "get_publication":
            self._pub(result.get("id"), result.get("title"))
        if tool_name == "update_cluster_label":
            self._pub(result.get("publication_id"), result.get("title"))
        chunks = result.get("chunks")
        if isinstance(chunks, list):  # search_documents
            for chunk in chunks:
                if (
                    isinstance(chunk, dict)
                    and chunk.get("document_id")
                    and isinstance(chunk.get("chunk_index"), int)
                ):
                    key = (str(chunk["document_id"]).lower(), chunk["chunk_index"])
                    self.chunks[key] = (chunk.get("filename"), chunk.get("page"))
        if tool_name == "run_readonly_sql":
            columns = result.get("columns") or []
            id_column = next((c for c in ("id", "publication_id") if c in columns), None)
            if id_column:
                id_index = columns.index(id_column)
                title_index = columns.index("title") if "title" in columns else None
                for row in result.get("rows") or []:
                    self._pub(row[id_index], row[title_index] if title_index is not None else None)

    def record_citations(self, citations: list[dict[str, Any]] | None) -> None:
        """Trust citations that were verified in earlier answers of the same conversation."""
        for c in citations or []:
            if c.get("kind") == "publication":
                self._pub(c.get("id"), c.get("title"))
            elif c.get("kind") == "document" and isinstance(c.get("chunk_index"), int):
                self.chunks[(str(c.get("id")).lower(), c["chunk_index"])] = (c.get("filename"), c.get("page"))

    def finalize(self, content: str) -> tuple[str, list[Citation], list[str]]:
        """Returns (content with only verified markers, citations in first-use order, unverified markers)."""
        citations: dict[str, Citation] = {}
        unverified: list[str] = []

        def replace(match: re.Match[str]) -> str:
            if match.group("pub"):
                pub_id = int(match.group("pub"))
                marker = f"pub:{pub_id}"
                if pub_id not in self.publications:
                    unverified.append(marker)
                    return ""
                citations.setdefault(
                    marker,
                    Citation(
                        kind="publication", id=str(pub_id), marker=marker, title=self.publications[pub_id]
                    ),
                )
                return f"[{marker}]"
            doc_id, chunk = match.group("doc").lower(), int(match.group("chunk"))
            marker = f"doc:{doc_id}:{chunk}"
            if (doc_id, chunk) not in self.chunks:
                unverified.append(marker)
                return ""
            filename, page = self.chunks[(doc_id, chunk)]
            citations.setdefault(
                marker,
                Citation(
                    kind="document", id=doc_id, marker=marker, filename=filename, page=page, chunk_index=chunk
                ),
            )
            return f"[{marker}]"

        cleaned = _MARKER.sub(replace, content)
        if unverified:
            cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)  # "paper ." -> "paper."
            cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip(), list(citations.values()), unverified
