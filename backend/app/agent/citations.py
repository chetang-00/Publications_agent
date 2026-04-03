"""Citation checking: an answer may only cite ids that a tool actually returned.

The model writes markers like [pub:6] or [doc:<uuid>:2]. At the end of a run every marker is
checked against what tools returned in this run (plus citations already verified in earlier
answers of the conversation). Verified markers stay in the text and become structured citations;
anything else is removed, so a hallucinated reference never reaches the user.
"""

import re
from typing import Any

from app.agent.events import Citation

# A bracket that contains at least one marker: "[pub:6]", "[pub:6, pub:9]", "[pub: 6; 9]", "[doc:<uuid>:2]".
_GROUP = re.compile(r"\[(?P<body>[^\[\]]*?\b(?:pub|doc)\s*:[^\[\]]*)\]", re.IGNORECASE)
_PUB = re.compile(r"^(?:pub\s*:\s*)?(?P<id>\d+)$", re.IGNORECASE)
_DOC = re.compile(r"^doc\s*:\s*(?P<doc>[0-9a-fA-F-]{36})\s*:\s*(?P<chunk>\d+)$", re.IGNORECASE)


MIN_TITLE_FOR_AUTOCITE = 20  # shorter titles are too likely to match ordinary prose


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

    def _cite_quoted_titles(self, content: str, citations: dict[str, Citation]) -> str:
        """Safety net: a retrieved paper named by its exact title but left uncited gets its marker."""
        lowered = content.lower()
        for pub_id, title in self.publications.items():
            marker = f"pub:{pub_id}"
            if not title or len(title) < MIN_TITLE_FOR_AUTOCITE or f"[{marker}]" in content:
                continue
            start = lowered.find(title.lower())
            if start < 0:
                continue
            end = start + len(title)
            content = f"{content[:end]} [{marker}]{content[end:]}"
            lowered = content.lower()
            citations.setdefault(
                marker, Citation(kind="publication", id=str(pub_id), marker=marker, title=title)
            )
        return content

    def finalize(self, content: str) -> tuple[str, list[Citation], list[str]]:
        """Returns (content with only verified markers, citations in first-use order, unverified markers)."""
        citations: dict[str, Citation] = {}
        unverified: list[str] = []

        def verify_pub(pub_id: int) -> str:
            marker = f"pub:{pub_id}"
            if pub_id not in self.publications:
                unverified.append(marker)
                return ""
            citations.setdefault(
                marker,
                Citation(kind="publication", id=str(pub_id), marker=marker, title=self.publications[pub_id]),
            )
            return f"[{marker}]"

        def verify_doc(doc_id: str, chunk: int) -> str:
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

        def replace(match: re.Match[str]) -> str:
            # Each id in a bracket is checked on its own; verified ones become single markers.
            out: list[str] = []
            for part in re.split(r"[,;]", match.group("body")):
                part = part.strip()
                if doc := _DOC.match(part):
                    out.append(verify_doc(doc.group("doc").lower(), int(doc.group("chunk"))))
                elif pub := _PUB.match(part):
                    out.append(verify_pub(int(pub.group("id"))))
            return "".join(out)

        cleaned = _GROUP.sub(replace, content)
        cleaned = self._cite_quoted_titles(cleaned, citations)
        if unverified:
            cleaned = re.sub(r"[ \t]+([.,;:!?])", r"\1", cleaned)  # "paper ." -> "paper."
            cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        return cleaned.strip(), list(citations.values()), unverified
