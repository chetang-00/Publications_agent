"""Load the Scopus publications CSV into SQLite, then embed publications into Qdrant.

Both steps are idempotent: rows are upserted on `eid`, and only publications without
`vector_indexed_at` are embedded, so an interrupted run resumes where it stopped.
"""

import csv
import logging
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel
from sqlalchemy import delete, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import AppMeta, ClusterLabelChange, Publication, PublicationAuthor, PublicationKeyword
from app.db.session import Database
from app.llm.embeddings import Embedder
from app.rag.vectorstore import PUBLICATIONS, PublicationPoint, VectorStore
from app.text import normalize

log = logging.getLogger(__name__)

REQUIRED_COLUMNS = frozenset({"eid", "title", "year", "authors", "author_full_names"})
UNKNOWN_LABELS = frozenset({"", "unknown", "unknown label"})
EMBEDDING_TEXT_LIMIT = 6000
_SCOPUS_ID_SUFFIX = re.compile(r"\s*\(\d+\)\s*$")


class EmbeddingModelChanged(Exception):
    pass


class PublicationRow(BaseModel):
    eid: str
    title: str
    year: int | None
    authors: list[str]
    author_full_names: list[str | None]
    source_title: str | None = None
    publisher: str | None = None
    document_type: str | None = None
    doi: str | None = None
    link: str | None = None
    cited_by: int | None = None
    open_access: str | None = None
    affiliations: str | None = None
    authors_with_affiliations: str | None = None
    abstract: str | None = None
    author_keywords: list[str] = []
    index_keywords: list[str] = []
    cluster_label: str | None = None


@dataclass
class ReadResult:
    rows: list[PublicationRow]
    skipped: list[tuple[int, str]] = field(default_factory=list)  # (CSV line, reason)
    duplicates: int = 0


@dataclass
class LoadStats:
    inserted: int = 0
    updated: int = 0
    unchanged: int = 0


# ── CSV parsing ──────────────────────────────────────────────────────────────


def _clean(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    return value or None


def _split(value: str | None) -> list[str]:
    seen: set[str] = set()
    items: list[str] = []
    for part in (value or "").split(";"):
        part = part.strip()
        if part and part.lower() not in seen:
            seen.add(part.lower())
            items.append(part)
    return items


def _parse_int(value: str | None, field_name: str) -> int | None:
    value = _clean(value)
    if value is None:
        return None
    try:
        return int(float(value))
    except ValueError:
        raise ValueError(f"{field_name} is not a number: {value!r}") from None


def _parse_row(raw: dict[str, str]) -> PublicationRow:
    eid = _clean(raw.get("eid"))
    if not eid:
        raise ValueError("missing eid")
    title = _clean(raw.get("title"))
    if not title:
        raise ValueError("empty title")
    year = _parse_int(raw.get("year"), "year")
    if year is not None and not 1800 <= year <= 2100:
        raise ValueError(f"year out of range: {year}")

    authors = [a for a in (part.strip() for part in (raw.get("authors") or "").split(";")) if a]
    full_names = [
        _SCOPUS_ID_SUFFIX.sub("", part).strip() or None
        for part in (raw.get("author_full_names") or "").split(";")
        if part.strip()
    ]
    if len(full_names) != len(authors):
        full_names = [None] * len(authors)

    document_type = _clean(raw.get("document_type"))
    if document_type:
        document_type = document_type[0].upper() + document_type[1:].lower()

    label = _clean(raw.get("cluster_label"))
    if label and label.lower() in UNKNOWN_LABELS:
        label = None

    return PublicationRow(
        eid=eid,
        title=title,
        year=year,
        authors=authors,
        author_full_names=full_names,
        source_title=_clean(raw.get("source_title")),
        publisher=_clean(raw.get("publisher")),
        document_type=document_type,
        doi=_clean(raw.get("doi")),
        link=_clean(raw.get("link")),
        cited_by=_parse_int(raw.get("cited_by"), "cited_by"),
        open_access=_clean(raw.get("open_access")),
        affiliations=_clean(raw.get("affiliations")),
        authors_with_affiliations=_clean(raw.get("authors_with_affiliations")),
        abstract=_clean(raw.get("abstract")),
        author_keywords=_split(raw.get("author_keywords")),
        index_keywords=_split(raw.get("index_keywords")),
        cluster_label=label,
    )


def read_csv(path: Path) -> ReadResult:
    csv.field_size_limit(2**31 - 1)  # abstracts and reference lists exceed the 128 KB default
    result = ReadResult(rows=[])
    seen: set[str] = set()
    with Path(path).open(newline="", encoding="utf-8-sig", errors="replace") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        for raw in reader:
            try:
                row = _parse_row(raw)
            except ValueError as exc:
                result.skipped.append((reader.line_num, str(exc)))
                continue
            if row.eid in seen:
                result.duplicates += 1
                continue
            seen.add(row.eid)
            result.rows.append(row)
    return result


# ── SQLite load ──────────────────────────────────────────────────────────────


def _columns(row: PublicationRow) -> dict[str, Any]:
    return {
        "title": row.title,
        "year": row.year,
        "authors": "; ".join(row.authors) or None,
        "author_full_names": "; ".join(n for n in row.author_full_names if n) or None,
        "source_title": row.source_title,
        "publisher": row.publisher,
        "document_type": row.document_type,
        "doi": row.doi,
        "link": row.link,
        "cited_by": row.cited_by,
        "open_access": row.open_access,
        "affiliations": row.affiliations,
        "authors_with_affiliations": row.authors_with_affiliations,
        "abstract": row.abstract,
        "author_keywords": "; ".join(row.author_keywords) or None,
        "index_keywords": "; ".join(row.index_keywords) or None,
        "cluster_label": row.cluster_label,
    }


def _add_children(session: AsyncSession, publication_id: int, row: PublicationRow) -> None:
    for position, (author, full_name) in enumerate(zip(row.authors, row.author_full_names, strict=True)):
        session.add(
            PublicationAuthor(
                publication_id=publication_id,
                position=position,
                author=author,
                author_norm=normalize(author),
                full_name=full_name,
            )
        )
    for kind, keywords in (("author", row.author_keywords), ("index", row.index_keywords)):
        for keyword in keywords:
            session.add(
                PublicationKeyword(
                    publication_id=publication_id, keyword=keyword, keyword_norm=normalize(keyword), kind=kind
                )
            )


async def load_publications(db: Database, rows: Sequence[PublicationRow]) -> LoadStats:
    stats = LoadStats()
    async with db.sessionmaker() as session:
        existing = {p.eid: p for p in (await session.execute(select(Publication))).scalars()}
        # Labels changed through the approval flow are human decisions; a re-seed must not undo them.
        locked_labels = set((await session.execute(select(ClusterLabelChange.publication_id))).scalars())

        for row in rows:
            values = _columns(row)
            pub = existing.get(row.eid)
            if pub is None:
                pub = Publication(eid=row.eid, **values)
                session.add(pub)
                await session.flush()
                _add_children(session, pub.id, row)
                stats.inserted += 1
                continue

            if pub.id in locked_labels:
                values.pop("cluster_label")
            changed = {key: value for key, value in values.items() if getattr(pub, key) != value}
            if not changed:
                stats.unchanged += 1
                continue
            for key, value in changed.items():
                setattr(pub, key, value)
            pub.vector_indexed_at = None
            await session.execute(delete(PublicationAuthor).where(PublicationAuthor.publication_id == pub.id))
            await session.execute(
                delete(PublicationKeyword).where(PublicationKeyword.publication_id == pub.id)
            )
            _add_children(session, pub.id, row)
            stats.updated += 1

        await session.commit()
    return stats


# ── Vector indexing ──────────────────────────────────────────────────────────


def publication_embedding_text(pub: Publication) -> str:
    parts = [f"Title: {pub.title}"]
    if pub.authors:
        parts.append("Authors: " + "; ".join(pub.authors.split("; ")[:10]))
    if pub.year:
        parts.append(f"Year: {pub.year}")
    if pub.source_title:
        parts.append(f"Journal: {pub.source_title}")
    keywords = "; ".join(k for k in (pub.author_keywords, pub.index_keywords) if k)
    if keywords:
        parts.append(f"Keywords: {keywords}")
    if pub.abstract:
        parts.append(f"Abstract: {pub.abstract}")
    return "\n".join(parts)[:EMBEDDING_TEXT_LIMIT]


def publication_payload(pub: Publication, authors: Sequence[PublicationAuthor]) -> dict[str, Any]:
    return {
        "publication_id": pub.id,
        "year": pub.year,
        "cluster_label": pub.cluster_label,
        "cluster_label_norm": normalize(pub.cluster_label) if pub.cluster_label else None,
        "authors_norm": [a.author_norm for a in authors],
        "source_title_norm": normalize(pub.source_title) if pub.source_title else None,
        "document_type": pub.document_type,
    }


async def _read_meta(session: AsyncSession) -> dict[str, str]:
    return dict((await session.execute(select(AppMeta.key, AppMeta.value))).all())


async def index_publications(
    db: Database,
    store: VectorStore,
    embedder: Embedder,
    batch_size: int,
    *,
    reindex: bool = False,
    progress: Callable[[int], None] | None = None,
) -> int:
    async with db.sessionmaker() as session:
        meta = await _read_meta(session)
        if reindex:
            await store.reset_collection(PUBLICATIONS)
            await session.execute(update(Publication).values(vector_indexed_at=None))
            await session.execute(
                delete(AppMeta).where(AppMeta.key.in_(["embedding_model", "embedding_dim"]))
            )
            await session.commit()
        elif (
            meta.get("embedding_model")
            and meta["embedding_model"] != embedder.model
            and await store.count(PUBLICATIONS) > 0
        ):
            raise EmbeddingModelChanged(
                f"Publications were indexed with '{meta['embedding_model']}' but EMBEDDING_MODEL is "
                f"'{embedder.model}'. Run `make reindex` to rebuild the vector index."
            )

    total = 0
    while True:
        async with db.sessionmaker() as session:
            pubs = list(
                (
                    await session.execute(
                        select(Publication)
                        .where(Publication.vector_indexed_at.is_(None))
                        .order_by(Publication.id)
                        .limit(batch_size)
                    )
                ).scalars()
            )
            if not pubs:
                break
            ids = [p.id for p in pubs]
            authors: dict[int, list[PublicationAuthor]] = {pid: [] for pid in ids}
            for author in (
                await session.execute(
                    select(PublicationAuthor)
                    .where(PublicationAuthor.publication_id.in_(ids))
                    .order_by(PublicationAuthor.publication_id, PublicationAuthor.position)
                )
            ).scalars():
                authors[author.publication_id].append(author)

            vectors = await embedder.embed([publication_embedding_text(p) for p in pubs])
            await store.upsert_publications(
                [
                    PublicationPoint(id=p.id, vector=v, payload=publication_payload(p, authors[p.id]))
                    for p, v in zip(pubs, vectors, strict=True)
                ]
            )
            await session.execute(
                update(Publication).where(Publication.id.in_(ids)).values(vector_indexed_at=datetime.now(UTC))
            )
            await session.merge(AppMeta(key="embedding_model", value=embedder.model))
            await session.merge(AppMeta(key="embedding_dim", value=str(len(vectors[0]))))
            await session.commit()
        total += len(pubs)
        if progress:
            progress(total)
    return total
