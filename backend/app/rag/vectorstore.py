"""Qdrant access for the two vector collections: publications and uploaded-document chunks."""

import asyncio
from dataclasses import dataclass, field
from typing import Any

from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qm

from app.text import normalize

PUBLICATIONS = "publications"
DOCUMENT_CHUNKS = "document_chunks"

_PAYLOAD_INDEXES: dict[str, dict[str, qm.PayloadSchemaType]] = {
    PUBLICATIONS: {
        "year": qm.PayloadSchemaType.INTEGER,
        "cluster_label_norm": qm.PayloadSchemaType.KEYWORD,
        "authors_norm": qm.PayloadSchemaType.KEYWORD,
        "source_title_norm": qm.PayloadSchemaType.KEYWORD,
    },
    DOCUMENT_CHUNKS: {"document_id": qm.PayloadSchemaType.KEYWORD},
}


class VectorStoreNotReady(Exception):
    """A collection that should exist has not been built yet."""


class EmbeddingDimensionMismatch(Exception):
    pass


@dataclass
class PublicationPoint:
    id: int
    vector: list[float]
    payload: dict[str, Any]


@dataclass
class ChunkPoint:
    id: str
    vector: list[float]
    payload: dict[str, Any]


@dataclass
class PublicationVectorFilter:
    year_from: int | None = None
    year_to: int | None = None
    authors_norm: list[str] = field(default_factory=list)
    cluster_labels_norm: list[str] = field(default_factory=list)
    source_title_norm: str | None = None

    def to_qdrant(self) -> qm.Filter | None:
        must: list[qm.Condition] = []
        if self.year_from is not None or self.year_to is not None:
            must.append(qm.FieldCondition(key="year", range=qm.Range(gte=self.year_from, lte=self.year_to)))
        if self.authors_norm:
            must.append(qm.FieldCondition(key="authors_norm", match=qm.MatchAny(any=self.authors_norm)))
        if self.cluster_labels_norm:
            must.append(
                qm.FieldCondition(key="cluster_label_norm", match=qm.MatchAny(any=self.cluster_labels_norm))
            )
        if self.source_title_norm:
            must.append(
                qm.FieldCondition(key="source_title_norm", match=qm.MatchValue(value=self.source_title_norm))
            )
        return qm.Filter(must=must) if must else None


@dataclass
class VectorHit:
    id: int | str
    score: float
    payload: dict[str, Any]


class VectorStore:
    def __init__(self, client: AsyncQdrantClient) -> None:
        self.client = client
        # Documents are processed concurrently; collection creation must happen exactly once.
        self._create_lock = asyncio.Lock()

    @classmethod
    def from_url(cls, url: str) -> "VectorStore":
        if url == ":memory:":
            return cls(AsyncQdrantClient(location=":memory:"))
        return cls(AsyncQdrantClient(url=url, timeout=10))

    async def close(self) -> None:
        await self.client.close()

    async def ping(self) -> bool:
        try:
            await self.client.get_collections()
        except Exception:
            return False
        return True

    async def collection_dim(self, name: str) -> int | None:
        if not await self.client.collection_exists(name):
            return None
        info = await self.client.get_collection(name)
        vectors = info.config.params.vectors
        return vectors.size if isinstance(vectors, qm.VectorParams) else None

    async def ensure_collection(self, name: str, dim: int) -> None:
        existing = await self.collection_dim(name)
        if existing is None:
            async with self._create_lock:
                existing = await self.collection_dim(name)  # another task may have created it meanwhile
                if existing is None:
                    try:
                        await self.client.create_collection(
                            name, vectors_config=qm.VectorParams(size=dim, distance=qm.Distance.COSINE)
                        )
                    except Exception:
                        # Created concurrently by another process (HTTP 409); anything else re-raises below.
                        if await self.collection_dim(name) is None:
                            raise
                    else:
                        for field_name, schema in _PAYLOAD_INDEXES[name].items():
                            await self.client.create_payload_index(name, field_name, field_schema=schema)
                    existing = await self.collection_dim(name)
        if existing != dim:
            raise EmbeddingDimensionMismatch(_mismatch_message(name, existing or 0, dim))

    async def reset_collection(self, name: str) -> None:
        if await self.client.collection_exists(name):
            await self.client.delete_collection(name)

    async def count(self, name: str) -> int:
        if not await self.client.collection_exists(name):
            return 0
        return (await self.client.count(name, exact=True)).count

    # ── publications ──

    async def upsert_publications(self, points: list[PublicationPoint]) -> None:
        if not points:
            return
        await self.ensure_collection(PUBLICATIONS, len(points[0].vector))
        await self.client.upsert(
            PUBLICATIONS,
            points=[qm.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points],
            wait=True,
        )

    async def search_publications(
        self, vector: list[float], filt: PublicationVectorFilter, limit: int
    ) -> list[VectorHit]:
        dim = await self.collection_dim(PUBLICATIONS)
        if dim is None:
            raise VectorStoreNotReady(
                "The publications vector index has not been built yet. Run `make seed` to load and index them."
            )
        if dim != len(vector):
            raise EmbeddingDimensionMismatch(_mismatch_message(PUBLICATIONS, dim, len(vector)))
        response = await self.client.query_points(
            PUBLICATIONS, query=vector, query_filter=filt.to_qdrant(), limit=limit, with_payload=True
        )
        return [VectorHit(id=p.id, score=p.score, payload=p.payload or {}) for p in response.points]

    async def set_publication_label(self, publication_id: int, label: str | None) -> None:
        await self.client.set_payload(
            PUBLICATIONS,
            payload={"cluster_label": label, "cluster_label_norm": normalize(label) if label else None},
            points=[publication_id],
            wait=True,
        )

    # ── document chunks ──

    async def upsert_chunks(self, points: list[ChunkPoint]) -> None:
        if not points:
            return
        await self.ensure_collection(DOCUMENT_CHUNKS, len(points[0].vector))
        await self.client.upsert(
            DOCUMENT_CHUNKS,
            points=[qm.PointStruct(id=p.id, vector=p.vector, payload=p.payload) for p in points],
            wait=True,
        )

    async def search_chunks(
        self, vector: list[float], document_ids: list[str] | None, limit: int
    ) -> list[VectorHit]:
        dim = await self.collection_dim(DOCUMENT_CHUNKS)
        if dim is None:
            return []
        if dim != len(vector):
            raise EmbeddingDimensionMismatch(_mismatch_message(DOCUMENT_CHUNKS, dim, len(vector)))
        query_filter = None
        if document_ids:
            query_filter = qm.Filter(
                must=[qm.FieldCondition(key="document_id", match=qm.MatchAny(any=document_ids))]
            )
        response = await self.client.query_points(
            DOCUMENT_CHUNKS, query=vector, query_filter=query_filter, limit=limit, with_payload=True
        )
        return [VectorHit(id=p.id, score=p.score, payload=p.payload or {}) for p in response.points]

    async def delete_document(self, document_id: str) -> None:
        if not await self.client.collection_exists(DOCUMENT_CHUNKS):
            return
        await self.client.delete(
            DOCUMENT_CHUNKS,
            points_selector=qm.FilterSelector(
                filter=qm.Filter(
                    must=[qm.FieldCondition(key="document_id", match=qm.MatchValue(value=document_id))]
                )
            ),
            wait=True,
        )


def _mismatch_message(name: str, existing: int, got: int) -> str:
    return (
        f"Vector collection '{name}' holds {existing}-dimension vectors but the embedding model returns "
        f"{got}. The embedding model changed; rebuild the index with `make reindex`."
    )
