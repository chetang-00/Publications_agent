import pytest
from qdrant_client import AsyncQdrantClient

from app.rag.vectorstore import (
    DOCUMENT_CHUNKS,
    PUBLICATIONS,
    ChunkPoint,
    EmbeddingDimensionMismatch,
    PublicationPoint,
    PublicationVectorFilter,
    VectorStore,
    VectorStoreNotReady,
)
from app.text import normalize
from tests.fakes import fake_vector


@pytest.fixture
async def store():
    s = VectorStore(AsyncQdrantClient(location=":memory:"))
    yield s
    await s.close()


def pub(
    pid: int, text: str, year: int, authors: list[str], label: str | None, source: str
) -> PublicationPoint:
    return PublicationPoint(
        id=pid,
        vector=fake_vector(text),
        payload={
            "publication_id": pid,
            "year": year,
            "cluster_label": label,
            "cluster_label_norm": normalize(label) if label else None,
            "authors_norm": [normalize(a) for a in authors],
            "source_title_norm": normalize(source),
            "document_type": "Article",
        },
    )


@pytest.fixture
async def seeded(store):
    await store.upsert_publications(
        [
            pub(1, "protein folding molecular dynamics", 2018, ["Smith J."], "Protein Structure", "Nature"),
            pub(2, "galaxy formation dark matter", 2021, ["Doe A.", "Smith J."], "Astrophysics", "ApJ"),
            pub(
                3,
                "protein structure prediction deep learning",
                2022,
                ["Lee K."],
                "Protein Structure",
                "Science",
            ),
        ]
    )
    return store


def test_normalize():
    assert normalize("  Ranganath   R. ") == "ranganath r"
    assert normalize("DNA Damage and Repair") == "dna damage and repair"


async def test_upsert_creates_collection_with_dimension(seeded):
    assert await seeded.count(PUBLICATIONS) == 3
    assert await seeded.collection_dim(PUBLICATIONS) == 64


async def test_dimension_mismatch_raises(store):
    await store.ensure_collection(PUBLICATIONS, 64)
    with pytest.raises(EmbeddingDimensionMismatch) as exc:
        await store.ensure_collection(PUBLICATIONS, 32)
    assert "reindex" in str(exc.value)


async def test_search_ranks_by_similarity(seeded):
    hits = await seeded.search_publications(
        fake_vector("protein folding"), PublicationVectorFilter(), limit=3
    )
    assert hits[0].id == 1
    assert hits[0].score >= hits[1].score


async def test_year_range_filter(seeded):
    hits = await seeded.search_publications(
        fake_vector("protein"), PublicationVectorFilter(year_from=2020, year_to=2022), limit=10
    )
    assert sorted(h.id for h in hits) == [2, 3]


async def test_author_filter_matches_any(seeded):
    hits = await seeded.search_publications(
        fake_vector("anything"), PublicationVectorFilter(authors_norm=["smith j"]), limit=10
    )
    assert sorted(h.id for h in hits) == [1, 2]


async def test_cluster_and_source_filters(seeded):
    hits = await seeded.search_publications(
        fake_vector("protein"),
        PublicationVectorFilter(cluster_labels_norm=["protein structure"], source_title_norm="science"),
        limit=10,
    )
    assert [h.id for h in hits] == [3]


async def test_search_with_wrong_dimension_raises(seeded):
    with pytest.raises(EmbeddingDimensionMismatch):
        await seeded.search_publications([0.1] * 32, PublicationVectorFilter(), limit=3)


async def test_search_before_indexing_raises_not_ready(store):
    with pytest.raises(VectorStoreNotReady) as exc:
        await store.search_publications(fake_vector("x"), PublicationVectorFilter(), limit=3)
    assert "seed" in str(exc.value)


async def test_set_publication_label_updates_payload(seeded):
    await seeded.set_publication_label(1, "Molecular Dynamics")
    hits = await seeded.search_publications(
        fake_vector("protein"), PublicationVectorFilter(cluster_labels_norm=["molecular dynamics"]), limit=10
    )
    assert [h.id for h in hits] == [1]
    assert hits[0].payload["cluster_label"] == "Molecular Dynamics"


async def test_set_publication_label_to_none(seeded):
    await seeded.set_publication_label(1, None)
    hits = await seeded.search_publications(
        fake_vector("protein"), PublicationVectorFilter(cluster_labels_norm=["protein structure"]), limit=10
    )
    assert [h.id for h in hits] == [3]


def chunk(cid: str, doc: str, idx: int, text: str) -> ChunkPoint:
    return ChunkPoint(
        id=cid,
        vector=fake_vector(text),
        payload={"document_id": doc, "chunk_index": idx, "page": 1, "filename": f"{doc}.pdf"},
    )


@pytest.fixture
async def chunks(store):
    await store.upsert_chunks(
        [
            chunk("00000000-0000-0000-0000-000000000001", "doc-a", 0, "methods sample size 120 patients"),
            chunk("00000000-0000-0000-0000-000000000002", "doc-a", 1, "results survival improved"),
            chunk("00000000-0000-0000-0000-000000000003", "doc-b", 0, "sample size calculation for trials"),
        ]
    )
    return store


async def test_chunk_search_scoped_to_documents(chunks):
    hits = await chunks.search_chunks(fake_vector("sample size"), ["doc-b"], limit=5)
    assert [h.payload["document_id"] for h in hits] == ["doc-b"]


async def test_chunk_search_all_documents(chunks):
    hits = await chunks.search_chunks(fake_vector("sample size"), None, limit=5)
    assert {h.payload["document_id"] for h in hits} == {"doc-a", "doc-b"}


async def test_delete_document_removes_only_its_chunks(chunks):
    await chunks.delete_document("doc-a")
    assert await chunks.count(DOCUMENT_CHUNKS) == 1


async def test_chunk_search_without_collection_returns_empty(store):
    assert await store.search_chunks(fake_vector("x"), None, limit=5) == []


async def test_delete_document_without_collection_is_noop(store):
    await store.delete_document("missing")


async def test_reset_collection(seeded):
    await seeded.reset_collection(PUBLICATIONS)
    assert await seeded.count(PUBLICATIONS) == 0
    assert await seeded.collection_dim(PUBLICATIONS) is None


async def test_ping(store):
    assert await store.ping() is True
