import csv
from datetime import UTC, datetime

import pytest
from sqlalchemy import func, select, update

from app.cli import main as cli_main
from app.db.models import (
    AppMeta,
    ClusterLabelChange,
    Publication,
    PublicationAuthor,
    PublicationKeyword,
)
from app.rag.vectorstore import PUBLICATIONS
from app.seed.publications import (
    EmbeddingModelChanged,
    index_publications,
    load_publications,
    publication_embedding_text,
    read_csv,
)
from tests.conftest import SAMPLE_CSV
from tests.fakes import FakeEmbedder


async def count(db, model) -> int:
    async with db.sessionmaker() as s:
        return (await s.execute(select(func.count()).select_from(model))).scalar_one()


async def get_pub(db, eid: str) -> Publication:
    async with db.sessionmaker() as s:
        return (await s.execute(select(Publication).where(Publication.eid == eid))).scalar_one()


# ── read_csv ──


def test_read_csv_counts_rows_skips_and_duplicates():
    result = read_csv(SAMPLE_CSV)
    assert len(result.rows) == 20
    assert result.duplicates == 1
    assert len(result.skipped) == 3


def test_seed_skips_bad_rows():
    reasons = {line: reason for line, reason in read_csv(SAMPLE_CSV).skipped}
    assert any("eid" in r for r in reasons.values())
    assert any("year" in r for r in reasons.values())
    assert any("title" in r for r in reasons.values())
    assert all(isinstance(line, int) and line >= 2 for line in reasons)


def test_duplicate_eid_keeps_first_row():
    first = next(r for r in read_csv(SAMPLE_CSV).rows if r.eid == "2-s2.0-SYN00001")
    assert first.title == "Score matching for nonlinear diffusion models"


def test_authors_and_full_names_are_aligned_without_ids():
    row = next(r for r in read_csv(SAMPLE_CSV).rows if r.eid == "2-s2.0-SYN00001")
    assert row.authors == ["Ranganath R.", "Smith J."]
    assert row.author_full_names == ["Ranganath, Rajesh", "Smith, John"]


def test_field_normalisation():
    rows = {r.eid: r for r in read_csv(SAMPLE_CSV).rows}
    assert rows["2-s2.0-SYN00011"].cluster_label is None  # "Unknown Label"
    assert rows["2-s2.0-SYN00001"].cited_by == 12  # "12.0"
    assert rows["2-s2.0-SYN00019"].cited_by is None  # ""
    assert rows["2-s2.0-SYN00001"].document_type == "Conference paper"  # "Conference Paper"
    assert rows["2-s2.0-SYN00001"].index_keywords == [
        "Generative model",
        "Diffusion process",
        "Gaussian priors",
    ]
    assert rows["2-s2.0-SYN00019"].author_keywords == []


def test_missing_required_column_is_reported(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("title,year\nA,2020\n")
    with pytest.raises(ValueError) as exc:
        read_csv(path)
    assert "eid" in str(exc.value)


# ── load_publications ──


async def test_load_inserts_publications_authors_and_keywords(db):
    stats = await load_publications(db, read_csv(SAMPLE_CSV).rows)
    assert (stats.inserted, stats.updated, stats.unchanged) == (20, 0, 0)
    assert await count(db, Publication) == 20
    async with db.sessionmaker() as s:
        authors = (
            (
                await s.execute(
                    select(PublicationAuthor)
                    .join(Publication, Publication.id == PublicationAuthor.publication_id)
                    .where(Publication.eid == "2-s2.0-SYN00020")
                    .order_by(PublicationAuthor.position)
                )
            )
            .scalars()
            .all()
        )
        kinds = dict(
            (
                await s.execute(
                    select(PublicationKeyword.kind, func.count()).group_by(PublicationKeyword.kind)
                )
            ).all()
        )
    assert [(a.author, a.author_norm, a.full_name) for a in authors] == [
        ("Smith J.", "smith j", "Smith, John"),
        ("Garcia M.", "garcia m", "Garcia, Maria"),
        ("Lee K.", "lee k", "Lee, Kim"),
    ]
    assert kinds["author"] > 0 and kinds["index"] > 0


async def test_reload_is_idempotent(db):
    rows = read_csv(SAMPLE_CSV).rows
    await load_publications(db, rows)
    stats = await load_publications(db, rows)
    assert (stats.inserted, stats.updated, stats.unchanged) == (0, 0, 20)
    assert await count(db, Publication) == 20
    assert await count(db, PublicationAuthor) == sum(len(r.authors) for r in rows)


async def test_reload_with_changed_abstract_updates_and_requeues_vectors(db):
    rows = read_csv(SAMPLE_CSV).rows
    await load_publications(db, rows)
    async with db.sessionmaker() as s:
        await s.execute(update(Publication).values(vector_indexed_at=datetime.now(UTC)))
        await s.commit()
    rows[0].abstract = "A completely rewritten abstract."
    stats = await load_publications(db, rows)
    assert stats.updated == 1
    pub = await get_pub(db, rows[0].eid)
    assert pub.abstract == "A completely rewritten abstract."
    assert pub.vector_indexed_at is None


async def test_reload_keeps_label_changed_through_approval(db):
    rows = read_csv(SAMPLE_CSV).rows
    await load_publications(db, rows)
    pub = await get_pub(db, "2-s2.0-SYN00011")
    async with db.sessionmaker() as s:
        await s.execute(
            update(Publication).where(Publication.id == pub.id).values(cluster_label="Astrophysics")
        )
        s.add(ClusterLabelChange(publication_id=pub.id, old_label=None, new_label="Astrophysics", reason="r"))
        await s.commit()
    await load_publications(db, rows)
    assert (await get_pub(db, "2-s2.0-SYN00011")).cluster_label == "Astrophysics"


# ── index_publications ──


async def test_index_embeds_all_and_records_model(db, store, embedder):
    await load_publications(db, read_csv(SAMPLE_CSV).rows)
    indexed = await index_publications(db, store, embedder, batch_size=8)
    assert indexed == 20
    assert await store.count(PUBLICATIONS) == 20
    assert [len(batch) for batch in embedder.calls] == [8, 8, 4]
    async with db.sessionmaker() as s:
        meta = dict((await s.execute(select(AppMeta.key, AppMeta.value))).all())
        pending = (
            await s.execute(select(func.count()).where(Publication.vector_indexed_at.is_(None)))
        ).scalar_one()
    assert meta == {"embedding_model": "fake-embedding", "embedding_dim": "64"}
    assert pending == 0


async def test_index_resumes_with_only_unindexed_rows(db, store, embedder):
    await load_publications(db, read_csv(SAMPLE_CSV).rows)
    await index_publications(db, store, embedder, batch_size=8)
    async with db.sessionmaker() as s:
        await s.execute(update(Publication).where(Publication.id.in_([1, 2])).values(vector_indexed_at=None))
        await s.commit()
    embedder.calls.clear()
    assert await index_publications(db, store, embedder, batch_size=8) == 2
    assert [len(b) for b in embedder.calls] == [2]


async def test_index_refuses_a_different_embedding_model(db, store, embedder):
    await load_publications(db, read_csv(SAMPLE_CSV).rows)
    await index_publications(db, store, embedder, batch_size=8)
    async with db.sessionmaker() as s:
        await s.execute(update(Publication).values(vector_indexed_at=None))
        await s.commit()
    with pytest.raises(EmbeddingModelChanged) as exc:
        await index_publications(db, store, FakeEmbedder(model="other-model"), batch_size=8)
    assert "reindex" in str(exc.value)


async def test_reindex_rebuilds_with_new_model(db, store, embedder):
    await load_publications(db, read_csv(SAMPLE_CSV).rows)
    await index_publications(db, store, embedder, batch_size=8)
    other = FakeEmbedder(dim=32, model="other-model")
    assert await index_publications(db, store, other, batch_size=50, reindex=True) == 20
    assert await store.collection_dim(PUBLICATIONS) == 32


async def test_vector_payload_contents(seeded, store):
    hits = await store.client.retrieve(PUBLICATIONS, ids=[1], with_payload=True)
    payload = hits[0].payload
    assert payload["publication_id"] == 1
    assert payload["year"] == 2024
    assert payload["authors_norm"] == ["ranganath r", "smith j"]
    assert payload["cluster_label_norm"] == "generative modeling"
    assert payload["source_title_norm"] == "proceedings of machine learning research"


async def test_embedding_text_contains_key_fields(seeded):
    pub = await get_pub(seeded, "2-s2.0-SYN00006")
    text = publication_embedding_text(pub)
    assert "DNA double strand break repair in yeast" in text
    assert "homologous recombination" in text
    assert "Cell" in text


# ── CLI ──


def test_cli_seed_without_vectors(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PORTKEY_API_KEY", "k")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'cli.db'}")
    assert cli_main(["seed", "--csv", str(SAMPLE_CSV), "--skip-vectors"]) == 0
    out = capsys.readouterr().out
    assert "inserted=20" in out
    assert "skipped=3" in out
    assert "duplicates=1" in out


def test_cli_seed_missing_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PORTKEY_API_KEY", "k")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'cli.db'}")
    assert cli_main(["seed", "--csv", str(tmp_path / "nope.csv")]) == 2
    assert "not found" in capsys.readouterr().err


def test_cli_migrate(tmp_path, monkeypatch):
    monkeypatch.setenv("PORTKEY_API_KEY", "k")
    monkeypatch.setenv("DATABASE_URL", f"sqlite+aiosqlite:///{tmp_path / 'm.db'}")
    assert cli_main(["migrate"]) == 0
    assert (tmp_path / "m.db").exists()


def test_fixture_has_real_csv_columns():
    with SAMPLE_CSV.open(newline="", encoding="utf-8") as f:
        header = next(csv.reader(f))
    assert {"eid", "authors", "author_full_names", "cluster_label", "cited_by"} <= set(header)


def test_cli_reindex_targets():
    from app.cli import build_parser, reindex_targets

    assert reindex_targets(build_parser().parse_args(["reindex"])) == (True, True)
    assert reindex_targets(build_parser().parse_args(["reindex", "--documents"])) == (False, True)
    assert reindex_targets(build_parser().parse_args(["reindex", "--publications"])) == (True, False)
