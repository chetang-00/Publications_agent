import pytest
from sqlalchemy import select

from app.db.models import ClusterLabelChange, Publication
from app.rag.vectorstore import PublicationVectorFilter
from app.seed.publications import load_publications, read_csv
from app.tools import build_registry
from app.tools.base import InvalidToolArguments, ToolContext
from app.tools.publications import cluster_label_approval_context
from tests.conftest import SAMPLE_CSV
from tests.fakes import fake_vector


def ids(outcome) -> list[int]:
    assert outcome.status == "ok", outcome.error
    return [item["id"] for item in outcome.result["items"]]


# ── registry wiring ──


def test_publication_tools_are_registered():
    registry = build_registry()
    for name in (
        "search_publications",
        "filter_publications",
        "publication_stats",
        "get_publication",
        "resolve_author",
        "update_cluster_label",
    ):
        assert registry.get(name) is not None
    assert registry.get("update_cluster_label").requires_approval is True
    assert registry.get("filter_publications").requires_approval is False


# ── filter_publications ──


async def test_filter_by_stored_author(run_tool):
    out = await run_tool("filter_publications", author="Ranganath R.", limit=50)
    assert sorted(ids(out)) == [1, 2, 3, 4, 5, 17]
    assert out.result["total"] == 6


async def test_filter_by_surname_only(run_tool):
    assert sorted(ids(await run_tool("filter_publications", author="ranganath", limit=50))) == [
        1,
        2,
        3,
        4,
        5,
        17,
    ]


async def test_filter_author_initials_do_not_merge_different_people(run_tool):
    out = await run_tool("filter_publications", author="Smith J", limit=50)
    assert sorted(ids(out)) == [1, 6, 7, 10, 16, 20]


async def test_filter_by_single_year(run_tool):
    assert sorted(ids(await run_tool("filter_publications", year_from=2021, year_to=2021))) == [5, 9, 18, 20]


async def test_filter_by_keyword_matches_keywords_title_and_abstract(run_tool):
    assert sorted(ids(await run_tool("filter_publications", keyword="CRISPR"))) == [8, 9, 10]
    assert sorted(ids(await run_tool("filter_publications", keyword="dna repair"))) == [6, 10]


async def test_filter_by_cluster_label_case_insensitive(run_tool):
    assert sorted(ids(await run_tool("filter_publications", cluster_label="crispr functional genomics"))) == [
        8,
        9,
        10,
    ]


async def test_filter_unknown_label_finds_unlabelled_papers(run_tool):
    assert sorted(ids(await run_tool("filter_publications", cluster_label="Unknown Label"))) == [
        11,
        12,
        17,
        18,
        19,
    ]


async def test_filter_by_journal_and_document_type(run_tool):
    assert sorted(ids(await run_tool("filter_publications", source_title="nature"))) == [4, 9, 10, 14]
    assert ids(await run_tool("filter_publications", document_type="review")) == [14]


async def test_filters_combine(run_tool):
    out = await run_tool("filter_publications", author="Garcia M.", year_from=2021)
    assert sorted(ids(out)) == [5, 9, 10, 18, 20]


async def test_sort_and_page(run_tool):
    top = await run_tool("filter_publications", sort_by="cited_by", order="desc", limit=3)
    assert ids(top) == [14, 11, 6]
    page = await run_tool("filter_publications", sort_by="cited_by", order="desc", limit=2, offset=2)
    assert ids(page) == [6, 7]
    assert page.result["total"] == 20
    assert page.result["offset"] == 2


async def test_sql_injection_text_matches_nothing(run_tool):
    out = await run_tool("filter_publications", author="x' OR 1=1 --")
    assert out.result["total"] == 0


async def test_like_wildcards_are_literal(run_tool):
    assert (await run_tool("filter_publications", title_contains="%")).result["total"] == 0


async def test_summary_fields(run_tool):
    item = (await run_tool("filter_publications", title_contains="yeast")).result["items"][0]
    assert item == {
        "cite": "[pub:6]",
        "id": 6,
        "title": "DNA double strand break repair in yeast",
        "year": 2016,
        "authors": "Smith J.; Lee K.",
        "source_title": "Cell",
        "cited_by": 120,
        "cluster_label": "DNA Damage and Repair Mechanisms",
        "doi": "10.9999/synthetic.006",
    }


def test_year_range_must_be_ordered():
    registry = build_registry()
    tool = registry.get("filter_publications")
    with pytest.raises(InvalidToolArguments) as exc:
        registry.parse_args(tool, '{"year_from": 2022, "year_to": 2020}')
    assert "year_from" in exc.value.details[0]["msg"]


def test_limit_is_capped():
    registry = build_registry()
    with pytest.raises(InvalidToolArguments):
        registry.parse_args(registry.get("filter_publications"), '{"limit": 500}')


# ── publication_stats ──


async def test_stats_by_year_are_chronological(run_tool):
    out = await run_tool("publication_stats", group_by="year", year_from=2020, sort="key")
    assert out.result["total_publications"] == 12
    assert out.result["ordered_by"] == "year ascending"
    assert [(b["key"], b["count"]) for b in out.result["buckets"]] == [
        (2020, 2),
        (2021, 4),
        (2022, 2),
        (2023, 2),
        (2024, 2),
    ]
    assert out.result["more_groups"] is False


async def test_stats_by_year_keeps_most_recent_years_when_capped(run_tool):
    out = await run_tool("publication_stats", group_by="year", top_n=2, sort="key")
    assert [b["key"] for b in out.result["buckets"]] == [2023, 2024]
    assert out.result["more_groups"] is True


async def test_stats_by_cluster_label(run_tool):
    out = await run_tool("publication_stats", group_by="cluster_label", top_n=3)
    buckets = [(b["key"], b["count"]) for b in out.result["buckets"]]
    assert buckets[0] == ("Unknown Label", 5)
    assert set(buckets[1:]) == {("CRISPR Functional Genomics", 3), ("DNA Damage and Repair Mechanisms", 3)}
    assert out.result["more_groups"] is True


async def test_stats_by_author_with_keyword_filter(run_tool):
    out = await run_tool("publication_stats", group_by="author", keyword="CRISPR")
    first = out.result["buckets"][0]
    assert (first["key"], first["count"]) == ("Garcia M.", 3)
    assert out.result["total_publications"] == 3


async def test_stats_by_keyword_for_author(run_tool):
    out = await run_tool("publication_stats", group_by="keyword", author="Ranganath R.", top_n=50)
    counts = {str(b["key"]).lower(): b["count"] for b in out.result["buckets"]}
    assert counts["causal inference"] == 2
    assert counts["diffusion models"] == 2


async def test_stats_by_journal_and_type(run_tool):
    journals = await run_tool("publication_stats", group_by="source_title", top_n=1)
    assert journals.result["buckets"] == [{"key": "Cell", "count": 4}]
    types = await run_tool("publication_stats", group_by="document_type")
    assert {b["key"]: b["count"] for b in types.result["buckets"]} == {
        "Article": 17,
        "Conference paper": 2,
        "Review": 1,
    }


# ── get_publication ──


async def test_get_publication(run_tool):
    out = await run_tool("get_publication", publication_id=6)
    pub = out.result
    assert pub["title"] == "DNA double strand break repair in yeast"
    assert pub["authors"] == ["Smith J.", "Lee K."]
    assert pub["author_keywords"] == ["DNA repair", "homologous recombination"]
    assert pub["abstract"].startswith("Mechanisms of DNA")


async def test_get_missing_publication_is_a_tool_error(run_tool):
    out = await run_tool("get_publication", publication_id=999)
    assert out.status == "error"
    assert "999" in out.error


# ── resolve_author ──


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Rajesh Ranganath", "Ranganath R."),
        ("Jane Smith", "Smith J.A."),
        ("John Smith", "Smith J."),
        ("lee", "Lee K."),
    ],
)
async def test_resolve_author(run_tool, name, expected):
    out = await run_tool("resolve_author", name=name)
    assert out.result["matches"][0]["author"] == expected


async def test_resolve_author_reports_counts_and_full_name(run_tool):
    match = (await run_tool("resolve_author", name="Rajesh Ranganath")).result["matches"][0]
    assert match["full_name"] == "Ranganath, Rajesh"
    assert match["paper_count"] == 6


async def test_resolve_author_no_match(run_tool):
    assert (await run_tool("resolve_author", name="Zzyzx Qwerty")).result["matches"] == []


# ── search_publications ──


async def test_search_ranks_semantically(run_tool):
    out = await run_tool("search_publications", query="protein folding molecular dynamics")
    assert ids(out)[0] == 13
    assert out.result["items"][0]["abstract_snippet"].startswith("Long timescale")


async def test_search_with_author_and_year_filters(run_tool):
    out = await run_tool("search_publications", query="models", authors=["Ranganath R."], year_from=2021)
    assert set(ids(out)) <= {1, 4, 5, 17}
    assert ids(out)


async def test_search_with_cluster_filter(run_tool):
    out = await run_tool("search_publications", query="genes", cluster_labels=["CRISPR Functional Genomics"])
    assert set(ids(out)) == {8, 9, 10}


async def test_search_before_indexing_explains_how_to_fix(db, store, embedder, settings):
    await load_publications(db, read_csv(SAMPLE_CSV).rows)
    registry = build_registry()
    tool = registry.get("search_publications")
    ctx = ToolContext(db=db, store=store, embedder=embedder, settings=settings)
    out = await registry.execute(tool, registry.parse_args(tool, '{"query": "protein"}'), ctx)
    assert out.status == "error"
    assert "make seed" in out.error


async def test_search_embedding_failure_is_a_tool_error(run_tool, embedder):
    from app.llm.embeddings import EmbeddingError

    embedder.fail_with = EmbeddingError("gateway down")
    out = await run_tool("search_publications", query="protein")
    assert out.status == "error"
    assert "unavailable" in out.error


# ── update_cluster_label ──


async def test_update_cluster_label_updates_sqlite_audit_and_vectors(run_tool, tool_ctx, store):
    out = await run_tool(
        "update_cluster_label", publication_id=11, new_label="Galaxy Formation", reason="clearly astrophysics"
    )
    assert out.result == {
        "publication_id": 11,
        "title": "Galaxy formation in dark matter halos",
        "old_label": None,
        "new_label": "Galaxy Formation",
        "status": "updated",
    }
    async with tool_ctx.db.sessionmaker() as s:
        assert (await s.get(Publication, 11)).cluster_label == "Galaxy Formation"
        change = (await s.execute(select(ClusterLabelChange))).scalar_one()
    assert (change.publication_id, change.new_label, change.run_id) == (11, "Galaxy Formation", "run-test")
    hits = await store.search_publications(
        fake_vector("galaxy"), PublicationVectorFilter(cluster_labels_norm=["galaxy formation"]), limit=5
    )
    assert [h.id for h in hits] == [11]


async def test_update_to_same_label_is_unchanged(run_tool, tool_ctx):
    out = await run_tool(
        "update_cluster_label", publication_id=6, new_label="DNA Damage and Repair Mechanisms", reason="same"
    )
    assert out.result["status"] == "unchanged"
    async with tool_ctx.db.sessionmaker() as s:
        assert (await s.execute(select(ClusterLabelChange))).first() is None


async def test_update_unknown_publication(run_tool):
    out = await run_tool("update_cluster_label", publication_id=999, new_label="X", reason="why not")
    assert out.status == "error"


def test_update_rejects_blank_label():
    registry = build_registry()
    with pytest.raises(InvalidToolArguments):
        registry.parse_args(
            registry.get("update_cluster_label"), '{"publication_id": 1, "new_label": "   ", "reason": "abc"}'
        )


async def test_approval_context_suggests_existing_labels(tool_ctx):
    registry = build_registry()
    args = registry.parse_args(
        registry.get("update_cluster_label"),
        '{"publication_id": 8, "new_label": "CRISPR Functional Genomic", "reason": "typo test"}',
    )
    context = await cluster_label_approval_context(args, tool_ctx)
    assert context["title"] == "Genome wide CRISPR screens in neurons"
    assert context["current_label"] == "CRISPR Functional Genomics"
    assert context["proposed_label"] == "CRISPR Functional Genomic"
    assert context["label_exists"] is False
    assert "CRISPR Functional Genomics" in context["closest_existing_labels"]


async def test_setting_unknown_label_clears_the_label(run_tool, tool_ctx):
    out = await run_tool(
        "update_cluster_label", publication_id=6, new_label="Unknown Label", reason="not sure it fits"
    )
    assert out.result["status"] == "updated"
    async with tool_ctx.db.sessionmaker() as s:
        assert (await s.get(Publication, 6)).cluster_label is None
    unlabelled = await run_tool("filter_publications", cluster_label="Unknown Label")
    assert 6 in ids(unlabelled)
    again = await run_tool(
        "update_cluster_label", publication_id=6, new_label="unknown label", reason="same again"
    )
    assert again.result["status"] == "unchanged"


async def test_stats_are_largest_first_by_default(run_tool):
    out = await run_tool("publication_stats", group_by="year", year_from=2020)
    assert out.result["buckets"][0] == {"key": 2021, "count": 4}
    assert out.result["ordered_by"] == "count, largest first"


async def test_results_carry_a_ready_made_citation(run_tool):
    item = (await run_tool("filter_publications", title_contains="yeast")).result["items"][0]
    assert item["cite"] == "[pub:6]"
    hit = (await run_tool("search_publications", query="protein folding")).result["items"][0]
    assert hit["cite"] == f"[pub:{hit['id']}]"
    detail = (await run_tool("get_publication", publication_id=6)).result
    assert detail["cite"] == "[pub:6]"
