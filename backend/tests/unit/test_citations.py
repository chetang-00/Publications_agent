from app.agent.citations import CitationTracker

DOC = "3f2b8c1e-1111-4a5b-9c9d-0123456789ab"


def tracker_with_results() -> CitationTracker:
    t = CitationTracker()
    t.record_tool_result(
        "search_publications",
        {"items": [{"id": 6, "title": "Yeast repair"}, {"id": 7, "title": "Checkpoints"}]},
    )
    t.record_tool_result(
        "filter_publications", {"total": 1, "offset": 0, "items": [{"id": 9, "title": "CRISPRi"}]}
    )
    t.record_tool_result("get_publication", {"id": 13, "title": "Protein folding"})
    t.record_tool_result(
        "update_cluster_label", {"publication_id": 11, "title": "Galaxies", "status": "updated"}
    )
    t.record_tool_result(
        "search_documents",
        {
            "chunks": [
                {"document_id": DOC, "chunk_index": 2, "page": 5, "filename": "trial.pdf", "text": "..."}
            ]
        },
    )
    t.record_tool_result("run_readonly_sql", {"columns": ["id", "title"], "rows": [[14, "Deep protein"]]})
    t.record_tool_result("publication_stats", {"buckets": [{"key": 2020, "count": 3}]})
    return t


def test_verified_citations_are_kept_in_order_without_duplicates():
    content, citations, unverified = tracker_with_results().finalize(
        f"Repair [pub:6] and folding [pub:13]; again [pub:6]. Trial [doc:{DOC}:2]."
    )
    assert content == f"Repair [pub:6] and folding [pub:13]; again [pub:6]. Trial [doc:{DOC}:2]."
    assert [(c.kind, c.id) for c in citations] == [
        ("publication", "6"),
        ("publication", "13"),
        ("document", DOC),
    ]
    assert citations[0].title == "Yeast repair"
    assert (citations[2].filename, citations[2].page, citations[2].chunk_index) == ("trial.pdf", 5, 2)
    assert unverified == []


def test_ids_from_every_tool_shape_are_trusted():
    _, citations, unverified = tracker_with_results().finalize("[pub:7] [pub:9] [pub:11] [pub:14]")
    assert [c.id for c in citations] == ["7", "9", "11", "14"]
    assert unverified == []


def test_unverified_markers_are_stripped():
    content, citations, unverified = tracker_with_results().finalize(
        f"Made up [pub:999]. Real [pub:6]. Wrong chunk [doc:{DOC}:9]."
    )
    assert content == "Made up. Real [pub:6]. Wrong chunk."
    assert [c.id for c in citations] == ["6"]
    assert unverified == ["pub:999", f"doc:{DOC}:9"]


def test_lenient_marker_spacing_is_normalised():
    content, citations, _ = tracker_with_results().finalize("Spaced [pub: 6] and [ pub:13 ].")
    assert content == "Spaced [pub:6] and [pub:13]."
    assert len(citations) == 2


def test_citations_from_earlier_answers_are_trusted():
    t = CitationTracker()
    t.record_citations([{"kind": "publication", "id": "6", "title": "Yeast repair", "marker": "pub:6"}])
    _, citations, unverified = t.finalize("As before [pub:6].")
    assert [c.title for c in citations] == ["Yeast repair"]
    assert unverified == []


def test_tool_errors_and_unknown_shapes_are_ignored():
    t = CitationTracker()
    t.record_tool_result("search_publications", None)
    t.record_tool_result("run_readonly_sql", {"columns": ["n"], "rows": [[5]]})
    t.record_tool_result("something_else", {"items": "not a list"})
    assert t.finalize("[pub:5]")[1] == []


def test_grouped_markers_are_verified_one_by_one():
    content, citations, unverified = tracker_with_results().finalize("Claim [pub:6, pub:999].")
    assert content == "Claim [pub:6]."
    assert [c.id for c in citations] == ["6"]
    assert unverified == ["pub:999"]


def test_grouped_markers_are_split_into_single_markers():
    t = tracker_with_results()
    assert t.finalize("A [pub:6; pub:13].")[0] == "A [pub:6][pub:13]."
    assert t.finalize("B [pub: 6, 13].")[0] == "B [pub:6][pub:13]."
    assert t.finalize(f"C [doc:{DOC}:2, pub:7].")[0] == f"C [doc:{DOC}:2][pub:7]."


def test_group_of_only_unverified_markers_is_removed():
    content, citations, unverified = tracker_with_results().finalize("Made up [pub:900, pub:901].")
    assert content == "Made up."
    assert citations == []
    assert unverified == ["pub:900", "pub:901"]


def test_prompt_asks_for_one_id_per_bracket():
    from datetime import date

    from app.agent.prompts import build_system_prompt

    assert "[pub:12][pub:31]" in build_system_prompt([], date(2026, 10, 3))


def test_uncited_retrieved_titles_get_their_citation():
    t = tracker_with_results()
    t.record_tool_result("get_publication", {"id": 20, "title": "Replication stress and genome instability"})
    content, citations, _ = t.finalize(
        "| Title | Year |\n|---|---|\n| Replication stress and genome instability | 2021 |"
    )
    assert "Replication stress and genome instability [pub:20]" in content
    assert [c.id for c in citations] == ["20"]


def test_titles_already_cited_or_too_short_are_left_alone():
    t = tracker_with_results()
    t.record_tool_result("get_publication", {"id": 21, "title": "Short"})
    t.record_tool_result("get_publication", {"id": 22, "title": "A sufficiently long publication title"})
    content, citations, _ = t.finalize("Short. A sufficiently long publication title [pub:22].")
    assert content == "Short. A sufficiently long publication title [pub:22]."
    assert [c.id for c in citations] == ["22"]


def test_prompt_points_the_model_at_the_cite_field():
    from datetime import date

    from app.agent.prompts import build_system_prompt

    prompt = build_system_prompt([], date(2026, 10, 3))
    assert "`cite`" in prompt
    assert "table" in prompt


HANDBOOK = "Lab handbook. Freezer policy: samples are stored at -80 C. The freezer alarm code is 7731."


def tracker_with_passages() -> CitationTracker:
    t = CitationTracker()
    t.record_tool_result(
        "search_documents",
        {
            "chunks": [
                {
                    "document_id": DOC,
                    "chunk_index": 0,
                    "page": None,
                    "filename": "handbook.md",
                    "text": HANDBOOK,
                },
                {
                    "document_id": DOC,
                    "chunk_index": 1,
                    "page": None,
                    "filename": "handbook.md",
                    "text": "Visitor parking is available behind building seven on weekdays.",
                },
            ]
        },
    )
    return t


def test_answer_drawn_from_a_passage_gets_its_citation():
    content, citations, _ = tracker_with_passages().finalize(
        "The handbook describes the freezer policy: samples are stored at -80 C and the alarm code is 7731."
    )
    assert content.endswith(f"[doc:{DOC}:0]")
    assert [c.marker for c in citations] == [f"doc:{DOC}:0"]


def test_unrelated_passages_are_not_attached():
    content, citations, _ = tracker_with_passages().finalize(
        "CRISPR is a genome editing technique used widely."
    )
    assert citations == []
    assert "[doc:" not in content


def test_answers_that_already_cite_documents_are_left_alone():
    content, citations, _ = tracker_with_passages().finalize(
        f"Samples are stored at -80 C and the freezer alarm code is 7731 [doc:{DOC}:0]."
    )
    assert content.count("[doc:") == 1
    assert len(citations) == 1


def test_prompt_says_where_the_marker_goes():
    from datetime import date

    from app.agent.prompts import build_system_prompt

    assert "never as a separate" in build_system_prompt([], date(2026, 10, 3))
