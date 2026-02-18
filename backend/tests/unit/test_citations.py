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
