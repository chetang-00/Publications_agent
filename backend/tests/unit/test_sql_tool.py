import pytest

import app.tools.sql as sql_tool
from app.tools import build_registry


async def rows(run_tool, sql: str, **kwargs):
    out = await run_tool("run_readonly_sql", sql=sql, **kwargs)
    assert out.status == "ok", out.error
    return out.result


def test_sql_tool_is_registered_and_describes_schema():
    tool = build_registry().get("run_readonly_sql")
    assert tool is not None
    assert "publication_authors" in tool.description
    assert "author_norm" in tool.description


async def test_simple_select(run_tool):
    result = await rows(run_tool, "SELECT count(*) AS n FROM publications")
    assert result == {"columns": ["n"], "rows": [[20]], "row_count": 1, "truncated": False}


async def test_join_across_publication_tables(run_tool):
    result = await rows(
        run_tool,
        "SELECT a.author, count(*) AS papers FROM publication_authors a "
        "JOIN publications p ON p.id = a.publication_id WHERE p.year >= 2021 "
        "GROUP BY a.author_norm ORDER BY papers DESC, a.author LIMIT 1",
    )
    assert result["rows"] == [["Garcia M.", 5]]  # papers 5, 9, 10, 18, 20


async def test_cte_and_trailing_semicolon(run_tool):
    result = await rows(
        run_tool,
        "WITH recent AS (SELECT * FROM publications WHERE year >= 2023) SELECT count(*) FROM recent;",
    )
    assert result["rows"] == [[4]]


async def test_keyword_table_is_readable(run_tool):
    result = await rows(
        run_tool,
        "SELECT count(DISTINCT publication_id) FROM publication_keywords WHERE keyword_norm LIKE '%crispr%'",
    )
    assert result["rows"] == [[3]]


@pytest.mark.parametrize(
    "sql",
    [
        "INSERT INTO publications (eid, title, updated_at) VALUES ('x', 'x', '2020-01-01')",
        "UPDATE publications SET title = 'hacked'",
        "DELETE FROM publications",
        "DROP TABLE publications",
        "ATTACH DATABASE ':memory:' AS other",
        "PRAGMA table_info(publications)",
        "SELECT * FROM messages",
        "SELECT * FROM conversations",
        "SELECT name FROM sqlite_master",
        "SELECT 1; DROP TABLE publications",
        "SELECT load_extension('evil')",
    ],
)
async def test_dangerous_or_out_of_scope_sql_is_rejected(run_tool, sql):
    out = await run_tool("run_readonly_sql", sql=sql)
    assert out.status == "error"
    assert out.error
    # The data is untouched whatever was attempted.
    assert (await rows(run_tool, "SELECT count(*) FROM publications WHERE title != 'hacked'"))["rows"] == [
        [20]
    ]


async def test_rejection_message_explains_the_rules(run_tool):
    out = await run_tool("run_readonly_sql", sql="SELECT * FROM messages")
    assert "publications" in out.error and "SELECT" in out.error


async def test_row_cap_and_truncated_flag(run_tool):
    capped = await rows(run_tool, "SELECT id FROM publications ORDER BY id", limit=5)
    assert capped["rows"] == [[1], [2], [3], [4], [5]]
    assert capped["truncated"] is True
    full = await rows(run_tool, "SELECT id FROM publications", limit=50)
    assert full["row_count"] == 20
    assert full["truncated"] is False


async def test_runaway_query_is_aborted(run_tool, monkeypatch):
    monkeypatch.setattr(sql_tool, "TIME_LIMIT_SECONDS", 0.2)
    out = await run_tool(
        "run_readonly_sql",
        sql="WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT count(*) FROM c",
    )
    assert out.status == "error"
    assert "time limit" in out.error


async def test_syntax_error_is_reported(run_tool):
    out = await run_tool("run_readonly_sql", sql="SELEC title FROM publications")
    assert out.status == "error"
    assert "SQL error" in out.error


async def test_long_text_values_are_shortened(run_tool):
    result = await rows(run_tool, "SELECT printf('%.3000c', 'x') AS long_text")
    assert len(result["rows"][0][0]) <= 1001
