"""`run_readonly_sql`: a guarded fallback for questions the typed tools cannot express.

Defence in depth, each layer sufficient on its own for writes:
1. the connection is opened read-only (`mode=ro`);
2. an authorizer allows only SELECT, reads of the three publication tables, recursion and
   harmless functions — every other action (writes, ATTACH, PRAGMA, other tables) is denied;
3. the statement is wrapped as a sub-select, so a second statement is a syntax error;
4. a progress handler aborts queries that run past the time limit; rows are capped.
"""

import asyncio
import sqlite3
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from app.tools.base import Tool, ToolArgs, ToolContext, ToolError

ALLOWED_TABLES = frozenset({"publications", "publication_authors", "publication_keywords"})
DENIED_FUNCTIONS = frozenset({"load_extension", "readfile", "writefile", "edit", "fts3_tokenizer"})
TIME_LIMIT_SECONDS = 3.0
MAX_VALUE_CHARS = 1000

SCHEMA_HELP = """\
Tables (SQLite):
- publications(id, eid, title, year, authors, author_full_names, source_title, publisher,
  document_type, doi, link, cited_by, open_access, affiliations, authors_with_affiliations,
  abstract, author_keywords, index_keywords, cluster_label)  -- authors/keywords are '; '-separated text;
  cluster_label is NULL for unlabelled papers
- publication_authors(publication_id, position, author, author_norm, full_name)  -- one row per author;
  author like 'Ranganath R.', author_norm lowercase without trailing dot like 'ranganath r'
- publication_keywords(publication_id, keyword, keyword_norm, kind)  -- kind is 'author' or 'index';
  keyword_norm is lowercase"""

_REJECTED = (
    "Query rejected: only a single SELECT statement reading the publications, publication_authors "
    "and publication_keywords tables is allowed."
)


class RunReadonlySqlArgs(ToolArgs):
    sql: str = Field(
        min_length=6, max_length=4000, description="One SQLite SELECT statement (WITH ... SELECT is fine)."
    )
    limit: int = Field(50, ge=1, le=200, description="Maximum rows to return.")


class RunReadonlySqlResult(BaseModel):
    columns: list[str]
    rows: list[list[Any]]
    row_count: int
    truncated: bool


def _authorizer(real_tables: frozenset[str]):
    def check(action: int, arg1: str | None, arg2: str | None, _db: str | None, _trigger: str | None) -> int:
        if action in (sqlite3.SQLITE_SELECT, sqlite3.SQLITE_RECURSIVE):
            return sqlite3.SQLITE_OK
        if action == sqlite3.SQLITE_READ:
            table = (arg1 or "").lower()
            # Names that are not real tables are CTEs or sub-query aliases over allowed tables.
            if table in ALLOWED_TABLES or (table and table not in real_tables):
                return sqlite3.SQLITE_OK
            return sqlite3.SQLITE_DENY
        if action == sqlite3.SQLITE_FUNCTION:
            return sqlite3.SQLITE_DENY if (arg2 or "").lower() in DENIED_FUNCTIONS else sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    return check


def _clean_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return f"<{len(value)} bytes>"
    if isinstance(value, str) and len(value) > MAX_VALUE_CHARS:
        return value[:MAX_VALUE_CHARS] + "…"
    return value


def _run(path: Path, sql: str, limit: int, time_limit: float) -> RunReadonlySqlResult:
    statement = sql.strip().rstrip(";").strip()
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5, check_same_thread=False)
    try:
        real_tables = frozenset(
            name.lower()
            for (name,) in conn.execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")
        ) | {"sqlite_master", "sqlite_schema", "sqlite_temp_master"}
        conn.set_authorizer(_authorizer(real_tables))
        deadline = time.monotonic() + time_limit
        conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
        try:
            cursor = conn.execute(f"SELECT * FROM ({statement}) LIMIT ?", (limit + 1,))
            columns = [d[0] for d in cursor.description or []]
            fetched = cursor.fetchall()
        except sqlite3.DatabaseError as exc:
            message = str(exc)
            if "interrupted" in message:
                raise ToolError(
                    f"Query exceeded the {time_limit:g}s time limit. Add filters or simplify it."
                ) from None
            if "not authorized" in message or "prohibited" in message:
                raise ToolError(_REJECTED) from None
            raise ToolError(f"SQL error: {message}. {_REJECTED}") from None
    finally:
        conn.close()

    truncated = len(fetched) > limit
    data = [[_clean_value(v) for v in row] for row in fetched[:limit]]
    return RunReadonlySqlResult(columns=columns, rows=data, row_count=len(data), truncated=truncated)


async def run_readonly_sql(args: RunReadonlySqlArgs, ctx: ToolContext) -> RunReadonlySqlResult:
    return await asyncio.to_thread(_run, ctx.db.sqlite_path, args.sql, args.limit, TIME_LIMIT_SECONDS)


TOOLS: list[Tool] = [
    Tool(
        name="run_readonly_sql",
        description=(
            "Run ONE read-only SQLite SELECT over the publication tables, for questions the other tools cannot "
            "express (unusual aggregations, co-authorship, cross-field conditions). Prefer the typed tools when "
            "they fit. Use LOWER(...) LIKE for case-insensitive text matching. Include publications.id in the "
            "output when you will cite papers.\n" + SCHEMA_HELP
        ),
        args_model=RunReadonlySqlArgs,
        result_model=RunReadonlySqlResult,
        handler=run_readonly_sql,
    )
]
