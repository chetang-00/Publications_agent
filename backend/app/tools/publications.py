"""Publication tools: semantic search, structured filtering, statistics, lookup, author resolution,
and the one write action (cluster label change, approval required)."""

import difflib
import logging
import re
from collections import Counter
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator, model_validator
from sqlalchemy import desc, distinct, func, or_, select

from app.db.models import ClusterLabelChange, Publication, PublicationAuthor, PublicationKeyword
from app.llm.types import LLMError
from app.rag.vectorstore import (
    PUBLICATIONS,
    EmbeddingDimensionMismatch,
    PublicationVectorFilter,
    VectorStoreNotReady,
)
from app.text import normalize
from app.tools.base import Tool, ToolArgs, ToolContext, ToolError
from app.tools.filters import UNKNOWN_LABEL, UNKNOWN_LABEL_NAMES, PublicationFilters, apply_filters

log = logging.getLogger(__name__)

SNIPPET_CHARS = 400
SUMMARY_AUTHORS = 8


# ── shared result models ─────────────────────────────────────────────────────


class PublicationSummary(BaseModel):
    cite: str  # ready-made citation marker for the answer, e.g. "[pub:6]"
    id: int
    title: str
    year: int | None
    authors: str | None
    source_title: str | None
    cited_by: int | None
    cluster_label: str | None
    doi: str | None


def _short_authors(authors: str | None) -> str | None:
    if not authors:
        return None
    names = authors.split("; ")
    if len(names) <= SUMMARY_AUTHORS:
        return authors
    return "; ".join(names[:SUMMARY_AUTHORS]) + f"; et al. ({len(names)} authors)"


def _summary(pub: Publication) -> PublicationSummary:
    return PublicationSummary(
        cite=f"[pub:{pub.id}]",
        id=pub.id,
        title=pub.title,
        year=pub.year,
        authors=_short_authors(pub.authors),
        source_title=pub.source_title,
        cited_by=pub.cited_by,
        cluster_label=pub.cluster_label,
        doi=pub.doi,
    )


def _split(value: str | None) -> list[str]:
    return [part for part in (value or "").split("; ") if part]


# ── filter_publications ──────────────────────────────────────────────────────


class FilterPublicationsArgs(PublicationFilters):
    sort_by: Literal["year", "cited_by", "title"] = Field("year", description="Sort field.")
    order: Literal["asc", "desc"] = Field("desc", description="Sort direction.")
    limit: int = Field(20, ge=1, le=50, description="Maximum rows to return.")
    offset: int = Field(0, ge=0, le=10_000, description="Rows to skip, for paging.")


class FilterPublicationsResult(BaseModel):
    total: int
    offset: int
    items: list[PublicationSummary]


async def filter_publications(args: FilterPublicationsArgs, ctx: ToolContext) -> FilterPublicationsResult:
    column = {"year": Publication.year, "cited_by": Publication.cited_by, "title": Publication.title}[
        args.sort_by
    ]
    ordering = column.desc().nulls_last() if args.order == "desc" else column.asc().nulls_last()
    async with ctx.db.sessionmaker() as session:
        total = (await session.execute(apply_filters(select(func.count(Publication.id)), args))).scalar_one()
        rows = (
            await session.execute(
                apply_filters(select(Publication), args)
                .order_by(ordering, Publication.id)
                .limit(args.limit)
                .offset(args.offset)
            )
        ).scalars()
        return FilterPublicationsResult(total=total, offset=args.offset, items=[_summary(p) for p in rows])


# ── publication_stats ────────────────────────────────────────────────────────


class PublicationStatsArgs(PublicationFilters):
    group_by: Literal["year", "cluster_label", "source_title", "author", "keyword", "document_type"] = Field(
        description="Dimension to count publications by."
    )
    top_n: int = Field(20, ge=1, le=50, description="Maximum number of groups to return.")
    sort: Literal["count", "key"] = Field(
        "count",
        description="'count': largest groups first (for 'which/most/top' questions). "
        "'key': by group value, chronological for years (for trends over time).",
    )


class StatsBucket(BaseModel):
    key: str | int | None
    count: int


class PublicationStatsResult(BaseModel):
    group_by: str
    ordered_by: str
    total_publications: int
    buckets: list[StatsBucket]
    more_groups: bool


async def publication_stats(args: PublicationStatsArgs, ctx: ToolContext) -> PublicationStatsResult:
    matching = apply_filters(select(Publication.id), args).scalar_subquery()
    n = desc("n")

    if args.group_by == "author":
        key = func.min(PublicationAuthor.author)
        stmt = (
            select(key, func.count(distinct(PublicationAuthor.publication_id)).label("n"))
            .where(PublicationAuthor.publication_id.in_(matching))
            .group_by(PublicationAuthor.author_norm)
        )
    elif args.group_by == "keyword":
        key = func.min(PublicationKeyword.keyword)
        stmt = (
            select(key, func.count(distinct(PublicationKeyword.publication_id)).label("n"))
            .where(PublicationKeyword.publication_id.in_(matching))
            .group_by(PublicationKeyword.keyword_norm)
        )
    else:
        key = getattr(Publication, args.group_by)
        stmt = (
            select(key, func.count(Publication.id).label("n"))
            .where(Publication.id.in_(matching))
            .group_by(key)
        )

    async with ctx.db.sessionmaker() as session:
        total = (await session.execute(apply_filters(select(func.count(Publication.id)), args))).scalar_one()
        if args.sort == "key" and args.group_by == "year":
            # Chronological; when there are more years than top_n, keep the most recent ones.
            rows = (await session.execute(stmt.order_by(key.desc().nulls_last()).limit(args.top_n + 1))).all()
            more = len(rows) > args.top_n
            rows = sorted(rows[: args.top_n], key=lambda row: (row[0] is None, row[0] or 0))
            ordered_by = "year ascending"
        elif args.sort == "key":
            rows = (await session.execute(stmt.order_by(key.asc().nulls_last()).limit(args.top_n + 1))).all()
            more = len(rows) > args.top_n
            rows = rows[: args.top_n]
            ordered_by = f"{args.group_by} ascending"
        else:
            rows = (await session.execute(stmt.order_by(n, key).limit(args.top_n + 1))).all()
            more = len(rows) > args.top_n
            rows = rows[: args.top_n]
            ordered_by = "count, largest first"

    def label(value: Any) -> Any:
        if args.group_by == "cluster_label" and value is None:
            return UNKNOWN_LABEL
        return value

    return PublicationStatsResult(
        group_by=args.group_by,
        ordered_by=ordered_by,
        total_publications=total,
        buckets=[StatsBucket(key=label(k), count=count) for k, count in rows],
        more_groups=more,
    )


# ── get_publication ──────────────────────────────────────────────────────────


class GetPublicationArgs(ToolArgs):
    publication_id: int = Field(ge=1, description="Publication id from another tool's results.")


class PublicationDetail(BaseModel):
    cite: str
    id: int
    eid: str
    title: str
    year: int | None
    authors: list[str]
    author_full_names: list[str]
    source_title: str | None
    publisher: str | None
    document_type: str | None
    doi: str | None
    link: str | None
    cited_by: int | None
    open_access: str | None
    affiliations: str | None
    abstract: str | None
    author_keywords: list[str]
    index_keywords: list[str]
    cluster_label: str | None


async def _get_or_error(ctx: ToolContext, publication_id: int) -> Publication:
    async with ctx.db.sessionmaker() as session:
        pub = await session.get(Publication, publication_id)
    if pub is None:
        raise ToolError(f"No publication with id {publication_id}. Use ids returned by other tools.")
    return pub


async def get_publication(args: GetPublicationArgs, ctx: ToolContext) -> PublicationDetail:
    pub = await _get_or_error(ctx, args.publication_id)
    return PublicationDetail(
        cite=f"[pub:{pub.id}]",
        id=pub.id,
        eid=pub.eid,
        title=pub.title,
        year=pub.year,
        authors=_split(pub.authors),
        author_full_names=_split(pub.author_full_names),
        source_title=pub.source_title,
        publisher=pub.publisher,
        document_type=pub.document_type,
        doi=pub.doi,
        link=pub.link,
        cited_by=pub.cited_by,
        open_access=pub.open_access,
        affiliations=(pub.affiliations or "")[:1000] or None,
        abstract=pub.abstract,
        author_keywords=_split(pub.author_keywords),
        index_keywords=_split(pub.index_keywords),
        cluster_label=pub.cluster_label,
    )


# ── resolve_author ───────────────────────────────────────────────────────────


class ResolveAuthorArgs(ToolArgs):
    name: str = Field(
        min_length=2, max_length=100, description="Author name as the user wrote it, e.g. 'Rajesh Ranganath'."
    )
    limit: int = Field(5, ge=1, le=10)


class AuthorMatch(BaseModel):
    author: str
    full_name: str | None
    other_full_names: list[str] = []
    paper_count: int
    score: float


class ResolveAuthorResult(BaseModel):
    query: str
    matches: list[AuthorMatch]


NAME_STOPWORDS = frozenset(
    {"or", "and", "the", "by", "of", "et", "al", "dr", "prof", "mr", "ms", "mrs", "named"}
)
MIN_AUTHOR_SCORE = 0.6


_NAME_WORD = re.compile(r"[^\W\d_][^\W\d_'\-]*")  # unicode letters: "Bühler", "Núñez", "O'Brien"


def name_words(name: str) -> list[str]:
    """Every part of a name, initials included, minus stopwords and SQL/punctuation debris."""
    words = (w.strip("'-") for w in _NAME_WORD.findall(name.lower()))
    return [w for w in words if w and w not in NAME_STOPWORDS]


def name_tokens(name: str) -> list[str]:
    """The parts of a name long enough to search on (initials only help ranking)."""
    return [w for w in name_words(name) if len(w) >= 2]


def _name_forms(author_norm: str, full_names: list[str]) -> list[str]:
    forms = [author_norm]
    for full_name in full_names:
        last, _, first = full_name.lower().partition(",")
        last, first = last.strip(), first.strip()
        forms += [f"{first} {last}".strip(), f"{last} {first}".strip()]
    return forms


async def resolve_author(args: ResolveAuthorArgs, ctx: ToolContext) -> ResolveAuthorResult:
    tokens = name_tokens(args.name)
    if not tokens:
        return ResolveAuthorResult(query=args.name, matches=[])
    # Initials take part in similarity ("S.S. Brody" should prefer "Brody S.S." over "Brody M.").
    query = " ".join(name_words(args.name))

    prefilter = []
    for token in tokens:
        prefilter.append(PublicationAuthor.author_norm.startswith(token, autoescape=True))
        prefilter.append(func.lower(PublicationAuthor.full_name).contains(token, autoescape=True))
    async with ctx.db.sessionmaker() as session:
        variants = (
            await session.execute(
                select(
                    PublicationAuthor.author_norm,
                    PublicationAuthor.author,
                    PublicationAuthor.full_name,
                    func.count(PublicationAuthor.publication_id),
                )
                .where(or_(*prefilter))
                .group_by(
                    PublicationAuthor.author_norm, PublicationAuthor.author, PublicationAuthor.full_name
                )
                .limit(2000)
            )
        ).all()
        norms = list({v[0] for v in variants})[:500]
        totals = dict(
            (
                await session.execute(
                    select(
                        PublicationAuthor.author_norm, func.count(distinct(PublicationAuthor.publication_id))
                    )
                    .where(PublicationAuthor.author_norm.in_(norms))
                    .group_by(PublicationAuthor.author_norm)
                )
            ).all()
        )

    # One candidate per stored author; its spellings of the full name are variants, not people.
    spellings: dict[str, Counter] = {}
    full_names: dict[str, Counter] = {}
    for author_norm, author, full_name, count in variants:
        spellings.setdefault(author_norm, Counter())[author] += count
        if full_name:
            full_names.setdefault(author_norm, Counter())[full_name] += count

    ranked: list[tuple[float, AuthorMatch]] = []
    for author_norm, authors in spellings.items():
        names = [n for n, _ in full_names.get(author_norm, Counter()).most_common()]
        similarity = max(
            difflib.SequenceMatcher(None, query, form).ratio() for form in _name_forms(author_norm, names)
        )
        # A surname hit is strong evidence; the bonus is applied before capping so that two
        # same-surname candidates are still ordered by how well the rest of the name matches.
        raw = similarity + (0.2 if author_norm.split(" ")[0] in tokens else 0.0)
        if raw >= MIN_AUTHOR_SCORE:
            match = AuthorMatch(
                author=authors.most_common(1)[0][0],
                full_name=names[0] if names else None,
                other_full_names=names[1:4],
                paper_count=totals.get(author_norm, 0),
                score=round(min(raw, 1.0), 2),
            )
            ranked.append((raw, match))
    ranked.sort(key=lambda item: (-item[0], -item[1].paper_count, item[1].author))
    return ResolveAuthorResult(query=args.name, matches=[match for _, match in ranked[: args.limit]])


# ── search_publications ──────────────────────────────────────────────────────


class SearchPublicationsArgs(ToolArgs):
    query: str = Field(
        min_length=2, max_length=500, description="What the papers should be about, in natural language."
    )
    year_from: int | None = Field(None, ge=1800, le=2100)
    year_to: int | None = Field(None, ge=1800, le=2100)
    authors: list[str] = Field(
        default_factory=list,
        max_length=5,
        description="Authors in stored form, e.g. ['Ranganath R.'] (use resolve_author first). Matches papers by ANY of them.",
    )
    cluster_labels: list[str] = Field(
        default_factory=list, max_length=5, description="Exact cluster labels (any of)."
    )
    source_title: str | None = Field(
        None, max_length=300, description="Exact journal/conference name, case-insensitive."
    )
    top_k: int = Field(8, ge=1, le=20, description="Number of results.")

    @model_validator(mode="after")
    def _ordered_years(self) -> "SearchPublicationsArgs":
        if self.year_from is not None and self.year_to is not None and self.year_from > self.year_to:
            raise ValueError("year_from must be less than or equal to year_to")
        return self


class PublicationHit(PublicationSummary):
    score: float
    abstract_snippet: str | None


class SearchPublicationsResult(BaseModel):
    query: str
    items: list[PublicationHit]


async def search_publications(args: SearchPublicationsArgs, ctx: ToolContext) -> SearchPublicationsResult:
    try:
        vector = (await ctx.embedder.embed([args.query]))[0]
        hits = await ctx.store.search_publications(
            vector,
            PublicationVectorFilter(
                year_from=args.year_from,
                year_to=args.year_to,
                authors_norm=[normalize(a) for a in args.authors],
                cluster_labels_norm=[normalize(c) for c in args.cluster_labels],
                source_title_norm=normalize(args.source_title) if args.source_title else None,
            ),
            limit=args.top_k,
        )
    except (VectorStoreNotReady, EmbeddingDimensionMismatch) as exc:
        raise ToolError(str(exc)) from exc
    except LLMError as exc:
        raise ToolError(
            f"Semantic search is temporarily unavailable ({exc.message}). Try filter_publications instead."
        ) from exc

    ids = [int(h.id) for h in hits]
    async with ctx.db.sessionmaker() as session:
        pubs = {
            p.id: p
            for p in (await session.execute(select(Publication).where(Publication.id.in_(ids)))).scalars()
        }
    items = []
    for hit in hits:
        pub = pubs.get(int(hit.id))
        if pub is None:  # vector exists for a row deleted since indexing
            continue
        snippet = pub.abstract[:SNIPPET_CHARS] if pub.abstract else None
        items.append(
            PublicationHit(**_summary(pub).model_dump(), score=round(hit.score, 4), abstract_snippet=snippet)
        )
    return SearchPublicationsResult(query=args.query, items=items)


# ── update_cluster_label (approval required) ─────────────────────────────────


class UpdateClusterLabelArgs(ToolArgs):
    publication_id: int = Field(ge=1)
    new_label: str = Field(min_length=1, max_length=120, description="The new cluster label.")
    reason: str = Field(
        min_length=3, max_length=300, description="Why the label should change (shown to the user)."
    )

    @field_validator("new_label", "reason")
    @classmethod
    def _not_blank(cls, value: str) -> str:
        value = " ".join(value.split())
        if not value:
            raise ValueError("must not be blank")
        return value


class UpdateClusterLabelResult(BaseModel):
    publication_id: int
    title: str
    old_label: str | None
    new_label: str
    status: Literal["updated", "unchanged"]


async def _existing_labels(ctx: ToolContext) -> list[str]:
    async with ctx.db.sessionmaker() as session:
        return list(
            (
                await session.execute(
                    select(Publication.cluster_label).where(Publication.cluster_label.is_not(None)).distinct()
                )
            ).scalars()
        )


async def cluster_label_approval_context(args: UpdateClusterLabelArgs, ctx: ToolContext) -> dict[str, Any]:
    pub = await _get_or_error(ctx, args.publication_id)
    labels = await _existing_labels(ctx)
    return {
        "publication_id": pub.id,
        "title": pub.title,
        "current_label": pub.cluster_label,
        "proposed_label": args.new_label,
        "label_exists": args.new_label in labels or normalize(args.new_label) in UNKNOWN_LABEL_NAMES,
        "closest_existing_labels": difflib.get_close_matches(args.new_label, labels, n=3, cutoff=0.6),
        "reason": args.reason,
    }


async def update_cluster_label(args: UpdateClusterLabelArgs, ctx: ToolContext) -> UpdateClusterLabelResult:
    async with ctx.db.sessionmaker() as session:
        pub = await session.get(Publication, args.publication_id)
        if pub is None:
            raise ToolError(f"No publication with id {args.publication_id}.")
        old_label = pub.cluster_label
        # "Unknown Label" means no label, stored as NULL exactly like the seeded data.
        stored = None if normalize(args.new_label) in UNKNOWN_LABEL_NAMES else args.new_label
        if old_label == stored:
            return UpdateClusterLabelResult(
                publication_id=pub.id,
                title=pub.title,
                old_label=old_label,
                new_label=args.new_label,
                status="unchanged",
            )
        pub.cluster_label = stored
        session.add(
            ClusterLabelChange(
                publication_id=pub.id,
                old_label=old_label,
                new_label=args.new_label,
                reason=args.reason,
                run_id=ctx.run_id,
            )
        )
        await session.flush()
        # Keep the vector payload in step; if that fails, the SQLite change is rolled back.
        vectors_indexed = await ctx.store.collection_dim(PUBLICATIONS) is not None
        if vectors_indexed:
            await ctx.store.set_publication_label(pub.id, stored)
        try:
            await session.commit()
        except Exception:
            if vectors_indexed:
                await ctx.store.set_publication_label(pub.id, old_label)
            raise
        log.info(
            "Cluster label changed", extra={"publication_id": pub.id, "old": old_label, "new": args.new_label}
        )
        return UpdateClusterLabelResult(
            publication_id=pub.id,
            title=pub.title,
            old_label=old_label,
            new_label=args.new_label,
            status="updated",
        )


TOOLS: list[Tool] = [
    Tool(
        name="search_publications",
        description=(
            "Semantic search over the publications catalogue (title, abstract, keywords). Use for topic or "
            "concept questions ('papers about protein folding'). Optional filters narrow by year, author, "
            "cluster label or journal. Returns ranked papers with ids for citation."
        ),
        args_model=SearchPublicationsArgs,
        result_model=SearchPublicationsResult,
        handler=search_publications,
    ),
    Tool(
        name="filter_publications",
        description=(
            "Exact, structured listing of publications with filters (title text, keyword, author, year range, "
            "cluster label, journal, document type), sorting and paging. Returns the total match count plus "
            "a page of papers. Use for 'list/show papers by X in 2021', exact counts, and rankings such as the "
            "most cited papers (sort_by='cited_by', order='desc') or the newest papers (sort_by='year')."
        ),
        args_model=FilterPublicationsArgs,
        result_model=FilterPublicationsResult,
        handler=filter_publications,
    ),
    Tool(
        name="publication_stats",
        description=(
            "Count publications grouped by year, cluster_label, source_title, author, keyword or document_type, "
            "with the same filters as filter_publications. Use for 'which/most/top' questions (default "
            "sort='count', largest first) and for trends over time (sort='key': chronological years; if there "
            "are more years than top_n the most recent are returned)."
        ),
        args_model=PublicationStatsArgs,
        result_model=PublicationStatsResult,
        handler=publication_stats,
    ),
    Tool(
        name="get_publication",
        description="Full record of one publication (all authors, abstract, keywords, DOI, link) by id.",
        args_model=GetPublicationArgs,
        result_model=PublicationDetail,
        handler=get_publication,
    ),
    Tool(
        name="resolve_author",
        description=(
            "Map an author name as a person would write it ('Rajesh Ranganath', 'Dr. Smith') to the stored "
            "author forms ('Ranganath R.') with paper counts. Call this before filtering by an author whose "
            "stored form you do not know, or when an author filter returned nothing."
        ),
        args_model=ResolveAuthorArgs,
        result_model=ResolveAuthorResult,
        handler=resolve_author,
    ),
    Tool(
        name="update_cluster_label",
        description=(
            "Change one publication's research cluster label. This modifies data, so the user must approve it "
            "in the UI before it runs; explain the change and reason. Prefer existing labels."
        ),
        args_model=UpdateClusterLabelArgs,
        result_model=UpdateClusterLabelResult,
        handler=update_cluster_label,
        requires_approval=True,
        approval_context=cluster_label_approval_context,
    ),
]
