"""Structured publication filters shared by `filter_publications` and `publication_stats`.

Every value reaches SQL as a bound parameter; LIKE wildcards in user text are escaped.
"""

from pydantic import Field, model_validator
from sqlalchemy import Select, exists, func, or_

from app.db.models import Publication, PublicationAuthor, PublicationKeyword
from app.text import normalize
from app.tools.base import ToolArgs

UNKNOWN_LABEL = "Unknown Label"


class PublicationFilters(ToolArgs):
    title_contains: str | None = Field(
        None, max_length=200, description="Case-insensitive text the title must contain."
    )
    keyword: str | None = Field(
        None,
        max_length=200,
        description="Topic word or phrase. Matches author/index keywords, the title or the abstract (case-insensitive).",
    )
    author: str | None = Field(
        None,
        max_length=100,
        description="Author in stored form 'Surname I.' (e.g. 'Ranganath R.'), or a surname alone. "
        "For a full name like 'Rajesh Ranganath', call resolve_author first.",
    )
    year_from: int | None = Field(
        None, ge=1800, le=2100, description="Earliest publication year (inclusive)."
    )
    year_to: int | None = Field(None, ge=1800, le=2100, description="Latest publication year (inclusive).")
    cluster_label: str | None = Field(
        None,
        max_length=200,
        description=f"Research cluster label, exact but case-insensitive. Use '{UNKNOWN_LABEL}' for unlabelled papers.",
    )
    source_title: str | None = Field(
        None, max_length=300, description="Journal or conference name; case-insensitive substring."
    )
    document_type: str | None = Field(
        None, max_length=64, description="e.g. 'Article', 'Review', 'Conference paper', 'Book chapter'."
    )

    @model_validator(mode="after")
    def _ordered_years(self) -> "PublicationFilters":
        if self.year_from is not None and self.year_to is not None and self.year_from > self.year_to:
            raise ValueError("year_from must be less than or equal to year_to")
        return self


def _contains(column, text: str):
    return func.lower(column).contains(text.lower(), autoescape=True)


def apply_filters(stmt: Select, filters: PublicationFilters) -> Select:
    conditions = []
    if filters.title_contains:
        conditions.append(_contains(Publication.title, filters.title_contains))
    if filters.keyword:
        keyword = normalize(filters.keyword)
        conditions.append(
            or_(
                exists().where(
                    PublicationKeyword.publication_id == Publication.id,
                    PublicationKeyword.keyword_norm.contains(keyword, autoescape=True),
                ),
                _contains(Publication.title, keyword),
                _contains(Publication.abstract, keyword),
            )
        )
    if filters.author:
        author = normalize(filters.author)
        conditions.append(
            exists().where(
                PublicationAuthor.publication_id == Publication.id,
                or_(
                    PublicationAuthor.author_norm == author,
                    PublicationAuthor.author_norm.startswith(f"{author} ", autoescape=True),
                    _contains(PublicationAuthor.full_name, author),
                ),
            )
        )
    if filters.year_from is not None:
        conditions.append(Publication.year >= filters.year_from)
    if filters.year_to is not None:
        conditions.append(Publication.year <= filters.year_to)
    if filters.cluster_label:
        label = filters.cluster_label.strip()
        if normalize(label) in ("unknown label", "unknown", "unlabelled", "unlabeled", "none"):
            conditions.append(Publication.cluster_label.is_(None))
        else:
            conditions.append(func.lower(Publication.cluster_label) == label.lower())
    if filters.source_title:
        conditions.append(_contains(Publication.source_title, filters.source_title.strip()))
    if filters.document_type:
        conditions.append(func.lower(Publication.document_type) == filters.document_type.strip().lower())
    return stmt.where(*conditions)
