"""Command line entry points: `python -m app.cli <command>`."""

import argparse
import asyncio
import sys
from pathlib import Path

from app.config import Settings
from app.db.session import Database, run_migrations
from app.llm.embeddings import PortkeyEmbedder
from app.llm.types import LLMError
from app.rag.ingest import IngestionService
from app.rag.vectorstore import VectorStore
from app.seed.publications import EmbeddingModelChanged, index_publications, load_publications, read_csv


def _print_progress(total: int) -> None:
    print(f"  indexed {total} publications", flush=True)


async def _index(settings: Settings, db: Database, *, reindex: bool) -> int:
    store = VectorStore.from_url(settings.qdrant_url)
    embedder = PortkeyEmbedder.from_settings(settings)
    try:
        return await index_publications(
            db, store, embedder, settings.embedding_batch_size, reindex=reindex, progress=_print_progress
        )
    finally:
        await embedder.aclose()
        await store.close()


async def _seed(settings: Settings, csv_path: Path, skip_vectors: bool) -> int:
    result = read_csv(csv_path)
    for line, reason in result.skipped[:20]:
        print(f"  skipped CSV line {line}: {reason}")
    db = Database(settings.database_url)
    try:
        stats = await load_publications(db, result.rows)
        indexed = 0 if skip_vectors else await _index(settings, db, reindex=False)
    finally:
        await db.dispose()
    print(
        f"seed complete: inserted={stats.inserted} updated={stats.updated} unchanged={stats.unchanged} "
        f"skipped={len(result.skipped)} duplicates={result.duplicates} indexed={indexed}"
    )
    return 0


def reindex_targets(args: argparse.Namespace) -> tuple[bool, bool]:
    """(publications, documents); with neither flag given, rebuild both."""
    if not args.publications and not args.documents:
        return True, True
    return args.publications, args.documents


async def _reindex(settings: Settings, publications: bool, documents: bool) -> int:
    db = Database(settings.database_url)
    store = VectorStore.from_url(settings.qdrant_url)
    embedder = PortkeyEmbedder.from_settings(settings)
    try:
        pubs = docs = 0
        if publications:
            pubs = await index_publications(
                db, store, embedder, settings.embedding_batch_size, reindex=True, progress=_print_progress
            )
        if documents:
            service = IngestionService(db, store, embedder, settings.upload_dir, settings.max_upload_bytes)
            docs = await service.reindex_all()
    finally:
        await embedder.aclose()
        await store.close()
        await db.dispose()
    print(f"reindex complete: publications={pubs} document_chunks={docs}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m app.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("migrate", help="apply database migrations")
    seed = sub.add_parser("seed", help="load the publications CSV and build the vector index")
    seed.add_argument("--csv", required=True, type=Path, help="path to papers_with_cluster_labels.csv")
    seed.add_argument("--skip-vectors", action="store_true", help="load SQLite only, skip embeddings")
    reindex = sub.add_parser(
        "reindex", help="rebuild vector indexes from SQLite (e.g. after changing EMBEDDING_MODEL)"
    )
    reindex.add_argument("--publications", action="store_true", help="only the publications index")
    reindex.add_argument("--documents", action="store_true", help="only the uploaded-documents index")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = Settings()
    run_migrations(settings.database_url)
    try:
        if args.command == "migrate":
            print("migrations applied")
            return 0
        if args.command == "seed":
            if not args.csv.is_file():
                print(f"CSV file not found: {args.csv}", file=sys.stderr)
                return 2
            return asyncio.run(_seed(settings, args.csv, args.skip_vectors))
        if args.command == "reindex":
            return asyncio.run(_reindex(settings, *reindex_targets(args)))
    except (EmbeddingModelChanged, LLMError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 2


if __name__ == "__main__":
    sys.exit(main())
