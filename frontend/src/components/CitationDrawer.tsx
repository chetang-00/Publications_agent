import { useQuery } from "@tanstack/react-query";
import { ExternalLink, FileText, Loader2, X } from "lucide-react";
import { useEffect, type ReactNode } from "react";
import { Link } from "react-router";
import { api } from "../api/client";
import type { Citation, Publication } from "../api/types";

const MAX_AUTHORS = 15;

function Shell({ label, onClose, children }: { label: string; onClose: () => void; children: ReactNode }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div className="fixed inset-0 z-40">
      <div className="absolute inset-0 bg-zinc-950/30 backdrop-blur-[1px]" onClick={onClose} aria-hidden />
      <aside
        role="dialog"
        aria-modal="true"
        aria-label={label}
        className="absolute top-0 right-0 flex h-full w-full max-w-xl flex-col bg-white shadow-2xl dark:bg-zinc-900"
      >
        <div className="flex justify-end p-3">
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="rounded-md p-1.5 text-zinc-500 hover:bg-zinc-100 hover:text-zinc-800 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
          >
            <X className="size-5" />
          </button>
        </div>
        <div className="flex-1 overflow-y-auto px-6 pb-8">{children}</div>
      </aside>
    </div>
  );
}

function PublicationDetails({ pub }: { pub: Publication }) {
  const authors = pub.authors.slice(0, MAX_AUTHORS).join(", ");
  const more = pub.authors.length - MAX_AUTHORS;
  const meta = [pub.year, pub.source_title, pub.document_type].filter(Boolean).join(" · ");
  const keywords = [...pub.author_keywords, ...pub.index_keywords].slice(0, 24);
  return (
    <article className="space-y-4">
      <header>
        <h2 className="text-lg leading-snug font-semibold">{pub.title}</h2>
        {meta && <p className="mt-1 text-sm text-zinc-500">{meta}</p>}
      </header>
      {authors && (
        <p className="text-sm">
          {authors}
          {more > 0 && <span className="text-zinc-500"> and {more} more</span>}
        </p>
      )}
      <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1.5 text-sm">
        <dt className="text-zinc-500">Cluster</dt>
        <dd>{pub.cluster_label ?? <span className="text-zinc-400">Unlabelled</span>}</dd>
        {pub.cited_by != null && (
          <>
            <dt className="text-zinc-500">Cited by</dt>
            <dd>{pub.cited_by}</dd>
          </>
        )}
        {pub.publisher && (
          <>
            <dt className="text-zinc-500">Publisher</dt>
            <dd>{pub.publisher}</dd>
          </>
        )}
        <dt className="text-zinc-500">Record</dt>
        <dd>#{pub.id}</dd>
      </dl>
      {(pub.doi || pub.link) && (
        <a
          href={pub.doi ? `https://doi.org/${pub.doi}` : (pub.link ?? undefined)}
          target="_blank"
          rel="noopener noreferrer"
          className="inline-flex items-center gap-1.5 text-sm font-medium text-brand-600 hover:underline dark:text-brand-300"
        >
          {pub.doi ? `doi.org/${pub.doi}` : "Open record"}
          <ExternalLink className="size-3.5" />
        </a>
      )}
      {pub.abstract && (
        <section>
          <h3 className="mb-1 text-xs font-semibold tracking-wide text-zinc-500 uppercase">Abstract</h3>
          <p className="text-sm leading-relaxed">{pub.abstract}</p>
        </section>
      )}
      {keywords.length > 0 && (
        <section>
          <h3 className="mb-1.5 text-xs font-semibold tracking-wide text-zinc-500 uppercase">Keywords</h3>
          <div className="flex flex-wrap gap-1.5">
            {keywords.map((k, i) => (
              <span key={`${k}-${i}`} className="rounded-full bg-zinc-100 px-2 py-0.5 text-xs dark:bg-zinc-800">
                {k}
              </span>
            ))}
          </div>
        </section>
      )}
    </article>
  );
}

function PublicationDrawer({ id, onClose }: { id: string; onClose: () => void }) {
  const query = useQuery({ queryKey: ["publication", id], queryFn: () => api.getPublication(id) });
  return (
    <Shell label={query.data?.title ?? "Publication"} onClose={onClose}>
      {query.isPending && <Loader2 className="size-5 animate-spin text-zinc-400" aria-label="Loading" />}
      {query.isError && (
        <p role="alert" className="text-sm text-rose-600">
          Could not load this publication.
        </p>
      )}
      {query.data && <PublicationDetails pub={query.data} />}
    </Shell>
  );
}

function DocumentDrawer({ citation, onClose }: { citation: Citation; onClose: () => void }) {
  return (
    <Shell label={citation.filename ?? "Document"} onClose={onClose}>
      <div className="space-y-3">
        <div className="flex items-center gap-2">
          <FileText className="size-5 text-brand-600" aria-hidden />
          <h2 className="text-lg font-semibold">{citation.filename ?? "Uploaded document"}</h2>
        </div>
        <p className="text-sm text-zinc-600 dark:text-zinc-400">
          {citation.page ? `Page ${citation.page}, ` : ""}passage {(citation.chunk_index ?? 0) + 1}
        </p>
        <Link to="/documents" className="text-sm font-medium text-brand-600 hover:underline dark:text-brand-300">
          Manage documents
        </Link>
      </div>
    </Shell>
  );
}

export function CitationDrawer({ citation, onClose }: { citation: Citation; onClose: () => void }) {
  return citation.kind === "publication" ? (
    <PublicationDrawer id={citation.id} onClose={onClose} />
  ) : (
    <DocumentDrawer citation={citation} onClose={onClose} />
  );
}
