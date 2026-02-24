import { useMemo } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import type { Citation } from "../api/types";
import { linkifyCitations, parseCitationHref } from "../lib/citations";

function citationTitle(c: Citation): string {
  if (c.kind === "publication") return c.title ?? `Publication ${c.id}`;
  return [c.filename ?? "Document", c.page ? `page ${c.page}` : null].filter(Boolean).join(", ");
}

export function CitationChip({ number, citation, onClick }: { number: number; citation: Citation; onClick?: () => void }) {
  return (
    <button
      type="button"
      aria-label={`Citation ${number}`}
      title={citationTitle(citation)}
      onClick={onClick}
      className="mx-0.5 inline-flex h-[1.15rem] min-w-[1.15rem] -translate-y-px items-center justify-center rounded-full bg-brand-100 px-1 align-middle text-[10px] font-bold text-brand-700 hover:bg-brand-600 hover:text-white dark:bg-brand-700/40 dark:text-brand-200"
    >
      {number}
    </button>
  );
}

/** An answer: Markdown (GFM tables/lists) with citation markers drawn as clickable chips. */
export function Markdown({
  content,
  citations,
  onCitation,
}: {
  content: string;
  citations: Citation[];
  onCitation?: (citation: Citation) => void;
}) {
  const text = useMemo(() => linkifyCitations(content, citations), [content, citations]);
  return (
    <div className="markdown">
      <ReactMarkdown
        remarkPlugins={[remarkGfm]}
        components={{
          a: ({ href, children }) => {
            const hit = parseCitationHref(href, citations);
            if (hit) {
              return <CitationChip number={hit.number} citation={hit.citation} onClick={() => onCitation?.(hit.citation)} />;
            }
            return (
              <a href={href} target="_blank" rel="noopener noreferrer">
                {children}
              </a>
            );
          },
        }}
      >
        {text}
      </ReactMarkdown>
    </div>
  );
}
