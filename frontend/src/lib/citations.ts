import type { Citation } from "../api/types";

const MARKER = /\s?\[(pub:\d+|doc:[0-9a-fA-F-]{36}:\d+)\]/g;
const HREF_PREFIX = "#cite-";

/**
 * Replace citation markers with numbered Markdown links ("[1](#cite-pub:6)") so the Markdown
 * renderer can draw them as chips. Numbers follow the order of the verified citation list.
 */
export function linkifyCitations(content: string, citations: Citation[]): string {
  const numbers = new Map(citations.map((c, i) => [c.marker.toLowerCase(), i + 1]));
  return content.replace(MARKER, (match, marker: string) => {
    const n = numbers.get(marker.toLowerCase());
    if (n === undefined) return "";
    const space = match.startsWith(" ") ? " " : "";
    return `${space}[${n}](${HREF_PREFIX}${marker})`;
  });
}

export function parseCitationHref(
  href: string | undefined,
  citations: Citation[],
): { citation: Citation; number: number } | null {
  if (!href?.startsWith(HREF_PREFIX)) return null;
  const marker = href.slice(HREF_PREFIX.length).toLowerCase();
  const index = citations.findIndex((c) => c.marker.toLowerCase() === marker);
  const citation = citations[index];
  return citation ? { citation, number: index + 1 } : null;
}
