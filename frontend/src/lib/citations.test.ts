import { describe, expect, it } from "vitest";
import type { Citation } from "../api/types";
import { linkifyCitations, parseCitationHref } from "./citations";

const DOC = "3f2b8c1e-1111-4a5b-9c9d-0123456789ab";
const citations: Citation[] = [
  { kind: "publication", id: "6", marker: "pub:6", title: "Yeast" },
  { kind: "document", id: DOC, marker: `doc:${DOC}:2`, filename: "trial.pdf", page: 5, chunk_index: 2 },
];

describe("linkifyCitations", () => {
  it("turns verified markers into numbered citation links", () => {
    expect(linkifyCitations(`Repair [pub:6], trial [doc:${DOC}:2], again [pub:6].`, citations)).toBe(
      `Repair [1](#cite-pub:6), trial [2](#cite-doc:${DOC}:2), again [1](#cite-pub:6).`,
    );
  });

  it("drops markers that are not in the verified list", () => {
    expect(linkifyCitations("Claim [pub:99].", citations)).toBe("Claim.");
  });

  it("leaves text without markers untouched", () => {
    expect(linkifyCitations("No citations here.", [])).toBe("No citations here.");
  });
});

describe("parseCitationHref", () => {
  it("finds the citation for a link", () => {
    expect(parseCitationHref("#cite-pub:6", citations)).toEqual({ citation: citations[0], number: 1 });
    expect(parseCitationHref(`#cite-doc:${DOC}:2`, citations)?.number).toBe(2);
  });

  it("returns null for ordinary links", () => {
    expect(parseCitationHref("https://example.org", citations)).toBeNull();
    expect(parseCitationHref(undefined, citations)).toBeNull();
  });
});
