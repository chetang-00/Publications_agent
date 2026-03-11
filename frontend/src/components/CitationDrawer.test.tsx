import { screen, within } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { json, mockApi, renderWithProviders } from "../test/utils";
import { CitationDrawer } from "./CitationDrawer";

const DOC = "3f2b8c1e-1111-4a5b-9c9d-0123456789ab";

describe("CitationDrawer for a document passage", () => {
  it("shows the cited passage with its file and page", async () => {
    mockApi({
      [`GET /documents/${DOC}/chunks/2`]: () =>
        json({ document_id: DOC, filename: "trial.pdf", chunk_index: 2, page: 5, text: "We enrolled 120 adult patients." }),
    });
    renderWithProviders(
      <CitationDrawer
        citation={{ kind: "document", id: DOC, marker: `doc:${DOC}:2`, filename: "trial.pdf", page: 5, chunk_index: 2 }}
        onClose={() => {}}
      />,
    );
    const drawer = screen.getByRole("dialog", { name: "trial.pdf" });
    expect(await within(drawer).findByText("We enrolled 120 adult patients.")).toBeInTheDocument();
    expect(within(drawer).getByText(/Page 5/)).toBeInTheDocument();
  });

  it("explains when the passage is gone", async () => {
    mockApi({
      [`GET /documents/${DOC}/chunks/2`]: () => json({ error: { code: "not_found", message: "Passage not found." } }, 404),
    });
    renderWithProviders(
      <CitationDrawer citation={{ kind: "document", id: DOC, marker: `doc:${DOC}:2`, filename: "trial.pdf", page: 5, chunk_index: 2 }} onClose={() => {}} />,
    );
    expect(await screen.findByRole("alert")).toHaveTextContent(/no longer available/i);
  });
});
