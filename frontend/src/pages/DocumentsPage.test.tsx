import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import type { DocumentInfo } from "../api/types";
import { json, mockApi, renderWithProviders } from "../test/utils";
import { DocumentsPage, pollInterval } from "./DocumentsPage";

const doc = (id: string, extra: Partial<DocumentInfo> = {}): DocumentInfo => ({
  id,
  filename: `${id}.pdf`,
  content_type: "application/pdf",
  size_bytes: 2048,
  status: "ready",
  error: null,
  page_count: 3,
  chunk_count: 4,
  created_at: "2026-10-03T10:00:00Z",
  ...extra,
});

describe("DocumentsPage", () => {
  it("lists documents with their status and failure reason", async () => {
    mockApi({
      "GET /documents": () =>
        json([doc("ready"), doc("busy", { status: "processing" }), doc("scan", { status: "failed", error: "No extractable text found." })]),
    });
    renderWithProviders(<DocumentsPage />);
    const row = (name: string) => screen.getByText(name).closest("li") as HTMLElement;
    await screen.findByText("ready.pdf");
    expect(within(row("ready.pdf")).getByText("Ready")).toBeInTheDocument();
    expect(within(row("ready.pdf")).getByText(/3 pages/)).toBeInTheDocument();
    expect(within(row("busy.pdf")).getByText("Processing")).toBeInTheDocument();
    expect(within(row("scan.pdf")).getByText("No extractable text found.")).toBeInTheDocument();
  });

  it("uploads selected files", async () => {
    const { calls } = mockApi({
      "GET /documents": () => json([]),
      "POST /documents": () => json(doc("new", { status: "processing" }), 202),
    });
    renderWithProviders(<DocumentsPage />);
    const input = await screen.findByLabelText(/upload documents/i);
    await userEvent.upload(input, new File(["hello"], "notes.txt", { type: "text/plain" }));
    await waitFor(() => expect(calls.some((c) => c.method === "POST")).toBe(true));
  });

  it("shows upload errors from the server", async () => {
    mockApi({
      "GET /documents": () => json([]),
      "POST /documents": () => json({ error: { code: "unsupported_file", message: "The file is empty." } }, 400),
    });
    renderWithProviders(<DocumentsPage />);
    await userEvent.upload(await screen.findByLabelText(/upload documents/i), new File([""], "empty.txt", { type: "text/plain" }));
    expect(await screen.findByRole("alert")).toHaveTextContent("empty.txt: The file is empty.");
  });

  it("deletes after an inline confirmation", async () => {
    const { calls } = mockApi({
      "GET /documents": () => json([doc("old")]),
      "DELETE /documents/old": () => json(null, 204),
    });
    renderWithProviders(<DocumentsPage />);
    await userEvent.click(await screen.findByRole("button", { name: "Delete old.pdf" }));
    expect(calls.some((c) => c.method === "DELETE")).toBe(false);
    await userEvent.click(screen.getByRole("button", { name: "Confirm delete old.pdf" }));
    await waitFor(() => expect(calls.some((c) => c.method === "DELETE" && c.url === "/documents/old")).toBe(true));
  });

  it("polls only while something is processing", () => {
    expect(pollInterval([doc("a"), doc("b", { status: "processing" })])).toBe(1500);
    expect(pollInterval([doc("a")])).toBe(false);
    expect(pollInterval(undefined)).toBe(false);
  });
});
