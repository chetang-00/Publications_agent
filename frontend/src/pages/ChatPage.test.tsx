import { screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { Route, Routes } from "react-router";
import { describe, expect, it } from "vitest";
import type { AgentEvent, ConversationDetail } from "../api/types";
import { json, mockApi, renderWithProviders, sse } from "../test/utils";
import { ChatRoute } from "./ChatPage";

const emptyDetail: ConversationDetail = {
  conversation: { id: "c1", title: "New conversation", created_at: "2026-10-03T10:00:00Z", updated_at: "2026-10-03T10:00:00Z" },
  messages: [],
  runs: [],
  tool_calls: [],
  documents: [],
};

const answeredDetail: ConversationDetail = {
  ...emptyDetail,
  conversation: { ...emptyDetail.conversation, title: "What is paper 6?" },
  runs: [
    {
      id: "r1", status: "completed", steps: 2, model: "m", prompt_tokens: 1, completion_tokens: 1,
      error_code: null, error: null, started_at: "2026-10-03T10:00:00Z", finished_at: null, pending_approval: null,
    },
  ],
  messages: [
    { id: 1, run_id: "r1", role: "user", content: "What is paper 6?", citations: [], created_at: "" },
    {
      id: 2, run_id: "r1", role: "assistant", content: "It studies yeast [pub:6].",
      citations: [{ kind: "publication", id: "6", marker: "pub:6", title: "DNA repair in yeast" }], created_at: "",
    },
  ],
  tool_calls: [
    { run_id: "r1", call_id: "t1", step: 1, name: "get_publication", arguments: { publication_id: 6 }, status: "ok", result_preview: "{}", error: null, duration_ms: 9 },
  ],
};

const streamed: AgentEvent[] = [
  { type: "run_started", run_id: "r1", conversation_id: "c1", user_message_id: 1 },
  { type: "tool_call_started", tool_call_id: "t1", name: "get_publication", arguments: { publication_id: 6 }, step: 1 },
  { type: "tool_call_finished", tool_call_id: "t1", name: "get_publication", status: "ok", result_preview: "{}", duration_ms: 9, step: 1 },
  { type: "token", text: "It studies ", step: 2 },
  { type: "token", text: "yeast [pub:6].", step: 2 },
  {
    type: "message_completed", run_id: "r1", message_id: 2, content: "It studies yeast [pub:6].",
    citations: [{ kind: "publication", id: "6", marker: "pub:6", title: "DNA repair in yeast" }],
    usage: { prompt_tokens: 1, completion_tokens: 1 }, steps: 2,
  },
];

function renderChat() {
  return renderWithProviders(
    <Routes>
      <Route path="/c/:conversationId" element={<ChatRoute />} />
    </Routes>,
    { route: "/c/c1" },
  );
}

describe("ChatPage", () => {
  it("sends a question, shows tool steps and the cited answer", async () => {
    let answered = false;
    const { calls } = mockApi({
      "GET /conversations/c1": () => json(answered ? answeredDetail : emptyDetail),
      "GET /documents": () => json([]),
      "POST /conversations/c1/messages": () => {
        answered = true;
        return sse(streamed);
      },
      "GET /publications/6": () =>
        json({
          id: 6, eid: "e", title: "DNA repair in yeast", year: 2016, authors: ["Smith J.", "Lee K."], author_full_names: [],
          source_title: "Cell", publisher: null, document_type: "Article", doi: "10.1/x", link: null, cited_by: 120,
          open_access: null, affiliations: null, abstract: "Mechanisms of repair.", author_keywords: ["DNA repair"],
          index_keywords: [], cluster_label: "DNA Damage",
        }),
    });
    renderChat();

    const box = await screen.findByRole("textbox", { name: /message/i });
    await userEvent.type(box, "What is paper 6?{Enter}");

    expect(await screen.findByText("Get publication")).toBeInTheDocument();
    const chip = await screen.findByRole("button", { name: /citation 1/i });
    expect(screen.getByText(/It studies yeast/)).toBeInTheDocument();
    expect(calls.find((c) => c.method === "POST")?.body).toEqual({ content: "What is paper 6?" });

    await userEvent.click(chip);
    const drawer = await screen.findByRole("dialog", { name: /DNA repair in yeast/ });
    expect(within(drawer).getByText("Mechanisms of repair.")).toBeInTheDocument();
    expect(within(drawer).getByText(/Smith J\./)).toBeInTheDocument();
  });

  it("shows an approval card and resumes after approving", async () => {
    const approval: AgentEvent = {
      type: "approval_required", run_id: "r1", tool_call_id: "w1", name: "update_cluster_label",
      arguments: { publication_id: 11, new_label: "Astrophysics", reason: "galaxies" },
      context: { title: "Galaxy formation", current_label: null, proposed_label: "Astrophysics", closest_existing_labels: [] },
    };
    const { calls } = mockApi({
      "GET /conversations/c1": () => json(emptyDetail),
      "GET /documents": () => json([]),
      "POST /conversations/c1/messages": () =>
        sse([
          { type: "run_started", run_id: "r1", conversation_id: "c1", user_message_id: 1 },
          { type: "tool_call_started", tool_call_id: "w1", name: "update_cluster_label", arguments: approval.arguments, step: 1 },
          approval,
        ]),
      "POST /runs/r1/approvals": () =>
        sse([
          { type: "tool_call_finished", tool_call_id: "w1", name: "update_cluster_label", status: "ok", result_preview: "{}", duration_ms: 3, step: 1 },
          {
            type: "message_completed", run_id: "r1", message_id: 2, content: "Label updated.", citations: [],
            usage: { prompt_tokens: 1, completion_tokens: 1 }, steps: 2,
          },
        ]),
    });
    renderChat();
    await userEvent.type(await screen.findByRole("textbox", { name: /message/i }), "Relabel 11{Enter}");
    await userEvent.click(await screen.findByRole("button", { name: "Approve" }));
    await waitFor(() => expect(calls.some((c) => c.url === "/runs/r1/approvals")).toBe(true));
    expect(calls.find((c) => c.url === "/runs/r1/approvals")?.body).toEqual({ tool_call_id: "w1", approved: true });
    expect(await screen.findByText("Label updated.")).toBeInTheDocument();
  });

  it("shows a readable error when the request is refused", async () => {
    mockApi({
      "GET /conversations/c1": () => json(emptyDetail),
      "GET /documents": () => json([]),
      "POST /conversations/c1/messages": () =>
        json({ error: { code: "run_in_progress", message: "This conversation is still answering the previous message." } }, 409),
    });
    renderChat();
    await userEvent.type(await screen.findByRole("textbox", { name: /message/i }), "Hello{Enter}");
    expect(await screen.findByRole("alert")).toHaveTextContent("still answering");
  });

  it("renders a saved conversation with its tool history", async () => {
    mockApi({ "GET /conversations/c1": () => json(answeredDetail), "GET /documents": () => json([]) });
    renderChat();
    expect(await screen.findByText("What is paper 6?", { selector: "p" })).toBeInTheDocument();
    expect(screen.getByText(/It studies yeast/)).toBeInTheDocument();
    expect(screen.getByRole("button", { name: /used 1 tool/i })).toBeInTheDocument();
  });
});

describe("ChatPage retry", () => {
  it("offers Retry on a failed answer and resends the question", async () => {
    const failed: ConversationDetail = {
      ...emptyDetail,
      runs: [
        {
          id: "r1", status: "failed", steps: 1, model: "m", prompt_tokens: 0, completion_tokens: 0,
          error_code: "llm_unavailable", error: "The language model is temporarily unavailable.",
          started_at: "2026-10-03T10:00:00Z", finished_at: null, pending_approval: null,
        },
      ],
      messages: [{ id: 1, run_id: "r1", role: "user", content: "How many papers in 2021?", citations: [], created_at: "" }],
    };
    const { calls } = mockApi({
      "GET /conversations/c1": () => json(failed),
      "GET /documents": () => json([]),
      "POST /conversations/c1/messages": () =>
        sse([{ type: "run_started", run_id: "r2", conversation_id: "c1", user_message_id: 2 }]),
    });
    renderChat();
    expect(await screen.findByRole("alert")).toHaveTextContent("temporarily unavailable");
    await userEvent.click(screen.getByRole("button", { name: /retry/i }));
    await waitFor(() =>
      expect(calls.find((c) => c.method === "POST")?.body).toEqual({ content: "How many papers in 2021?" }),
    );
  });
});
