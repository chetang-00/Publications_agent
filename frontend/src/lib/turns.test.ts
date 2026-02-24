import { describe, expect, it } from "vitest";
import type { ConversationDetail, Run } from "../api/types";
import { buildTurns, mergeSteps } from "./turns";
import type { StepView } from "./runReducer";

const run = (id: string, extra: Partial<Run> = {}): Run => ({
  id,
  status: "completed",
  steps: 1,
  model: "m",
  prompt_tokens: 0,
  completion_tokens: 0,
  error_code: null,
  error: null,
  started_at: "2026-10-03T10:00:00Z",
  finished_at: null,
  pending_approval: null,
  ...extra,
});

const detail: ConversationDetail = {
  conversation: { id: "c1", title: "T", created_at: "", updated_at: "" },
  documents: [],
  runs: [run("r1"), run("r2", { status: "failed", error_code: "llm_unavailable", error: "down" })],
  messages: [
    { id: 1, run_id: "r1", role: "user", content: "q1", citations: [], created_at: "" },
    { id: 3, run_id: "r1", role: "assistant", content: "a1 [pub:6]", citations: [{ kind: "publication", id: "6", marker: "pub:6" }], created_at: "" },
    { id: 4, run_id: "r2", role: "user", content: "q2", citations: [], created_at: "" },
  ],
  tool_calls: [
    { run_id: "r1", call_id: "c1", step: 1, name: "get_publication", arguments: { publication_id: 6 }, status: "ok", result_preview: "{}", error: null, duration_ms: 5 },
  ],
};

describe("buildTurns", () => {
  it("groups messages and tool calls by run", () => {
    const turns = buildTurns(detail);
    expect(turns).toHaveLength(2);
    expect(turns[0]).toMatchObject({ runId: "r1", userText: "q1", status: "completed" });
    expect(turns[0]?.answer?.content).toBe("a1 [pub:6]");
    expect(turns[0]?.steps).toEqual([
      { toolCallId: "c1", name: "get_publication", arguments: { publication_id: 6 }, status: "ok", step: 1, preview: "{}", durationMs: 5 },
    ]);
    expect(turns[1]).toMatchObject({ runId: "r2", userText: "q2", answer: null, error: { code: "llm_unavailable", message: "down" } });
  });
});

describe("mergeSteps", () => {
  const step = (id: string, status: StepView["status"]): StepView => ({ toolCallId: id, name: "t", arguments: null, status, step: 1 });

  it("lets live steps override history and appends new ones", () => {
    const merged = mergeSteps([step("a", "ok"), step("w", "awaiting_approval")], [step("w", "rejected"), step("b", "running")]);
    expect(merged.map((s) => [s.toolCallId, s.status])).toEqual([
      ["a", "ok"],
      ["w", "rejected"],
      ["b", "running"],
    ]);
  });
});
