import { describe, expect, it } from "vitest";
import type { AgentEvent } from "../api/types";
import { initialRun, runReducer, type RunView } from "./runReducer";

function play(events: (AgentEvent | { type: "resume" })[], start: RunView = initialRun): RunView {
  return events.reduce(runReducer, start);
}

const started: AgentEvent = { type: "run_started", run_id: "r1", conversation_id: "c1", user_message_id: 1 };

describe("runReducer", () => {
  it("starts streaming and accumulates tokens", () => {
    const state = play([started, { type: "token", text: "Hel", step: 1 }, { type: "token", text: "lo", step: 1 }]);
    expect(state.status).toBe("streaming");
    expect(state.runId).toBe("r1");
    expect(state.draft).toBe("Hello");
  });

  it("tracks the tool step lifecycle and clears preamble text", () => {
    const state = play([
      started,
      { type: "token", text: "Let me check.", step: 1 },
      { type: "tool_call_started", tool_call_id: "c1", name: "get_publication", arguments: { publication_id: 6 }, step: 1 },
    ]);
    expect(state.draft).toBe("");
    expect(state.steps).toEqual([
      { toolCallId: "c1", name: "get_publication", arguments: { publication_id: 6 }, status: "running", step: 1 },
    ]);
    const finished = runReducer(state, {
      type: "tool_call_finished",
      tool_call_id: "c1",
      name: "get_publication",
      status: "ok",
      result_preview: "{...}",
      duration_ms: 12,
      step: 1,
    });
    expect(finished.steps[0]).toMatchObject({ status: "ok", preview: "{...}", durationMs: 12 });
  });

  it("adds a finished step it never saw start (resumed after reload)", () => {
    const state = play([
      { type: "tool_call_finished", tool_call_id: "w1", name: "update_cluster_label", status: "rejected", result_preview: "Rejected", duration_ms: null, step: 1 },
    ]);
    expect(state.steps).toHaveLength(1);
    expect(state.steps[0]?.status).toBe("rejected");
  });

  it("pauses for approval and resumes", () => {
    const approval: AgentEvent = {
      type: "approval_required",
      run_id: "r1",
      tool_call_id: "w1",
      name: "update_cluster_label",
      arguments: { publication_id: 11, new_label: "Astro", reason: "r" },
      context: { current_label: null },
    };
    const paused = play([
      started,
      { type: "tool_call_started", tool_call_id: "w1", name: "update_cluster_label", arguments: {}, step: 1 },
      approval,
    ]);
    expect(paused.status).toBe("awaiting_approval");
    expect(paused.approval).toEqual(approval);
    expect(paused.steps[0]?.status).toBe("awaiting_approval");
    const resumed = runReducer(paused, { type: "resume" });
    expect(resumed.status).toBe("streaming");
    expect(resumed.approval).toBeNull();
  });

  it("completes with the authoritative final message", () => {
    const state = play([
      started,
      { type: "token", text: "draft", step: 1 },
      {
        type: "message_completed",
        run_id: "r1",
        message_id: 9,
        content: "Final [pub:6].",
        citations: [{ kind: "publication", id: "6", marker: "pub:6", title: "T" }],
        usage: { prompt_tokens: 1, completion_tokens: 2 },
        steps: 2,
      },
    ]);
    expect(state.status).toBe("completed");
    expect(state.draft).toBe("");
    expect(state.final?.content).toBe("Final [pub:6].");
  });

  it("records errors", () => {
    const state = play([started, { type: "error", code: "llm_unavailable", message: "Try later", run_id: "r1" }]);
    expect(state.status).toBe("failed");
    expect(state.error).toEqual({ code: "llm_unavailable", message: "Try later" });
  });
});

describe("resume after a page reload", () => {
  it("adopts the run id when resuming a run the page never streamed", () => {
    const state = runReducer(initialRun, { type: "resume", runId: "r9" });
    expect(state).toMatchObject({ runId: "r9", status: "streaming", approval: null });
  });
});
