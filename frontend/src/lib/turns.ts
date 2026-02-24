import type {
  ApprovalRequiredEvent,
  Citation,
  ConversationDetail,
  PendingApproval,
  RunStatus,
  ToolCall,
} from "../api/types";
import type { TurnView } from "../components/Turn";
import type { RunView, StepView } from "./runReducer";

/** One question-and-answer exchange: a user message, the agent's tool steps, and its answer. */
export interface Turn {
  runId: string;
  userText: string | null;
  steps: StepView[];
  answer: { content: string; citations: Citation[] } | null;
  status: RunStatus;
  error: { code: string; message: string } | null;
  approval: PendingApproval | null;
}

export function toolCallToStep(call: ToolCall): StepView {
  return {
    toolCallId: call.call_id,
    name: call.name,
    arguments: call.arguments,
    status: call.status,
    step: call.step,
    preview: call.result_preview ?? call.error ?? undefined,
    durationMs: call.duration_ms,
  };
}

export function buildTurns(detail: ConversationDetail): Turn[] {
  return detail.runs.map((run) => {
    const user = detail.messages.find((m) => m.run_id === run.id && m.role === "user");
    const answer = detail.messages.find((m) => m.run_id === run.id && m.role === "assistant");
    return {
      runId: run.id,
      userText: user?.content ?? null,
      steps: detail.tool_calls.filter((t) => t.run_id === run.id).map(toolCallToStep),
      answer: answer ? { content: answer.content, citations: answer.citations } : null,
      status: run.status,
      error:
        run.status === "failed" || run.status === "cancelled"
          ? { code: run.error_code ?? "error", message: run.error ?? "The answer could not be completed." }
          : null,
      approval: run.pending_approval,
    };
  });
}

/** Steps saved in history, updated by live events for the same calls, plus any new live steps. */
export function mergeSteps(history: StepView[], live: StepView[]): StepView[] {
  const key = (s: StepView) => `${s.step}:${s.toolCallId}`;
  const liveByKey = new Map(live.map((s) => [key(s), s]));
  const merged = history.map((s) => liveByKey.get(key(s)) ?? s);
  const seen = new Set(history.map(key));
  return [...merged, ...live.filter((s) => !seen.has(key(s)))];
}

// ── merging the saved history with the run that is streaming right now ──

function toPending(event: ApprovalRequiredEvent): PendingApproval {
  const { run_id, tool_call_id, name, arguments: args, context } = event;
  return { run_id, tool_call_id, name, arguments: args, context };
}

/**
 * Saved turns, with the live run overlaid on its saved turn (approval resumes) or appended
 * (a new question). Keys stay stable when the live run lands in history, so nothing re-mounts.
 */
export function composeTurns(
  history: Turn[],
  live: RunView,
  pendingText: string | null,
  seenRunIds: ReadonlySet<string> = new Set(),
): TurnView[] {
  const liveActive = live.status !== "idle";
  const finalAnswer = live.final ? { content: live.final.content, citations: live.final.citations } : null;
  const views: TurnView[] = history.map((t) => {
    if (!(liveActive && live.runId === t.runId)) {
      return {
        key: t.runId,
        runId: t.runId,
        userText: t.userText,
        steps: t.steps,
        draft: "",
        answer: t.answer,
        approval: t.approval,
        error: t.status === "cancelled" ? null : t.error,
        cancelled: t.status === "cancelled",
        live: false,
        streaming: false,
        expandSteps: seenRunIds.has(t.runId),
      };
    }
    return {
      key: t.runId,
      runId: t.runId,
      userText: t.userText,
      steps: mergeSteps(t.steps, live.steps),
      draft: live.draft,
      answer: t.answer ?? finalAnswer,
      approval: live.status === "awaiting_approval" && live.approval ? toPending(live.approval) : null,
      error: live.error,
      cancelled: false,
      live: true,
      streaming: live.status === "streaming",
      expandSteps: true,
    };
  });

  const inHistory = live.runId !== null && history.some((t) => t.runId === live.runId);
  if (!inHistory && (liveActive || pendingText)) {
    views.push({
      key: live.runId ?? "pending",
      runId: live.runId,
      userText: pendingText,
      steps: live.steps,
      draft: live.draft,
      answer: finalAnswer,
      approval: live.status === "awaiting_approval" && live.approval ? toPending(live.approval) : null,
      error: live.error,
      cancelled: false,
      live: true,
      streaming: live.status === "streaming" || (!liveActive && pendingText !== null),
      expandSteps: true,
    });
  }
  return views;
}
