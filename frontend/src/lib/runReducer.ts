import type {
  AgentEvent,
  ApprovalRequiredEvent,
  MessageCompletedEvent,
  ToolCallStatus,
} from "../api/types";

export type StepStatus = "running" | "awaiting_approval" | ToolCallStatus;

export interface StepView {
  toolCallId: string;
  name: string;
  arguments: Record<string, unknown> | null;
  status: StepStatus;
  step: number;
  preview?: string;
  durationMs?: number | null;
}

export interface RunView {
  runId: string | null;
  status: "idle" | "streaming" | "awaiting_approval" | "completed" | "failed";
  draft: string;
  steps: StepView[];
  approval: ApprovalRequiredEvent | null;
  final: MessageCompletedEvent | null;
  error: { code: string; message: string } | null;
}

export const initialRun: RunView = {
  runId: null,
  status: "idle",
  draft: "",
  steps: [],
  approval: null,
  final: null,
  error: null,
};

export type RunAction = AgentEvent | { type: "resume"; runId?: string } | { type: "reset" };

/** Live state of one agent run, driven by the SSE events of that run. */
export function runReducer(state: RunView, action: RunAction): RunView {
  switch (action.type) {
    case "reset":
      return initialRun;
    case "resume":
      return { ...state, runId: action.runId ?? state.runId, status: "streaming", approval: null, error: null };
    case "run_started":
      return { ...initialRun, runId: action.run_id, status: "streaming" };
    case "token":
      return { ...state, draft: state.draft + action.text };
    case "tool_call_started":
      return {
        ...state,
        draft: "", // text before a tool call is the model thinking aloud, not the answer
        steps: [
          ...state.steps,
          {
            toolCallId: action.tool_call_id,
            name: action.name,
            arguments: action.arguments,
            status: "running",
            step: action.step,
          },
        ],
      };
    case "tool_call_finished": {
      const update = { status: action.status, preview: action.result_preview, durationMs: action.duration_ms };
      const exists = state.steps.some((s) => s.toolCallId === action.tool_call_id && s.step === action.step);
      return {
        ...state,
        steps: exists
          ? state.steps.map((s) => (s.toolCallId === action.tool_call_id && s.step === action.step ? { ...s, ...update } : s))
          : [...state.steps, { toolCallId: action.tool_call_id, name: action.name, arguments: null, step: action.step, ...update }],
      };
    }
    case "approval_required":
      return {
        ...state,
        status: "awaiting_approval",
        approval: action,
        steps: state.steps.map((s) =>
          s.toolCallId === action.tool_call_id && s.status === "running" ? { ...s, status: "awaiting_approval" } : s,
        ),
      };
    case "message_completed":
      return { ...state, status: "completed", draft: "", final: action };
    case "error":
      return { ...state, status: "failed", error: { code: action.code, message: action.message } };
    default:
      return state;
  }
}
