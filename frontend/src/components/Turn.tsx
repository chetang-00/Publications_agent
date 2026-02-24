import { AlertTriangle, Loader2, RotateCcw, Sparkles } from "lucide-react";
import type { Citation, PendingApproval } from "../api/types";
import type { StepView } from "../lib/runReducer";
import { ApprovalCard, type Decision } from "./ApprovalCard";
import { Markdown } from "./Markdown";
import { StepTimeline } from "./StepTimeline";

export interface TurnView {
  key: string;
  runId: string | null;
  userText: string | null;
  steps: StepView[];
  draft: string;
  answer: { content: string; citations: Citation[] } | null;
  approval: PendingApproval | null;
  error: { code: string; message: string } | null;
  cancelled: boolean;
  live: boolean;
  streaming: boolean;
  expandSteps: boolean;
}

export function Turn({
  turn,
  deciding,
  onDecide,
  onCitation,
  onRetry,
}: {
  turn: TurnView;
  deciding: boolean;
  onDecide: (approval: PendingApproval, decision: Decision) => void;
  onCitation: (citation: Citation) => void;
  onRetry?: () => void;
}) {
  const thinking = turn.streaming && !turn.draft && !turn.answer && !turn.approval && !turn.error;
  return (
    <div className="space-y-3">
      {turn.userText && (
        <div className="flex justify-end">
          <div className="max-w-[85%] rounded-2xl rounded-br-md bg-brand-600 px-4 py-2.5 text-white shadow-sm">
            <p className="text-[15px] leading-relaxed whitespace-pre-wrap">{turn.userText}</p>
          </div>
        </div>
      )}

      <div className="flex gap-3">
        <div className="mt-0.5 flex size-7 shrink-0 items-center justify-center rounded-full bg-brand-50 text-brand-600 dark:bg-brand-700/30 dark:text-brand-200">
          <Sparkles aria-hidden className="size-4" />
        </div>
        <div className="min-w-0 flex-1 space-y-3">
          <StepTimeline steps={turn.steps} collapsible={!turn.expandSteps} />

          {thinking && (
            <p className="flex items-center gap-2 text-sm text-zinc-500">
              <Loader2 aria-hidden className="size-4 animate-spin" />
              {turn.steps.length ? "Working on it…" : "Thinking…"}
            </p>
          )}

          {turn.approval && (
            <ApprovalCard
              approval={turn.approval}
              busy={deciding}
              onDecide={(decision) => turn.approval && onDecide(turn.approval, decision)}
            />
          )}

          {turn.answer ? (
            <div className="text-[15px]">
              <Markdown content={turn.answer.content} citations={turn.answer.citations} onCitation={onCitation} />
            </div>
          ) : (
            turn.draft && (
              <div className="typing-cursor text-[15px] leading-relaxed whitespace-pre-wrap text-zinc-700 dark:text-zinc-300">
                {turn.draft}
              </div>
            )
          )}

          {turn.cancelled && (
            <p className="text-sm text-zinc-500 italic">
              The proposed change was cancelled because a new message was sent.
            </p>
          )}

          {turn.error && !turn.cancelled && (
            <div
              role="alert"
              className="flex items-start gap-2 rounded-lg border border-rose-200 bg-rose-50 px-3 py-2 text-sm text-rose-800 dark:border-rose-900 dark:bg-rose-950/40 dark:text-rose-200"
            >
              <AlertTriangle aria-hidden className="mt-0.5 size-4 shrink-0" />
              <span className="flex-1">{turn.error.message}</span>
              {onRetry && (
                <button
                  type="button"
                  onClick={onRetry}
                  className="flex items-center gap-1 font-medium underline-offset-2 hover:underline"
                >
                  <RotateCcw aria-hidden className="size-3.5" /> Retry
                </button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
