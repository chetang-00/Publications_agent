import { ChevronRight, Workflow } from "lucide-react";
import { useState } from "react";
import { formatDuration } from "../lib/format";
import type { StepView } from "../lib/runReducer";
import { ToolStepCard } from "./ToolStepCard";

/** The agent's tool calls for one answer. Past answers start collapsed; live ones are open. */
export function StepTimeline({ steps, collapsible }: { steps: StepView[]; collapsible: boolean }) {
  const [open, setOpen] = useState(!collapsible);
  if (steps.length === 0) return null;

  const total = steps.reduce((sum, s) => sum + (s.durationMs ?? 0), 0);
  const summary = `Used ${steps.length} tool${steps.length === 1 ? "" : "s"}${total ? ` · ${formatDuration(total)}` : ""}`;

  if (collapsible && !open) {
    return (
      <button
        type="button"
        onClick={() => setOpen(true)}
        className="flex items-center gap-1.5 rounded-md px-1.5 py-1 text-xs font-medium text-zinc-500 hover:bg-zinc-100 hover:text-zinc-800 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
      >
        <Workflow aria-hidden className="size-3.5" />
        {summary}
        <ChevronRight aria-hidden className="size-3.5" />
      </button>
    );
  }

  return (
    <div className="space-y-1.5">
      {collapsible && (
        <button
          type="button"
          onClick={() => setOpen(false)}
          className="flex items-center gap-1.5 px-1.5 text-xs font-medium text-zinc-500 hover:text-zinc-800 dark:hover:text-zinc-200"
        >
          <Workflow aria-hidden className="size-3.5" />
          Hide tool steps
        </button>
      )}
      <ol className="space-y-1.5">
        {steps.map((step) => (
          <li key={`${step.step}:${step.toolCallId}`}>
            <ToolStepCard step={step} />
          </li>
        ))}
      </ol>
    </div>
  );
}
