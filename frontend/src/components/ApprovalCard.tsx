import { ArrowRight, ShieldCheck } from "lucide-react";
import { useId, useState } from "react";
import type { PendingApproval } from "../api/types";
import { humanizeToolName } from "../lib/format";

export interface Decision {
  approved: boolean;
  note?: string;
}

const text = (value: unknown): string | null => (typeof value === "string" && value.trim() ? value : null);

/** Human-in-the-loop gate: the agent's write action waits here until the user decides. */
export function ApprovalCard({
  approval,
  busy,
  onDecide,
}: {
  approval: PendingApproval;
  busy: boolean;
  onDecide: (decision: Decision) => void;
}) {
  const [note, setNote] = useState("");
  const noteId = useId();
  const { context, arguments: args } = approval;
  const isLabelChange = approval.name === "update_cluster_label";
  const title = text(context.title);
  const current = text(context.current_label);
  const proposed = text(context.proposed_label) ?? text(args.new_label);
  const reason = text(context.reason) ?? text(args.reason);
  const closest = (Array.isArray(context.closest_existing_labels) ? context.closest_existing_labels : []).filter(
    (label): label is string => typeof label === "string" && label !== proposed,
  );

  return (
    <section
      aria-label="Approval required"
      className="rounded-xl border border-amber-300 bg-amber-50/70 p-4 shadow-sm dark:border-amber-700/60 dark:bg-amber-950/30"
    >
      <header className="flex items-center gap-2 text-amber-900 dark:text-amber-200">
        <ShieldCheck aria-hidden className="size-5" />
        <h3 className="font-semibold">Approve this change?</h3>
      </header>
      <p className="mt-1 text-sm text-zinc-600 dark:text-zinc-400">
        {isLabelChange
          ? "The assistant wants to change a publication's research cluster label. Nothing changes until you approve."
          : `The assistant wants to run "${humanizeToolName(approval.name)}", which modifies data.`}
      </p>

      {isLabelChange ? (
        <div className="mt-3 space-y-2 rounded-lg bg-white/80 p-3 text-sm dark:bg-zinc-900/60">
          {title && <p className="font-medium">{title}</p>}
          <div className="flex flex-wrap items-center gap-2">
            <span className="rounded-md bg-zinc-100 px-2 py-0.5 text-zinc-600 dark:bg-zinc-800 dark:text-zinc-400">
              {current ?? "Unlabelled"}
            </span>
            <ArrowRight aria-hidden className="size-4 text-zinc-400" />
            <span className="rounded-md bg-brand-50 px-2 py-0.5 font-semibold text-brand-700 dark:bg-brand-700/30 dark:text-brand-200">
              {proposed}
            </span>
          </div>
          {context.label_exists === false && (
            <p className="text-xs text-amber-800 dark:text-amber-300">
              This creates a new label.
              {closest.length > 0 && (
                <>
                  {" "}
                  Similar existing labels:{" "}
                  {closest.map((label) => (
                    <span key={label} className="mr-1 inline-block rounded bg-amber-100 px-1.5 py-0.5 font-medium dark:bg-amber-900/50">
                      {label}
                    </span>
                  ))}
                </>
              )}
            </p>
          )}
          {reason && (
            <p className="text-sm">
              <span className="text-zinc-500">Reason: </span>
              <span>{reason}</span>
            </p>
          )}
        </div>
      ) : (
        <pre className="mt-3 overflow-x-auto rounded-lg bg-white/80 p-3 font-mono text-xs dark:bg-zinc-900/60">
          {JSON.stringify(args, null, 2)}
        </pre>
      )}

      <label htmlFor={noteId} className="mt-3 block text-xs font-medium text-zinc-600 dark:text-zinc-400">
        Note (optional, sent to the assistant if you reject)
      </label>
      <textarea
        id={noteId}
        value={note}
        onChange={(e) => setNote(e.target.value)}
        rows={2}
        maxLength={500}
        className="mt-1 w-full resize-none rounded-lg border border-zinc-300 bg-white px-3 py-2 text-sm outline-none focus:border-brand-500 focus:ring-2 focus:ring-brand-200 dark:border-zinc-700 dark:bg-zinc-900 dark:focus:ring-brand-700/40"
      />
      <div className="mt-3 flex gap-2">
        <button
          type="button"
          disabled={busy}
          onClick={() => onDecide({ approved: true })}
          className="rounded-lg bg-brand-600 px-4 py-2 text-sm font-semibold text-white shadow-sm hover:bg-brand-700 disabled:cursor-not-allowed disabled:opacity-50"
        >
          Approve
        </button>
        <button
          type="button"
          disabled={busy}
          onClick={() => onDecide(note.trim() ? { approved: false, note: note.trim() } : { approved: false })}
          className="rounded-lg border border-zinc-300 bg-white px-4 py-2 text-sm font-semibold text-zinc-700 hover:bg-zinc-50 disabled:cursor-not-allowed disabled:opacity-50 dark:border-zinc-700 dark:bg-zinc-900 dark:text-zinc-200 dark:hover:bg-zinc-800"
        >
          Reject
        </button>
      </div>
    </section>
  );
}
