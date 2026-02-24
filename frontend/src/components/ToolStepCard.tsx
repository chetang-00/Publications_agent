import {
  BarChart3,
  BookOpen,
  ChevronDown,
  Database,
  FileSearch,
  Files,
  Filter,
  type LucideIcon,
  Search,
  Tag,
  UserSearch,
  Wrench,
} from "lucide-react";
import { useState } from "react";
import { formatDuration, humanizeToolName } from "../lib/format";
import type { StepStatus, StepView } from "../lib/runReducer";
import { StatusBadge, type Tone } from "./StatusBadge";

const ICONS: Record<string, LucideIcon> = {
  search_publications: Search,
  filter_publications: Filter,
  publication_stats: BarChart3,
  get_publication: BookOpen,
  resolve_author: UserSearch,
  run_readonly_sql: Database,
  list_documents: Files,
  search_documents: FileSearch,
  update_cluster_label: Tag,
};

const STATUS: Record<StepStatus, { label: string; tone: Tone }> = {
  running: { label: "Running", tone: "info" },
  ok: { label: "Done", tone: "success" },
  invalid_arguments: { label: "Invalid arguments", tone: "warning" },
  error: { label: "Error", tone: "danger" },
  timeout: { label: "Timed out", tone: "danger" },
  rejected: { label: "Rejected", tone: "neutral" },
  unknown_tool: { label: "Unknown tool", tone: "danger" },
  awaiting_approval: { label: "Needs approval", tone: "brand" },
};

function formatValue(value: unknown): string {
  if (Array.isArray(value)) return value.map(formatValue).join(", ");
  if (value !== null && typeof value === "object") return JSON.stringify(value);
  const text = String(value);
  return text.length > 80 ? `${text.slice(0, 79)}…` : text;
}

function pretty(preview: string): string {
  try {
    return JSON.stringify(JSON.parse(preview), null, 2);
  } catch {
    return preview;
  }
}

export function ToolStepCard({ step, defaultOpen = false }: { step: StepView; defaultOpen?: boolean }) {
  const [open, setOpen] = useState(defaultOpen);
  const Icon = ICONS[step.name] ?? Wrench;
  const status = STATUS[step.status];
  const args = Object.entries(step.arguments ?? {}).filter(([, v]) => v !== null && v !== undefined && v !== "");

  return (
    <div className="rounded-lg border border-zinc-200 bg-white text-sm dark:border-zinc-800 dark:bg-zinc-900">
      <div className="flex items-start gap-2.5 px-3 py-2">
        <Icon aria-hidden className="mt-0.5 size-4 shrink-0 text-brand-600 dark:text-brand-300" />
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-center gap-2">
            <span className="font-medium">{humanizeToolName(step.name)}</span>
            <StatusBadge role="status" tone={status.tone} spinning={step.status === "running"}>
              {status.label}
            </StatusBadge>
            {step.durationMs != null && <span className="text-xs text-zinc-500">{formatDuration(step.durationMs)}</span>}
          </div>
          {args.length > 0 && (
            <div className="mt-1 flex flex-wrap gap-1">
              {args.map(([key, value]) => (
                <code
                  key={key}
                  className="max-w-full truncate rounded bg-zinc-100 px-1.5 py-0.5 font-mono text-[11px] text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300"
                >
                  {key}: {formatValue(value)}
                </code>
              ))}
            </div>
          )}
        </div>
        {step.preview && (
          <button
            type="button"
            aria-expanded={open}
            onClick={() => setOpen((o) => !o)}
            className="flex shrink-0 items-center gap-1 rounded px-1.5 py-0.5 text-xs text-zinc-500 hover:bg-zinc-100 hover:text-zinc-800 dark:hover:bg-zinc-800 dark:hover:text-zinc-200"
          >
            Details
            <ChevronDown aria-hidden className={`size-3.5 transition-transform ${open ? "rotate-180" : ""}`} />
          </button>
        )}
      </div>
      {open && step.preview && (
        <pre className="max-h-64 overflow-auto border-t border-zinc-100 bg-zinc-50 px-3 py-2 font-mono text-[11px] leading-relaxed whitespace-pre-wrap text-zinc-700 dark:border-zinc-800 dark:bg-zinc-950 dark:text-zinc-300">
          {pretty(step.preview)}
        </pre>
      )}
    </div>
  );
}
