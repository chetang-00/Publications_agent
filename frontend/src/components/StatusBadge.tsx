import { Loader2 } from "lucide-react";
import type { ReactNode } from "react";

export type Tone = "neutral" | "info" | "success" | "warning" | "danger" | "brand";

const TONES: Record<Tone, string> = {
  neutral: "bg-zinc-100 text-zinc-700 dark:bg-zinc-800 dark:text-zinc-300",
  info: "bg-sky-50 text-sky-700 dark:bg-sky-950 dark:text-sky-300",
  success: "bg-emerald-50 text-emerald-700 dark:bg-emerald-950 dark:text-emerald-300",
  warning: "bg-amber-50 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
  danger: "bg-rose-50 text-rose-700 dark:bg-rose-950 dark:text-rose-300",
  brand: "bg-brand-50 text-brand-700 dark:bg-brand-700/30 dark:text-brand-200",
};

export function StatusBadge({
  tone,
  spinning = false,
  role,
  children,
}: {
  tone: Tone;
  spinning?: boolean;
  role?: "status";
  children: ReactNode;
}) {
  return (
    <span
      role={role}
      className={`inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[11px] font-medium whitespace-nowrap ${TONES[tone]}`}
    >
      {spinning && <Loader2 aria-hidden className="size-3 animate-spin" />}
      {children}
    </span>
  );
}
