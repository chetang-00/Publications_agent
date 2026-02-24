import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Paperclip } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link } from "react-router";
import { api } from "../api/client";
import type { DocumentInfo } from "../api/types";

/** Choose which uploaded documents this conversation should search first. */
export function DocumentPicker({ conversationId, attached }: { conversationId: string; attached: DocumentInfo[] }) {
  const [open, setOpen] = useState(false);
  const panel = useRef<HTMLDivElement>(null);
  const queryClient = useQueryClient();
  const documents = useQuery({ queryKey: ["documents"], queryFn: api.listDocuments, enabled: open });
  const attachedIds = attached.map((d) => d.id);
  const save = useMutation({
    mutationFn: (ids: string[]) => api.attachDocuments(conversationId, ids),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ["conversation", conversationId] }),
  });

  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => {
      if (panel.current && !panel.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", close);
    return () => document.removeEventListener("mousedown", close);
  }, [open]);

  const ready = (documents.data ?? []).filter((d) => d.status === "ready");
  const toggle = (id: string) =>
    save.mutate(attachedIds.includes(id) ? attachedIds.filter((x) => x !== id) : [...attachedIds, id]);

  return (
    <div className="relative" ref={panel}>
      <button
        type="button"
        aria-expanded={open}
        onClick={() => setOpen((o) => !o)}
        className="flex items-center gap-1.5 rounded-lg border border-zinc-200 px-2.5 py-1.5 text-sm text-zinc-700 hover:bg-zinc-50 dark:border-zinc-800 dark:text-zinc-300 dark:hover:bg-zinc-900"
      >
        <Paperclip aria-hidden className="size-4" />
        Documents{attached.length > 0 && ` (${attached.length})`}
      </button>
      {open && (
        <div className="absolute right-0 z-30 mt-2 w-80 rounded-xl border border-zinc-200 bg-white p-2 shadow-lg dark:border-zinc-800 dark:bg-zinc-900">
          <p className="px-2 py-1 text-xs text-zinc-500">The assistant searches attached documents first.</p>
          {documents.isPending && <p className="px-2 py-2 text-sm text-zinc-500">Loading…</p>}
          {!documents.isPending && ready.length === 0 && (
            <p className="px-2 py-2 text-sm text-zinc-500">No processed documents yet.</p>
          )}
          <ul className="max-h-72 overflow-y-auto">
            {ready.map((doc) => (
              <li key={doc.id}>
                <label className="flex cursor-pointer items-center gap-2 rounded-lg px-2 py-1.5 text-sm hover:bg-zinc-50 dark:hover:bg-zinc-800">
                  <input
                    type="checkbox"
                    checked={attachedIds.includes(doc.id)}
                    disabled={save.isPending}
                    onChange={() => toggle(doc.id)}
                    className="accent-brand-600"
                  />
                  <span className="truncate">{doc.filename}</span>
                </label>
              </li>
            ))}
          </ul>
          <Link
            to="/documents"
            className="mt-1 block rounded-lg px-2 py-1.5 text-sm font-medium text-brand-600 hover:bg-brand-50 dark:text-brand-300 dark:hover:bg-zinc-800"
          >
            Upload or manage documents →
          </Link>
        </div>
      )}
    </div>
  );
}
