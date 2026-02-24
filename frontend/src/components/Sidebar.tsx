import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { BookOpenText, FileStack, MessageSquare, Plus, Trash2 } from "lucide-react";
import { useState } from "react";
import { NavLink, useMatch, useNavigate } from "react-router";
import { api } from "../api/client";
import { relativeTime } from "../lib/format";

const navItem = ({ isActive }: { isActive: boolean }) =>
  `flex items-center gap-2 rounded-lg px-3 py-2 text-sm font-medium ${
    isActive
      ? "bg-brand-50 text-brand-700 dark:bg-brand-700/30 dark:text-brand-200"
      : "text-zinc-700 hover:bg-zinc-100 dark:text-zinc-300 dark:hover:bg-zinc-800"
  }`;

export function Sidebar() {
  const queryClient = useQueryClient();
  const navigate = useNavigate();
  const active = useMatch("/c/:conversationId")?.params.conversationId;
  const [confirming, setConfirming] = useState<string | null>(null);
  const conversations = useQuery({ queryKey: ["conversations"], queryFn: api.listConversations });
  const remove = useMutation({
    mutationFn: api.deleteConversation,
    onSuccess: (_, id) => {
      setConfirming(null);
      void queryClient.invalidateQueries({ queryKey: ["conversations"] });
      if (id === active) navigate("/");
    },
  });

  return (
    <aside className="flex w-full shrink-0 flex-col border-b border-zinc-200 bg-white md:h-full md:w-72 md:border-r md:border-b-0 dark:border-zinc-800 dark:bg-zinc-900">
      <div className="flex items-center gap-2.5 px-4 py-4">
        <div className="flex size-8 items-center justify-center rounded-lg bg-brand-600 text-white">
          <BookOpenText aria-hidden className="size-4.5" />
        </div>
        <div className="leading-tight">
          <p className="text-sm font-semibold">Publications Agent</p>
          <p className="text-[11px] text-zinc-500">Research Q&amp;A with tools</p>
        </div>
      </div>

      <nav className="flex gap-1 px-3 md:flex-col">
        <NavLink to="/" end className={navItem}>
          <Plus aria-hidden className="size-4" /> New chat
        </NavLink>
        <NavLink to="/documents" className={navItem}>
          <FileStack aria-hidden className="size-4" /> Documents
        </NavLink>
      </nav>

      <div className="mt-4 hidden min-h-0 flex-1 flex-col md:flex">
        <p className="px-5 pb-1 text-[11px] font-semibold tracking-wide text-zinc-400 uppercase">Conversations</p>
        <ul className="flex-1 space-y-0.5 overflow-y-auto px-3 pb-4">
          {conversations.data?.length === 0 && <li className="px-3 py-2 text-xs text-zinc-400">No conversations yet.</li>}
          {conversations.data?.map((c) => (
            <li key={c.id} className="group relative">
              <NavLink to={`/c/${c.id}`} className={navItem}>
                <MessageSquare aria-hidden className="size-4 shrink-0 opacity-60" />
                <span className="min-w-0 flex-1">
                  <span className="block truncate">{c.title}</span>
                  <span className="block text-[11px] font-normal text-zinc-400">{relativeTime(c.updated_at)}</span>
                </span>
              </NavLink>
              {confirming === c.id ? (
                <div className="absolute inset-y-0 right-1 flex items-center gap-1 bg-white pl-2 dark:bg-zinc-900">
                  <button
                    type="button"
                    aria-label={`Confirm delete ${c.title}`}
                    onClick={() => remove.mutate(c.id)}
                    className="rounded-md bg-rose-600 px-2 py-1 text-[11px] font-semibold text-white hover:bg-rose-700"
                  >
                    Delete
                  </button>
                  <button
                    type="button"
                    onClick={() => setConfirming(null)}
                    className="rounded-md px-2 py-1 text-[11px] text-zinc-500 hover:bg-zinc-100 dark:hover:bg-zinc-800"
                  >
                    Cancel
                  </button>
                </div>
              ) : (
                <button
                  type="button"
                  aria-label={`Delete ${c.title}`}
                  onClick={() => setConfirming(c.id)}
                  className="absolute top-1/2 right-2 hidden -translate-y-1/2 rounded-md p-1 text-zinc-400 group-hover:block hover:bg-zinc-200 hover:text-rose-600 focus:block dark:hover:bg-zinc-700"
                >
                  <Trash2 className="size-3.5" />
                </button>
              )}
            </li>
          ))}
        </ul>
      </div>
    </aside>
  );
}
