import { useQuery, useQueryClient } from "@tanstack/react-query";
import { BookOpen, FileSearch, Loader2, MessageCircleQuestion } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { Link, useLocation, useNavigate, useParams } from "react-router";
import { ApiError, api } from "../api/client";
import type { Citation, PendingApproval } from "../api/types";
import type { Decision } from "../components/ApprovalCard";
import { CitationDrawer } from "../components/CitationDrawer";
import { Composer } from "../components/Composer";
import { DocumentPicker } from "../components/DocumentPicker";
import { Turn } from "../components/Turn";
import { useAgentRun } from "../hooks/useAgentRun";
import { buildTurns, composeTurns } from "../lib/turns";

const EXAMPLES = [
  "How many papers were published each year since 2018?",
  "Which authors publish most on CRISPR? Show their top papers.",
  "What does Rajesh Ranganath work on, and how has it changed? Cite papers.",
  "Summarize the methods section of my uploaded document.",
];

export function ChatRoute() {
  const { conversationId } = useParams();
  if (!conversationId) return null;
  return <ChatPage key={conversationId} conversationId={conversationId} />;
}

function ChatPage({ conversationId }: { conversationId: string }) {
  const location = useLocation();
  const navigate = useNavigate();
  const detail = useQuery({
    queryKey: ["conversation", conversationId],
    queryFn: () => api.getConversation(conversationId),
  });
  const run = useAgentRun(conversationId);
  const [citation, setCitation] = useState<Citation | null>(null);
  const [deciding, setDeciding] = useState(false);
  const bottom = useRef<HTMLDivElement>(null);

  // A question typed on the "New chat" page arrives here as navigation state.
  const initialMessage = (location.state as { initialMessage?: string } | null)?.initialMessage;
  const sentInitial = useRef(false);
  useEffect(() => {
    if (initialMessage && !sentInitial.current) {
      sentInitial.current = true;
      navigate(".", { replace: true, state: null });
      void run.send(initialMessage);
    }
  }, [initialMessage, navigate, run]);

  const history = detail.data ? buildTurns(detail.data) : [];
  const turns = composeTurns(history, run.state, run.pendingText, run.seenRunIds);
  const lastTurn = turns.at(-1);

  useEffect(() => {
    bottom.current?.scrollIntoView?.({ block: "end" });
  }, [turns.length, run.state.draft, run.state.steps.length, run.state.status]);

  async function decide(approval: PendingApproval, decision: Decision) {
    setDeciding(true);
    try {
      await run.decide(approval.run_id, { tool_call_id: approval.tool_call_id, ...decision });
    } finally {
      setDeciding(false);
    }
  }

  if (detail.isError) {
    const missing = detail.error instanceof ApiError && detail.error.status === 404;
    return (
      <div className="flex h-full flex-col items-center justify-center gap-3 p-8 text-center">
        <p className="font-medium">{missing ? "This conversation does not exist." : "The conversation could not be loaded."}</p>
        <Link to="/" className="text-sm font-medium text-brand-600 hover:underline dark:text-brand-300">
          Start a new chat
        </Link>
      </div>
    );
  }

  const awaitingApproval = turns.some((t) => t.approval);
  return (
    <div className="flex h-full min-h-0 flex-col">
      <header className="flex items-center justify-between gap-3 border-b border-zinc-200 bg-white/80 px-4 py-2.5 backdrop-blur md:px-6 dark:border-zinc-800 dark:bg-zinc-900/80">
        <h1 className="truncate text-sm font-semibold">{detail.data?.conversation.title ?? "Loading…"}</h1>
        {detail.data && <DocumentPicker conversationId={conversationId} attached={detail.data.documents} />}
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto max-w-3xl space-y-8 px-4 py-6">
          {detail.isPending && <Loader2 aria-label="Loading" className="mx-auto size-5 animate-spin text-zinc-400" />}
          {detail.data && turns.length === 0 && (
            <p className="py-16 text-center text-sm text-zinc-500">Ask a question to get started.</p>
          )}
          {turns.map((turn) => (
            <Turn
              key={turn.key}
              turn={turn}
              deciding={deciding}
              onDecide={decide}
              onCitation={setCitation}
              onRetry={
                turn === lastTurn && turn.live && turn.error && !turn.runId && run.pendingText
                  ? () => void run.send(run.pendingText as string)
                  : undefined
              }
            />
          ))}
          <div ref={bottom} />
        </div>
      </div>

      <div className="border-t border-zinc-200 bg-zinc-50 px-4 pt-3 pb-2 dark:border-zinc-800 dark:bg-zinc-950">
        <Composer
          disabled={run.busy || deciding}
          onSend={(text) => void run.send(text)}
          hint={awaitingApproval && !run.busy ? "Sending a new message cancels the change that is waiting for approval." : undefined}
        />
      </div>

      {citation && <CitationDrawer citation={citation} onClose={() => setCitation(null)} />}
    </div>
  );
}

const CAPABILITIES = [
  { icon: BookOpen, title: "Publications catalogue", text: "Counts, trends, authors, topics and full records, with citations." },
  { icon: FileSearch, title: "Your documents", text: "Upload PDF, Word, text or Markdown files and ask about them." },
  { icon: MessageCircleQuestion, title: "General questions", text: "Explanations and definitions that need no data." },
];

export function NewChatPage() {
  const navigate = useNavigate();
  const queryClient = useQueryClient();
  const [starting, setStarting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function start(text: string) {
    setStarting(true);
    setError(null);
    try {
      const conversation = await api.createConversation();
      void queryClient.invalidateQueries({ queryKey: ["conversations"] });
      navigate(`/c/${conversation.id}`, { state: { initialMessage: text } });
    } catch (err) {
      setError(err instanceof ApiError ? err.message : "Could not start a conversation. Is the backend running?");
      setStarting(false);
    }
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="min-h-0 flex-1 overflow-y-auto">
        <div className="mx-auto flex max-w-3xl flex-col gap-8 px-4 py-12 md:py-20">
          <div className="text-center">
            <h1 className="text-2xl font-semibold tracking-tight md:text-3xl">What would you like to know?</h1>
            <p className="mt-2 text-sm text-zinc-500">
              An AI agent that plans, calls tools, and cites where every answer came from.
            </p>
          </div>
          <div className="grid gap-3 md:grid-cols-3">
            {CAPABILITIES.map(({ icon: Icon, title, text }) => (
              <div key={title} className="rounded-xl border border-zinc-200 bg-white p-4 dark:border-zinc-800 dark:bg-zinc-900">
                <Icon aria-hidden className="size-5 text-brand-600 dark:text-brand-300" />
                <p className="mt-2 text-sm font-semibold">{title}</p>
                <p className="mt-1 text-xs leading-relaxed text-zinc-500">{text}</p>
              </div>
            ))}
          </div>
          <div className="grid gap-2 md:grid-cols-2">
            {EXAMPLES.map((example) => (
              <button
                key={example}
                type="button"
                disabled={starting}
                onClick={() => void start(example)}
                className="rounded-xl border border-zinc-200 bg-white px-4 py-3 text-left text-sm text-zinc-700 hover:border-brand-300 hover:bg-brand-50 disabled:opacity-60 dark:border-zinc-800 dark:bg-zinc-900 dark:text-zinc-300 dark:hover:bg-zinc-800"
              >
                {example}
              </button>
            ))}
          </div>
          {error && (
            <p role="alert" className="text-center text-sm text-rose-600">
              {error}
            </p>
          )}
        </div>
      </div>
      <div className="px-4 pt-3 pb-2">
        <Composer disabled={starting} onSend={(text) => void start(text)} />
      </div>
    </div>
  );
}
