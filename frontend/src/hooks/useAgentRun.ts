import { useQueryClient } from "@tanstack/react-query";
import { useCallback, useReducer, useRef, useState } from "react";
import { ApiError, decideApproval, sendMessage } from "../api/client";
import type { AgentEvent, ConversationDetail } from "../api/types";
import { initialRun, runReducer } from "../lib/runReducer";

type Stream = (onEvent: (event: AgentEvent) => void, signal: AbortSignal) => Promise<void>;

/**
 * Drives one live agent run for a conversation: streams events into the run reducer, then
 * refetches the conversation and hands over to the saved history once it contains the run.
 */
export function useAgentRun(conversationId: string) {
  const queryClient = useQueryClient();
  const [state, dispatch] = useReducer(runReducer, initialRun);
  const [pendingText, setPendingText] = useState<string | null>(null);
  const controller = useRef<AbortController | null>(null);
  const runId = useRef<string | null>(null);
  // Runs streamed in this session keep their tool steps expanded after they finish.
  const [seenRunIds, setSeenRunIds] = useState<ReadonlySet<string>>(() => new Set());
  const remember = useCallback((id: string) => setSeenRunIds((ids) => (ids.has(id) ? ids : new Set(ids).add(id))), []);

  const consume = useCallback(
    async (stream: Stream) => {
      controller.current?.abort();
      const ctrl = new AbortController();
      controller.current = ctrl;
      const onEvent = (event: AgentEvent) => {
        if (event.type === "run_started") {
          runId.current = event.run_id;
          remember(event.run_id);
        }
        dispatch(event);
      };
      try {
        await stream(onEvent, ctrl.signal);
      } catch (err) {
        if (ctrl.signal.aborted) return;
        dispatch({
          type: "error",
          code: err instanceof ApiError ? err.code : "network_error",
          message:
            err instanceof ApiError
              ? err.message
              : "The connection was interrupted. The answer may still be completing; reload the conversation in a moment.",
          run_id: runId.current,
        });
      }
      if (ctrl.signal.aborted) return;

      const key = ["conversation", conversationId];
      await Promise.all([
        queryClient.invalidateQueries({ queryKey: key }),
        queryClient.invalidateQueries({ queryKey: ["conversations"] }),
      ]);
      const saved = queryClient.getQueryData<ConversationDetail>(key);
      if (runId.current && saved?.runs.some((r) => r.id === runId.current)) {
        dispatch({ type: "reset" }); // the saved history now shows this run
        setPendingText(null);
      }
    },
    [conversationId, queryClient, remember],
  );

  const send = useCallback(
    (text: string) => {
      runId.current = null;
      dispatch({ type: "reset" });
      setPendingText(text);
      return consume((onEvent, signal) => sendMessage(conversationId, text, onEvent, signal));
    },
    [conversationId, consume],
  );

  const decide = useCallback(
    (approvalRunId: string, decision: { tool_call_id: string; approved: boolean; note?: string }) => {
      runId.current = approvalRunId;
      remember(approvalRunId);
      dispatch({ type: "resume", runId: approvalRunId });
      return consume((onEvent, signal) => decideApproval(approvalRunId, decision, onEvent, signal));
    },
    [consume, remember],
  );

  return { state, pendingText, seenRunIds, busy: state.status === "streaming", send, decide };
}
