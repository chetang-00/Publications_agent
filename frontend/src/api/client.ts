import { createSSEParser } from "../lib/sse";
import type {
  AgentEvent,
  ApprovalDecision,
  Conversation,
  ConversationDetail,
  DocumentInfo,
  Publication,
  ToolInfo,
} from "./types";

const BASE = "/api";

export class ApiError extends Error {
  readonly status: number;
  readonly code: string;
  readonly details?: unknown;

  constructor(status: number, code: string, message: string, details?: unknown) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

async function toApiError(response: Response): Promise<ApiError> {
  try {
    const body = (await response.json()) as { error?: { code?: string; message?: string; details?: unknown } };
    if (body.error?.code) {
      return new ApiError(response.status, body.error.code, body.error.message ?? response.statusText, body.error.details);
    }
  } catch {
    // not JSON (e.g. a proxy error page)
  }
  const message =
    response.status === 502 || response.status === 504
      ? "The server is not reachable. Is the backend running?"
      : `Request failed (${response.status}).`;
  return new ApiError(response.status, `http_${response.status}`, message);
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${BASE}${path}`, { method: "GET", ...init });
  if (!response.ok) throw await toApiError(response);
  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

function json(method: string, body: unknown): RequestInit {
  return { method, headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

export const api = {
  listConversations: () => request<Conversation[]>("/conversations"),
  createConversation: (title?: string) => request<Conversation>("/conversations", json("POST", { title })),
  getConversation: (id: string) => request<ConversationDetail>(`/conversations/${encodeURIComponent(id)}`),
  deleteConversation: (id: string) => request<void>(`/conversations/${encodeURIComponent(id)}`, { method: "DELETE" }),
  attachDocuments: (id: string, documentIds: string[]) =>
    request<{ document_ids: string[] }>(
      `/conversations/${encodeURIComponent(id)}/documents`,
      json("POST", { document_ids: documentIds }),
    ),
  listDocuments: () => request<DocumentInfo[]>("/documents"),
  uploadDocument: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request<DocumentInfo>("/documents", { method: "POST", body: form });
  },
  deleteDocument: (id: string) => request<void>(`/documents/${encodeURIComponent(id)}`, { method: "DELETE" }),
  getPublication: (id: string | number) => request<Publication>(`/publications/${encodeURIComponent(String(id))}`),
  listTools: () => request<ToolInfo[]>("/tools"),
};

/** POST a JSON body and feed the Server-Sent Events response to `onEvent` until the stream ends. */
export async function streamEvents(
  path: string,
  body: unknown,
  onEvent: (event: AgentEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${BASE}${path}`, { ...json("POST", body), signal });
  if (!response.ok) throw await toApiError(response);
  if (!response.body) throw new ApiError(response.status, "no_stream", "The server returned no event stream.");
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const parser = createSSEParser(onEvent);
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    parser.feed(decoder.decode(value, { stream: true }));
  }
  parser.feed(decoder.decode());
  parser.flush();
}

export const sendMessage = (
  conversationId: string,
  content: string,
  onEvent: (event: AgentEvent) => void,
  signal?: AbortSignal,
) => streamEvents(`/conversations/${encodeURIComponent(conversationId)}/messages`, { content }, onEvent, signal);

export const decideApproval = (
  runId: string,
  decision: ApprovalDecision,
  onEvent: (event: AgentEvent) => void,
  signal?: AbortSignal,
) => streamEvents(`/runs/${encodeURIComponent(runId)}/approvals`, decision, onEvent, signal);
