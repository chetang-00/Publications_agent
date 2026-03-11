// Mirrors backend/app/agent/events.py and backend/app/api/schemas.py.

export type ToolCallStatus = "ok" | "invalid_arguments" | "error" | "timeout" | "rejected" | "unknown_tool";

export interface Citation {
  kind: "publication" | "document";
  id: string;
  marker: string;
  title?: string | null;
  filename?: string | null;
  page?: number | null;
  chunk_index?: number | null;
}

export interface RunStartedEvent {
  type: "run_started";
  run_id: string;
  conversation_id: string;
  user_message_id: number | null;
}

export interface TokenEvent {
  type: "token";
  text: string;
  step: number;
}

export interface ToolCallStartedEvent {
  type: "tool_call_started";
  tool_call_id: string;
  name: string;
  arguments: Record<string, unknown> | null;
  step: number;
}

export interface ToolCallFinishedEvent {
  type: "tool_call_finished";
  tool_call_id: string;
  name: string;
  status: ToolCallStatus;
  result_preview: string;
  duration_ms: number | null;
  step: number;
}

export interface ApprovalRequiredEvent {
  type: "approval_required";
  run_id: string;
  tool_call_id: string;
  name: string;
  arguments: Record<string, unknown>;
  context: Record<string, unknown>;
}

export interface MessageCompletedEvent {
  type: "message_completed";
  run_id: string;
  message_id: number;
  content: string;
  citations: Citation[];
  usage: { prompt_tokens: number; completion_tokens: number };
  steps: number;
}

export interface RunErrorEvent {
  type: "error";
  code: string;
  message: string;
  run_id: string | null;
}

export type AgentEvent =
  | RunStartedEvent
  | TokenEvent
  | ToolCallStartedEvent
  | ToolCallFinishedEvent
  | ApprovalRequiredEvent
  | MessageCompletedEvent
  | RunErrorEvent;

export interface Conversation {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
}

export interface ChatMessage {
  id: number;
  run_id: string | null;
  role: "user" | "assistant";
  content: string;
  citations: Citation[];
  created_at: string;
}

export interface PendingApproval {
  run_id: string;
  tool_call_id: string;
  name: string;
  arguments: Record<string, unknown>;
  context: Record<string, unknown>;
}

export type RunStatus = "running" | "awaiting_approval" | "completed" | "failed" | "cancelled";

export interface Run {
  id: string;
  status: RunStatus;
  steps: number;
  model: string;
  prompt_tokens: number;
  completion_tokens: number;
  error_code: string | null;
  error: string | null;
  started_at: string;
  finished_at: string | null;
  pending_approval: PendingApproval | null;
}

export interface ToolCall {
  run_id: string;
  call_id: string;
  step: number;
  name: string;
  arguments: Record<string, unknown> | null;
  status: ToolCallStatus | "awaiting_approval";
  result_preview: string | null;
  error: string | null;
  duration_ms: number | null;
}

export type DocumentStatus = "processing" | "ready" | "failed";

export interface DocumentInfo {
  id: string;
  filename: string;
  content_type: string;
  size_bytes: number;
  status: DocumentStatus;
  error: string | null;
  page_count: number | null;
  chunk_count: number | null;
  created_at: string;
}

export interface DocumentChunk {
  document_id: string;
  filename: string;
  chunk_index: number;
  page: number | null;
  text: string;
}

export interface ConversationDetail {
  conversation: Conversation;
  messages: ChatMessage[];
  runs: Run[];
  tool_calls: ToolCall[];
  documents: DocumentInfo[];
}

export interface Publication {
  id: number;
  eid: string;
  title: string;
  year: number | null;
  authors: string[];
  author_full_names: string[];
  source_title: string | null;
  publisher: string | null;
  document_type: string | null;
  doi: string | null;
  link: string | null;
  cited_by: number | null;
  open_access: string | null;
  affiliations: string | null;
  abstract: string | null;
  author_keywords: string[];
  index_keywords: string[];
  cluster_label: string | null;
}

export interface ToolInfo {
  name: string;
  description: string;
  requires_approval: boolean;
  parameters: Record<string, unknown>;
}

export interface ApprovalDecision {
  tool_call_id: string;
  approved: boolean;
  note?: string;
}
