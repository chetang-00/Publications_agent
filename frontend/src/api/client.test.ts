import { afterEach, describe, expect, it, vi } from "vitest";
import type { AgentEvent } from "./types";
import { ApiError, api, sendMessage } from "./client";

function streamResponse(chunks: string[], init: ResponseInit = {}): Response {
  const encoder = new TextEncoder();
  const body = new ReadableStream<Uint8Array>({
    start(controller) {
      for (const chunk of chunks) controller.enqueue(encoder.encode(chunk));
      controller.close();
    },
  });
  return new Response(body, { status: 200, headers: { "content-type": "text/event-stream" }, ...init });
}

function jsonResponse(body: unknown, status = 200): Response {
  return new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
}

afterEach(() => vi.unstubAllGlobals());

describe("api client", () => {
  it("GETs JSON from /api", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse([{ id: "c1", title: "T", created_at: "x", updated_at: "x" }]));
    vi.stubGlobal("fetch", fetchMock);
    const conversations = await api.listConversations();
    expect(conversations[0]?.id).toBe("c1");
    expect(fetchMock).toHaveBeenCalledWith("/api/conversations", expect.objectContaining({ method: "GET" }));
  });

  it("raises ApiError with the server's error envelope", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ error: { code: "not_found", message: "Conversation not found." } }, 404)),
    );
    const error = await api.getConversation("missing").catch((e: unknown) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ status: 404, code: "not_found", message: "Conversation not found." });
  });

  it("raises a readable ApiError when the body is not JSON", async () => {
    vi.stubGlobal("fetch", vi.fn().mockResolvedValue(new Response("Bad Gateway", { status: 502 })));
    const error = await api.listDocuments().catch((e: unknown) => e);
    expect(error).toMatchObject({ status: 502, code: "http_502" });
  });

  it("uploads files as multipart form data", async () => {
    const fetchMock = vi.fn().mockResolvedValue(jsonResponse({ id: "d1", status: "processing" }, 202));
    vi.stubGlobal("fetch", fetchMock);
    await api.uploadDocument(new File(["hello"], "notes.txt", { type: "text/plain" }));
    const [, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(init.method).toBe("POST");
    expect((init.body as FormData).get("file")).toBeInstanceOf(File);
  });

  it("streams agent events from a POST", async () => {
    const event = (e: AgentEvent) => `data: ${JSON.stringify(e)}\n\n`;
    const raw =
      event({ type: "run_started", run_id: "r", conversation_id: "c", user_message_id: 1 }) +
      event({ type: "token", text: "Hi", step: 1 });
    const fetchMock = vi.fn().mockResolvedValue(streamResponse([raw.slice(0, 20), raw.slice(20)]));
    vi.stubGlobal("fetch", fetchMock);
    const received: AgentEvent[] = [];
    await sendMessage("c", "hello", (e) => received.push(e));
    expect(received.map((e) => e.type)).toEqual(["run_started", "token"]);
    const [url, init] = fetchMock.mock.calls[0] as [string, RequestInit];
    expect(url).toBe("/api/conversations/c/messages");
    expect(JSON.parse(init.body as string)).toEqual({ content: "hello" });
  });

  it("rejects a stream request that fails before streaming", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn().mockResolvedValue(jsonResponse({ error: { code: "run_in_progress", message: "busy" } }, 409)),
    );
    await expect(sendMessage("c", "x", () => {})).rejects.toMatchObject({ code: "run_in_progress" });
  });
});
