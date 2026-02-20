import { describe, expect, it } from "vitest";
import type { AgentEvent } from "../api/types";
import { createSSEParser } from "./sse";

function collect() {
  const events: AgentEvent[] = [];
  const parser = createSSEParser((event) => events.push(event));
  return { events, parser };
}

const token = (text: string) => `event: token\ndata: ${JSON.stringify({ type: "token", text, step: 1 })}\n\n`;

describe("createSSEParser", () => {
  it("parses complete events", () => {
    const { events, parser } = collect();
    parser.feed(token("Hel") + token("lo"));
    expect(events).toEqual([
      { type: "token", text: "Hel", step: 1 },
      { type: "token", text: "lo", step: 1 },
    ]);
  });

  it("joins events split across network chunks", () => {
    const { events, parser } = collect();
    const raw = token("split me");
    parser.feed(raw.slice(0, 7));
    parser.feed(raw.slice(7, 30));
    expect(events).toHaveLength(0);
    parser.feed(raw.slice(30));
    expect(events).toEqual([{ type: "token", text: "split me", step: 1 }]);
  });

  it("ignores keep-alive comments", () => {
    const { events, parser } = collect();
    parser.feed(": ping\n\n" + token("x") + ": ping\n\n");
    expect(events).toHaveLength(1);
  });

  it("joins multi-line data fields", () => {
    const { events, parser } = collect();
    parser.feed('data: {"type": "token",\ndata:  "text": "a", "step": 2}\n\n');
    expect(events).toEqual([{ type: "token", text: "a", step: 2 }]);
  });

  it("handles CRLF line endings even when split between chunks", () => {
    const { events, parser } = collect();
    parser.feed('data: {"type":"token","text":"crlf","step":1}\r');
    parser.feed("\n\r\n");
    expect(events).toEqual([{ type: "token", text: "crlf", step: 1 }]);
  });

  it("flushes a final event without a trailing blank line", () => {
    const { events, parser } = collect();
    parser.feed('data: {"type":"token","text":"end","step":1}');
    expect(events).toHaveLength(0);
    parser.flush();
    expect(events).toHaveLength(1);
  });

  it("skips malformed JSON instead of throwing", () => {
    const { events, parser } = collect();
    parser.feed("data: {not json}\n\n" + token("ok"));
    expect(events).toEqual([{ type: "token", text: "ok", step: 1 }]);
  });
});
