import type { AgentEvent } from "../api/types";

/**
 * Incremental Server-Sent Events parser. EventSource cannot POST, so the stream is read with
 * fetch and fed here chunk by chunk; events may be split anywhere across chunks.
 */
export function createSSEParser(onEvent: (event: AgentEvent) => void) {
  let buffer = "";

  function dispatch(block: string) {
    const data: string[] = [];
    for (const line of block.split("\n")) {
      if (line.startsWith(":")) continue; // comment / keep-alive
      if (line.startsWith("data:")) data.push(line.slice(line.startsWith("data: ") ? 6 : 5));
    }
    if (data.length === 0) return;
    try {
      onEvent(JSON.parse(data.join("\n")) as AgentEvent);
    } catch {
      // A malformed event must not break the rest of the stream.
    }
  }

  return {
    feed(chunk: string) {
      // A "\r" at the end of one chunk may pair with "\n" at the start of the next, so CRLF is
      // normalised on the whole buffer, never per chunk.
      buffer = (buffer + chunk).replace(/\r\n/g, "\n");
      let boundary = buffer.indexOf("\n\n");
      while (boundary >= 0) {
        dispatch(buffer.slice(0, boundary));
        buffer = buffer.slice(boundary + 2);
        boundary = buffer.indexOf("\n\n");
      }
    },
    flush() {
      if (buffer.trim()) dispatch(buffer.replace(/\r$/, ""));
      buffer = "";
    },
  };
}
