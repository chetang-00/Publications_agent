import { describe, expect, it } from "vitest";
import { formatBytes, formatDuration, humanizeToolName, relativeTime } from "./format";

describe("format helpers", () => {
  it("formats bytes", () => {
    expect(formatBytes(512)).toBe("512 B");
    expect(formatBytes(1536)).toBe("1.5 KB");
    expect(formatBytes(2 * 1024 * 1024)).toBe("2.0 MB");
  });

  it("formats durations", () => {
    expect(formatDuration(null)).toBe("");
    expect(formatDuration(120)).toBe("120 ms");
    expect(formatDuration(1234)).toBe("1.2 s");
  });

  it("humanizes tool names", () => {
    expect(humanizeToolName("search_publications")).toBe("Search publications");
    expect(humanizeToolName("run_readonly_sql")).toBe("Run read-only SQL");
  });

  it("describes relative times", () => {
    const now = Date.parse("2026-10-03T12:00:00Z");
    expect(relativeTime("2026-10-03T11:59:40Z", now)).toBe("just now");
    expect(relativeTime("2026-10-03T11:55:00Z", now)).toBe("5 min ago");
    expect(relativeTime("2026-10-03T09:00:00Z", now)).toBe("3 h ago");
    expect(relativeTime("2026-10-01T12:00:00Z", now)).toBe("2 d ago");
  });
});

describe("server timestamps without a timezone", () => {
  it("are treated as UTC", () => {
    const now = Date.parse("2026-10-03T12:00:00Z");
    expect(relativeTime("2026-10-03T09:00:00", now)).toBe("3 h ago");
  });
});
