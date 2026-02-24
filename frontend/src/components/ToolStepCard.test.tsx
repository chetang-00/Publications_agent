import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it } from "vitest";
import { ToolStepCard } from "./ToolStepCard";

describe("ToolStepCard", () => {
  it("shows the tool, its arguments and status", () => {
    render(
      <ToolStepCard
        step={{
          toolCallId: "c1",
          name: "filter_publications",
          arguments: { author: "Ranganath R.", year_from: 2020 },
          status: "ok",
          step: 1,
          preview: '{"total": 6}',
          durationMs: 42,
        }}
      />,
    );
    expect(screen.getByText("Filter publications")).toBeInTheDocument();
    expect(screen.getByText(/author: Ranganath R\./)).toBeInTheDocument();
    expect(screen.getByText(/year_from: 2020/)).toBeInTheDocument();
    expect(screen.getByText("42 ms")).toBeInTheDocument();
    expect(screen.getByRole("status")).toHaveTextContent("Done");
  });

  it("expands to show the result preview", async () => {
    render(
      <ToolStepCard
        step={{ toolCallId: "c1", name: "get_publication", arguments: { publication_id: 6 }, status: "ok", step: 1, preview: "{\"title\": \"Yeast\"}" }}
      />,
    );
    expect(screen.queryByText(/"title": "Yeast"/)).not.toBeInTheDocument();
    await userEvent.click(screen.getByRole("button", { name: /details/i }));
    expect(screen.getByText(/"title": "Yeast"/)).toBeInTheDocument();
  });

  it.each([
    ["running", "Running"],
    ["invalid_arguments", "Invalid arguments"],
    ["timeout", "Timed out"],
    ["awaiting_approval", "Needs approval"],
    ["rejected", "Rejected"],
  ] as const)("labels status %s", (status, label) => {
    render(<ToolStepCard step={{ toolCallId: "x", name: "resolve_author", arguments: null, status, step: 1 }} />);
    expect(screen.getByRole("status")).toHaveTextContent(label);
  });
});
