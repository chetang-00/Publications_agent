import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { Citation } from "../api/types";
import { Markdown } from "./Markdown";

const citations: Citation[] = [{ kind: "publication", id: "6", marker: "pub:6", title: "DNA repair in yeast" }];

describe("Markdown answer", () => {
  it("renders markdown and citation chips", async () => {
    const onCitation = vi.fn();
    render(<Markdown content={"**Bold** claim [pub:6].\n\n| a | b |\n|---|---|\n| 1 | 2 |"} citations={citations} onCitation={onCitation} />);
    expect(screen.getByText("Bold").tagName).toBe("STRONG");
    expect(screen.getByRole("table")).toBeInTheDocument();
    const chip = screen.getByRole("button", { name: /citation 1/i });
    expect(chip).toHaveAttribute("title", "DNA repair in yeast");
    await userEvent.click(chip);
    expect(onCitation).toHaveBeenCalledWith(citations[0]);
  });

  it("opens ordinary links in a new tab", () => {
    render(<Markdown content="See [docs](https://example.org)." citations={[]} />);
    const link = screen.getByRole("link", { name: "docs" });
    expect(link).toHaveAttribute("target", "_blank");
    expect(link).toHaveAttribute("rel", "noopener noreferrer");
  });
});
