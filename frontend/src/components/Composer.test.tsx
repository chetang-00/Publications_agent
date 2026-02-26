import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { Composer } from "./Composer";

describe("Composer", () => {
  it("keeps its natural one-line height while empty", () => {
    render(<Composer disabled={false} onSend={() => {}} />);
    expect(screen.getByRole("textbox", { name: "Message" }).style.height).toBe("");
  });

  it("sends trimmed text on Enter and clears", async () => {
    const onSend = vi.fn();
    render(<Composer disabled={false} onSend={onSend} />);
    const box = screen.getByRole("textbox", { name: "Message" });
    await userEvent.type(box, "  hello  {Enter}");
    expect(onSend).toHaveBeenCalledWith("hello");
    expect(box).toHaveValue("");
  });

  it("inserts a newline on Shift+Enter instead of sending", async () => {
    const onSend = vi.fn();
    render(<Composer disabled={false} onSend={onSend} />);
    await userEvent.type(screen.getByRole("textbox", { name: "Message" }), "a{Shift>}{Enter}{/Shift}b");
    expect(onSend).not.toHaveBeenCalled();
    expect(screen.getByRole("textbox", { name: "Message" })).toHaveValue("a\nb");
  });

  it("does not send while disabled", async () => {
    const onSend = vi.fn();
    render(<Composer disabled onSend={onSend} />);
    await userEvent.type(screen.getByRole("textbox", { name: "Message" }), "hi{Enter}");
    expect(onSend).not.toHaveBeenCalled();
  });
});
