import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import type { PendingApproval } from "../api/types";
import { ApprovalCard } from "./ApprovalCard";

const approval: PendingApproval = {
  run_id: "r1",
  tool_call_id: "w1",
  name: "update_cluster_label",
  arguments: { publication_id: 11, new_label: "Astrophysic", reason: "It is about galaxies" },
  context: {
    title: "Galaxy formation in dark matter halos",
    current_label: null,
    proposed_label: "Astrophysic",
    label_exists: false,
    closest_existing_labels: ["Astrophysics"],
    reason: "It is about galaxies",
  },
};

describe("ApprovalCard", () => {
  it("explains the proposed change", () => {
    render(<ApprovalCard approval={approval} busy={false} onDecide={() => {}} />);
    expect(screen.getByText("Galaxy formation in dark matter halos")).toBeInTheDocument();
    expect(screen.getByText("Unlabelled")).toBeInTheDocument();
    expect(screen.getByText("Astrophysic")).toBeInTheDocument();
    expect(screen.getByText("It is about galaxies")).toBeInTheDocument();
    expect(screen.getByText(/new label/i)).toBeInTheDocument();
    expect(screen.getByText("Astrophysics")).toBeInTheDocument();
  });

  it("approves", async () => {
    const onDecide = vi.fn();
    render(<ApprovalCard approval={approval} busy={false} onDecide={onDecide} />);
    await userEvent.click(screen.getByRole("button", { name: "Approve" }));
    expect(onDecide).toHaveBeenCalledWith({ approved: true });
  });

  it("rejects with an optional note", async () => {
    const onDecide = vi.fn();
    render(<ApprovalCard approval={approval} busy={false} onDecide={onDecide} />);
    await userEvent.type(screen.getByLabelText(/note/i), "Use Astrophysics");
    await userEvent.click(screen.getByRole("button", { name: "Reject" }));
    expect(onDecide).toHaveBeenCalledWith({ approved: false, note: "Use Astrophysics" });
  });

  it("disables buttons while a decision is being sent", () => {
    render(<ApprovalCard approval={approval} busy onDecide={() => {}} />);
    expect(screen.getByRole("button", { name: "Approve" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Reject" })).toBeDisabled();
  });
});
