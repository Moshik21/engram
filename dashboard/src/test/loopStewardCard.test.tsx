import { render, screen, waitFor } from "@testing-library/react";
import { describe, expect, it, vi, beforeEach } from "vitest";
import { LoopStewardCard } from "../components/LoopStewardCard";

vi.mock("../api/client", () => ({
  api: {
    getLoopStatus: vi.fn(),
  },
}));

import { api } from "../api/client";

describe("LoopStewardCard", () => {
  beforeEach(() => {
    vi.mocked(api.getLoopStatus).mockReset();
  });

  it("shows none when inactive", async () => {
    vi.mocked(api.getLoopStatus).mockResolvedValue({
      active: false,
      adjustment: null,
      remaining_ttl_seconds: 0,
    });
    render(<LoopStewardCard />);
    await waitFor(() => {
      expect(screen.getByTestId("loop-steward-active")).toHaveTextContent("none");
    });
    expect(screen.getByTestId("loop-steward-empty")).toBeInTheDocument();
  });

  it("shows regime TTL and budgets when active", async () => {
    vi.mocked(api.getLoopStatus).mockResolvedValue({
      active: true,
      regime: "debt_heavy",
      reason: "deferred high",
      remaining_ttl_seconds: 3600,
      expires_at: "2026-07-11T00:00:00+00:00",
      adjustment: {
        regime: "debt_heavy",
        reason: "deferred high",
        budgets: { evidence_drain: 2000, cue_hygiene: 500 },
        phase_boost: ["evidence_adjudication"],
        phase_defer: ["dream"],
        created_by: "harness:test",
      },
    });
    render(<LoopStewardCard />);
    await waitFor(() => {
      expect(screen.getByTestId("loop-steward-regime")).toHaveTextContent("debt_heavy");
    });
    expect(screen.getByTestId("loop-steward-ttl")).toHaveTextContent("3600");
    expect(screen.getByTestId("loop-steward-budgets")).toHaveTextContent("evidence_drain=2000");
    expect(screen.getByTestId("loop-steward-phases")).toHaveTextContent("dream");
  });
});
