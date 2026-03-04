import { describe, it, expect, beforeEach } from "vitest";
import { AnimationBudget } from "../../components/graph/AnimationBudget";

describe("AnimationBudget", () => {
  let budget: AnimationBudget;

  beforeEach(() => {
    budget = new AnimationBudget();
  });

  describe("basic allocation", () => {
    it("returns pulsing nodes with priority", () => {
      const result = budget.allocate(["p1", "p2"], ["f1", "f2", "f3"]);
      expect(result[0]).toBe("p1");
      expect(result[1]).toBe("p2");
      expect(result).toContain("f1");
      expect(result).toContain("f2");
      expect(result).toContain("f3");
    });

    it("includes focus nodes not in pulse set", () => {
      const result = budget.allocate(["p1"], ["p1", "f1", "f2"]);
      expect(result).toContain("p1");
      expect(result).toContain("f1");
      expect(result).toContain("f2");
    });

    it("handles empty pulse list", () => {
      const result = budget.allocate([], ["f1", "f2"]);
      expect(result).toEqual(["f1", "f2"]);
    });

    it("handles empty focus list", () => {
      const result = budget.allocate(["p1"], []);
      expect(result).toEqual(["p1"]);
    });

    it("handles both empty", () => {
      const result = budget.allocate([], []);
      expect(result).toEqual([]);
    });
  });

  describe("budget cap", () => {
    it("caps total animated at 50", () => {
      const pulses = Array.from({ length: 10 }, (_, i) => `p${i}`);
      const focus = Array.from({ length: 100 }, (_, i) => `f${i}`);

      const result = budget.allocate(pulses, focus);
      expect(result.length).toBe(50);
    });

    it("caps pulse nodes at 50", () => {
      const pulses = Array.from({ length: 60 }, (_, i) => `p${i}`);
      const result = budget.allocate(pulses, []);
      expect(result.length).toBe(50);
    });

    it("fills remaining budget after pulses with focus nodes", () => {
      const pulses = Array.from({ length: 45 }, (_, i) => `p${i}`);
      const focus = Array.from({ length: 20 }, (_, i) => `f${i}`);

      const result = budget.allocate(pulses, focus);
      expect(result.length).toBe(50); // 45 pulses + 5 focus
    });
  });

  describe("round-robin", () => {
    it("rotates through idle focus nodes across calls", () => {
      const focus = ["f0", "f1", "f2", "f3", "f4"];

      // First call: starts from index 0
      const r1 = budget.allocate([], focus);
      expect(r1).toEqual(focus);

      // With fewer budget slots, we'd see rotation.
      // Use 48 pulses to leave only 2 focus slots
      const pulses = Array.from({ length: 48 }, (_, i) => `p${i}`);

      const r2 = budget.allocate(pulses, focus);
      // 48 pulses + 2 focus nodes
      expect(r2.length).toBe(50);
      const focusInR2 = r2.filter((id) => id.startsWith("f"));
      expect(focusInR2.length).toBe(2);

      // Next call should continue round-robin
      const r3 = budget.allocate(pulses, focus);
      const focusInR3 = r3.filter((id) => id.startsWith("f"));
      expect(focusInR3.length).toBe(2);
    });

    it("wraps round-robin index correctly", () => {
      const focus = ["f0", "f1", "f2"];
      // 49 pulses, 1 focus slot
      const pulses = Array.from({ length: 49 }, (_, i) => `p${i}`);

      const results: string[][] = [];
      for (let i = 0; i < 4; i++) {
        const r = budget.allocate(pulses, focus);
        results.push(r.filter((id) => id.startsWith("f")));
      }

      // Each call gets 1 focus node, cycling through f0, f1, f2, f0
      expect(results[0]).toEqual(["f0"]);
      expect(results[1]).toEqual(["f1"]);
      expect(results[2]).toEqual(["f2"]);
      expect(results[3]).toEqual(["f0"]);
    });
  });

  describe("pulse exclusion from focus fill", () => {
    it("does not double-count pulsing nodes in focus fill", () => {
      // p1 is both pulsing and in focus list
      const result = budget.allocate(["p1"], ["p1", "f1", "f2"]);
      // p1 should appear once (as pulse), plus f1 and f2
      const p1Count = result.filter((id) => id === "p1").length;
      expect(p1Count).toBe(1);
      expect(result).toContain("f1");
      expect(result).toContain("f2");
    });
  });

  describe("reset", () => {
    it("resets round-robin cursor", () => {
      const focus = ["f0", "f1", "f2"];
      const pulses = Array.from({ length: 49 }, (_, i) => `p${i}`);

      budget.allocate(pulses, focus); // f0
      budget.allocate(pulses, focus); // f1
      budget.reset();
      const result = budget.allocate(pulses, focus);
      expect(result.filter((id) => id.startsWith("f"))).toEqual(["f0"]);
    });
  });
});
