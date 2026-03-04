import { describe, it, expect, beforeEach } from "vitest";
import { TierClassifier } from "../../components/graph/TierClassifier";

describe("TierClassifier", () => {
  let classifier: TierClassifier;

  beforeEach(() => {
    classifier = new TierClassifier();
  });

  describe("tier assignment", () => {
    it("classifies high activation as focus", () => {
      expect(classifier.classify("n1", 0.85)).toBe("focus");
    });

    it("classifies mid activation as active", () => {
      expect(classifier.classify("n1", 0.5)).toBe("active");
    });

    it("classifies low activation as dormant", () => {
      expect(classifier.classify("n1", 0.05)).toBe("dormant");
    });

    it("classifies zero activation as dormant (not hidden)", () => {
      expect(classifier.classify("n1", 0)).toBe("dormant");
    });

    it("classifies exactly at focus threshold as focus", () => {
      expect(classifier.classify("n1", 0.70)).toBe("focus");
    });

    it("classifies exactly at active threshold as active", () => {
      expect(classifier.classify("n1", 0.15)).toBe("active");
    });

    it("classifies just below active threshold as dormant", () => {
      expect(classifier.classify("n1", 0.14)).toBe("dormant");
    });
  });

  describe("hysteresis", () => {
    it("stays in focus when dropping slightly below enter threshold", () => {
      classifier.classify("n1", 0.75); // Enter focus
      // 0.67 is below focus enter (0.70) but above exit (0.65)
      expect(classifier.classify("n1", 0.67)).toBe("focus");
    });

    it("exits focus when dropping below exit threshold", () => {
      classifier.classify("n1", 0.75);
      // 0.64 is below focus exit (0.65)
      expect(classifier.classify("n1", 0.64)).toBe("active");
    });

    it("stays in active when dropping slightly below enter threshold", () => {
      classifier.classify("n1", 0.5); // Enter active
      // 0.12 is below active enter (0.15) but above exit (0.10)
      expect(classifier.classify("n1", 0.12)).toBe("active");
    });

    it("exits active when dropping below exit threshold", () => {
      classifier.classify("n1", 0.5);
      // 0.09 is below active exit (0.10)
      expect(classifier.classify("n1", 0.09)).toBe("dormant");
    });

    it("prevents flicker at focus boundary", () => {
      // Oscillate around the focus boundary
      classifier.classify("n1", 0.72); // focus
      expect(classifier.classify("n1", 0.68)).toBe("focus"); // hysteresis holds
      expect(classifier.classify("n1", 0.71)).toBe("focus"); // still focus
      expect(classifier.classify("n1", 0.66)).toBe("focus"); // still above exit
      expect(classifier.classify("n1", 0.64)).toBe("active"); // finally drops
    });

    it("allows promotion from dormant back to focus", () => {
      classifier.classify("n1", 0.01); // dormant
      expect(classifier.classify("n1", 0.75)).toBe("focus");
    });

    it("every node is always rendered (no deep tier)", () => {
      // Even activation = 0 gives dormant, not hidden
      expect(classifier.classify("n1", 0)).toBe("dormant");
      expect(classifier.classify("n2", 0.001)).toBe("dormant");
      expect(classifier.classify("n3", -1)).toBe("dormant");
    });
  });

  describe("transitions", () => {
    it("records transition on tier change", () => {
      classifier.classify("n1", 0.8); // focus
      classifier.classify("n1", 0.4); // drops to active

      const state = classifier.getState("n1");
      expect(state).toBeDefined();
      expect(state!.currentTier).toBe("active");
      expect(state!.previousTier).toBe("focus");
      expect(state!.transitionProgress).toBe(0);
      expect(state!.transitionStart).not.toBeNull();
    });

    it("updateTransitions progresses transitions", () => {
      classifier.classify("n1", 0.8);
      classifier.classify("n1", 0.4);

      const state = classifier.getState("n1")!;
      const start = state.transitionStart!;

      // Halfway through 300ms transition
      const transitioning = classifier.updateTransitions(start + 150);
      expect(transitioning).toContain("n1");
      expect(state.transitionProgress).toBeCloseTo(0.5, 1);
    });

    it("completes transition after duration", () => {
      classifier.classify("n1", 0.8);
      classifier.classify("n1", 0.4);

      const state = classifier.getState("n1")!;
      const start = state.transitionStart!;

      const transitioning = classifier.updateTransitions(start + 300);
      expect(transitioning).not.toContain("n1");
      expect(state.transitionProgress).toBe(1);
      expect(state.previousTier).toBeNull();
    });

    it("no initial transition for new nodes", () => {
      classifier.classify("n1", 0.8);
      const state = classifier.getState("n1")!;
      expect(state.transitionProgress).toBe(1);
      expect(state.previousTier).toBeNull();
    });
  });

  describe("getNodesInTier", () => {
    it("returns correct nodes per tier", () => {
      classifier.classify("a", 0.8);
      classifier.classify("b", 0.5);
      classifier.classify("c", 0.01);

      expect(classifier.getNodesInTier("focus")).toEqual(["a"]);
      expect(classifier.getNodesInTier("active")).toEqual(["b"]);
      expect(classifier.getNodesInTier("dormant")).toEqual(["c"]);
    });
  });

  describe("removal and cleanup", () => {
    it("removeNode clears state", () => {
      classifier.classify("n1", 0.8);
      expect(classifier.size).toBe(1);
      classifier.removeNode("n1");
      expect(classifier.size).toBe(0);
      expect(classifier.getState("n1")).toBeUndefined();
    });

    it("clear removes all state", () => {
      classifier.classify("a", 0.8);
      classifier.classify("b", 0.5);
      classifier.clear();
      expect(classifier.size).toBe(0);
    });
  });
});
