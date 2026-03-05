import { describe, it, expect, beforeEach } from "vitest";
import { LODController } from "../components/graph/LODController";

describe("LODController", () => {
  let lod: LODController;

  beforeEach(() => {
    lod = new LODController();
  });

  describe("zoom tier classification", () => {
    it("starts at neighborhood tier", () => {
      expect(lod.tier).toBe("neighborhood");
    });

    it("classifies synapse tier when very close", () => {
      lod.updateCamera(0, 0, 50, Date.now());
      expect(lod.tier).toBe("synapse");
    });

    it("classifies detail tier at medium-close distance", () => {
      lod.updateCamera(0, 0, 150, Date.now());
      expect(lod.tier).toBe("detail");
    });

    it("classifies neighborhood tier at medium distance", () => {
      lod.updateCamera(0, 0, 400, Date.now());
      expect(lod.tier).toBe("neighborhood");
    });

    it("classifies region tier at far distance", () => {
      lod.updateCamera(0, 0, 1000, Date.now());
      expect(lod.tier).toBe("region");
    });

    it("classifies macro tier at very far distance", () => {
      lod.updateCamera(0, 0, 2000, Date.now());
      expect(lod.tier).toBe("macro");
    });
  });

  describe("hysteresis", () => {
    it("resists tier change at boundary (exit threshold)", () => {
      // Enter detail tier
      lod.updateCamera(0, 0, 150, Date.now());
      expect(lod.tier).toBe("detail");

      // Move slightly past enter threshold for neighborhood (200)
      // but still within exit threshold for detail (250)
      lod.updateCamera(0, 0, 220, Date.now());
      expect(lod.tier).toBe("detail"); // Should stay due to hysteresis
    });

    it("transitions when past exit threshold", () => {
      lod.updateCamera(0, 0, 150, Date.now());
      expect(lod.tier).toBe("detail");

      // Move well past exit threshold
      lod.updateCamera(0, 0, 300, Date.now());
      expect(lod.tier).toBe("neighborhood");
    });
  });

  describe("centroid tracking", () => {
    it("updates centroid from node positions", () => {
      const nodes = [
        { x: 100, y: 0, z: 0 },
        { x: -100, y: 0, z: 0 },
      ];
      lod.updateCentroid(nodes);
      // Centroid should be at ~(0,0,0)
      // Camera at (0,0,50) should be 50 units from centroid
      lod.updateCamera(0, 0, 50, Date.now());
      expect(lod.tier).toBe("synapse");
    });

    it("handles empty node array", () => {
      lod.updateCentroid([]);
      // Should not crash, tier unchanged
      expect(lod.tier).toBe("neighborhood");
    });
  });

  describe("visibility budgeting", () => {
    it("shows all nodes when budget exceeds count", () => {
      const activations = new Map([
        ["a", 0.5],
        ["b", 0.3],
      ]);
      // Enter detail tier (budget=5000)
      lod.updateCamera(0, 0, 150, Date.now());
      lod.invalidateVisibility();
      lod.rebuildVisibility(activations, ["a", "b"]);

      expect(lod.isVisible("a")).toBe(true);
      expect(lod.isVisible("b")).toBe(true);
    });

    it("caps visibility to budget at macro tier", () => {
      // Macro tier has budget=100
      lod.updateCamera(0, 0, 2000, Date.now());
      lod.invalidateVisibility();

      const activations = new Map<string, number>();
      const nodeIds: string[] = [];
      for (let i = 0; i < 200; i++) {
        const id = `node-${i}`;
        nodeIds.push(id);
        activations.set(id, i / 200); // Higher index = higher activation
      }

      lod.rebuildVisibility(activations, nodeIds);

      // Should cap at 100 highest activation nodes
      let visibleCount = 0;
      for (const id of nodeIds) {
        if (lod.isVisible(id)) visibleCount++;
      }
      expect(visibleCount).toBe(100);

      // Top activation nodes should be visible
      expect(lod.isVisible("node-199")).toBe(true);
      expect(lod.isVisible("node-198")).toBe(true);
    });

    it("filters by activation floor", () => {
      // Region tier has activationFloor=0.08
      lod.updateCamera(0, 0, 1000, Date.now());
      lod.invalidateVisibility();

      const activations = new Map([
        ["high", 0.5],
        ["low", 0.02], // Below floor
      ]);
      lod.rebuildVisibility(activations, ["high", "low"]);

      expect(lod.isVisible("high")).toBe(true);
      expect(lod.isVisible("low")).toBe(false);
    });

    it("requires invalidation before rebuilding", () => {
      lod.updateCamera(0, 0, 2000, Date.now());
      // First rebuild works
      lod.invalidateVisibility();
      const changed1 = lod.rebuildVisibility(
        new Map([["a", 0.5]]),
        ["a"],
      );
      expect(changed1).toBe(true);

      // Second rebuild without invalidation is a no-op
      const changed2 = lod.rebuildVisibility(
        new Map([["a", 0.5]]),
        ["a"],
      );
      expect(changed2).toBe(false);
    });
  });

  describe("node detail level", () => {
    it("returns full detail at close zoom", () => {
      lod.updateCamera(0, 0, 50, Date.now()); // synapse
      expect(lod.getNodeDetailLevel("x", 0.5)).toBe(2);
    });

    it("returns reduced detail at far zoom", () => {
      lod.updateCamera(0, 0, 2000, Date.now()); // macro
      expect(lod.getNodeDetailLevel("x", 0.5)).toBe(0);
    });

    it("promotes high-activation nodes one detail level", () => {
      lod.updateCamera(0, 0, 1000, Date.now()); // region, detailLevel=0
      // High activation (0.8) promotes from 0 → 1
      expect(lod.getNodeDetailLevel("x", 0.8)).toBe(1);
    });

    it("does not promote beyond max detail", () => {
      lod.updateCamera(0, 0, 150, Date.now()); // detail, detailLevel=2
      // Already at max, no promotion
      expect(lod.getNodeDetailLevel("x", 0.9)).toBe(2);
    });
  });

  describe("transition alpha", () => {
    it("starts transition at 0 on tier change", () => {
      const now = Date.now();
      lod.updateCamera(0, 0, 500, now); // neighborhood
      lod.updateCamera(0, 0, 50, now); // synapse — tier change
      expect(lod.alpha).toBe(0);
    });

    it("reaches 1 after 300ms", () => {
      const now = Date.now();
      lod.updateCamera(0, 0, 500, now);
      lod.updateCamera(0, 0, 50, now); // tier change at t=now
      lod.updateCamera(0, 0, 50, now + 350); // 350ms later
      expect(lod.alpha).toBe(1);
    });
  });

  describe("LOD config", () => {
    it("returns correct config for each tier", () => {
      lod.updateCamera(0, 0, 50, Date.now());
      expect(lod.config.showLabels).toBe(true);
      expect(lod.config.showEdges).toBe(true);

      lod.updateCamera(0, 0, 2000, Date.now());
      expect(lod.config.showLabels).toBe(false);
      expect(lod.config.showEdges).toBe(false);
    });
  });
});
