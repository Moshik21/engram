import { describe, it, expect, beforeEach } from "vitest";
import { AdjacencyMap } from "../../components/graph/AdjacencyMap";

describe("AdjacencyMap", () => {
  let adj: AdjacencyMap;

  beforeEach(() => {
    adj = new AdjacencyMap();
  });

  describe("rebuild with string IDs", () => {
    it("builds bidirectional adjacency from edges", () => {
      adj.rebuild([
        { source: "a", target: "b", id: "e1" },
        { source: "b", target: "c", id: "e2" },
      ]);

      const aNeighbors = adj.getNeighbors("a");
      expect(aNeighbors).toHaveLength(1);
      expect(aNeighbors[0].neighborId).toBe("b");

      const bNeighbors = adj.getNeighbors("b");
      expect(bNeighbors).toHaveLength(2);
      const bIds = bNeighbors.map((n) => n.neighborId).sort();
      expect(bIds).toEqual(["a", "c"]);

      const cNeighbors = adj.getNeighbors("c");
      expect(cNeighbors).toHaveLength(1);
      expect(cNeighbors[0].neighborId).toBe("b");
    });

    it("returns empty array for unknown node", () => {
      adj.rebuild([{ source: "a", target: "b" }]);
      expect(adj.getNeighbors("x")).toEqual([]);
    });

    it("reports correct node count", () => {
      adj.rebuild([
        { source: "a", target: "b" },
        { source: "c", target: "d" },
      ]);
      expect(adj.nodeCount).toBe(4);
    });
  });

  describe("d3-force object-source handling", () => {
    it("handles source/target as objects with .id", () => {
      adj.rebuild([
        {
          source: { id: "a", x: 10, y: 20 },
          target: { id: "b", x: 30, y: 40 },
          id: "e1",
        },
      ]);

      expect(adj.getNeighbors("a")).toHaveLength(1);
      expect(adj.getNeighbors("a")[0].neighborId).toBe("b");
      expect(adj.getNeighbors("b")).toHaveLength(1);
      expect(adj.getNeighbors("b")[0].neighborId).toBe("a");
    });

    it("handles mixed string and object endpoints", () => {
      adj.rebuild([
        {
          source: "a",
          target: { id: "b" },
        },
      ]);

      expect(adj.getNeighbors("a")).toHaveLength(1);
      expect(adj.getNeighbors("b")).toHaveLength(1);
    });

    it("skips edges with null/undefined endpoints", () => {
      adj.rebuild([
        { source: "a", target: "b" },
        { source: null, target: "c" },
        { source: "d", target: undefined },
      ]);

      expect(adj.nodeCount).toBe(2); // only a and b
    });
  });

  describe("linkRef preservation", () => {
    it("preserves link reference for particle emission", () => {
      const link = { source: "a", target: "b", id: "e1", weight: 0.5 };
      adj.rebuild([link]);

      const neighbors = adj.getNeighbors("a");
      expect(neighbors[0].linkRef).toBe(link);
    });
  });

  describe("rebuild", () => {
    it("clears previous data on rebuild", () => {
      adj.rebuild([{ source: "a", target: "b" }]);
      expect(adj.nodeCount).toBe(2);

      adj.rebuild([{ source: "x", target: "y" }]);
      expect(adj.nodeCount).toBe(2);
      expect(adj.getNeighbors("a")).toEqual([]);
      expect(adj.getNeighbors("x")).toHaveLength(1);
    });
  });

  describe("clear", () => {
    it("empties the map", () => {
      adj.rebuild([{ source: "a", target: "b" }]);
      adj.clear();
      expect(adj.nodeCount).toBe(0);
    });
  });

  describe("multi-edge handling", () => {
    it("handles multiple edges between same nodes", () => {
      adj.rebuild([
        { source: "a", target: "b", predicate: "KNOWS" },
        { source: "a", target: "b", predicate: "WORKS_WITH" },
      ]);

      // Both edges create neighbor entries
      const aNeighbors = adj.getNeighbors("a");
      expect(aNeighbors).toHaveLength(2);
      expect(aNeighbors.every((n) => n.neighborId === "b")).toBe(true);
    });

    it("handles self-loops", () => {
      adj.rebuild([{ source: "a", target: "a" }]);
      const neighbors = adj.getNeighbors("a");
      expect(neighbors).toHaveLength(2); // bidirectional entry
    });
  });
});
