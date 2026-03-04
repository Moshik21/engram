/**
 * Bidirectional adjacency map for O(degree) neighbor lookup.
 * Replaces O(P * E) per-frame edge scans during pulse cascade.
 */

export interface Neighbor {
  neighborId: string;
  linkRef: Record<string, unknown>;
}

export class AdjacencyMap {
  private adj = new Map<string, Neighbor[]>();

  /** Rebuild from edge list. Handles d3-force object-sources (source/target can be string or {id}). */
  rebuild(links: Array<Record<string, unknown>>): void {
    this.adj.clear();

    for (const link of links) {
      const src = extractId(link.source);
      const tgt = extractId(link.target);
      if (!src || !tgt) continue;

      let srcNeighbors = this.adj.get(src);
      if (!srcNeighbors) {
        srcNeighbors = [];
        this.adj.set(src, srcNeighbors);
      }
      srcNeighbors.push({ neighborId: tgt, linkRef: link });

      let tgtNeighbors = this.adj.get(tgt);
      if (!tgtNeighbors) {
        tgtNeighbors = [];
        this.adj.set(tgt, tgtNeighbors);
      }
      tgtNeighbors.push({ neighborId: src, linkRef: link });
    }
  }

  /** Get neighbors of a node in O(degree) */
  getNeighbors(nodeId: string): Neighbor[] {
    return this.adj.get(nodeId) ?? [];
  }

  /** Number of distinct nodes in the map */
  get nodeCount(): number {
    return this.adj.size;
  }

  clear(): void {
    this.adj.clear();
  }
}

/** Extract node ID from a d3-force link endpoint (string or object with .id) */
function extractId(endpoint: unknown): string | null {
  if (typeof endpoint === "string") return endpoint;
  if (endpoint && typeof endpoint === "object" && "id" in endpoint) {
    return String((endpoint as { id: unknown }).id);
  }
  return null;
}
