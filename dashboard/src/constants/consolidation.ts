// Dashboard mirror of backend engram.consolidation.phase_registry.CONSOLIDATION_PHASE_ORDER.
export const CONSOLIDATION_PHASE_ORDER = [
  "triage",
  "merge",
  "calibrate",
  "infer",
  "evidence_adjudication",
  "edge_adjudication",
  "replay",
  "prune",
  "compact",
  "mature",
  "semanticize",
  "schema",
  "reindex",
  "graph_embed",
  "microglia",
  "dream",
] as const;

export type ConsolidationPhaseName = (typeof CONSOLIDATION_PHASE_ORDER)[number];
