import { useEffect, useState } from "react";
import { useEngramStore } from "../store";
import type { ConsolidationPhaseResult } from "../store/types";

const NEURAL_STAGES: Record<string, { stage: string; color: string }> = {
  triage: { stage: "SCAN", color: "#4ecdc4" },
  merge: { stage: "SYNC", color: "#ff6b6b" },
  infer: { stage: "LINK", color: "#818cf8" },
  calibrate: { stage: "TUNE", color: "#D4A84B" },
  evidence_adjudication: { stage: "WEIGH", color: "#f472b6" },
  edge_adjudication: { stage: "WEIGH", color: "#f472b6" },
  replay: { stage: "TRACE", color: "#34d399" },
  prune: { stage: "FILTER", color: "#f87171" },
  compact: { stage: "PACK", color: "#7a7a94" },
  mature: { stage: "HARDEN", color: "#a78bfa" },
  semanticize: { stage: "DISTILL", color: "#67e8f9" },
  reflect: { stage: "SYNTH", color: "#5eead4" },
  schema: { stage: "STRATIFY", color: "#fbbf24" },
  reindex: { stage: "INDEX", color: "#7a7a94" },
  graph_embed: { stage: "VECTOR", color: "#818cf8" },
  microglia: { stage: "PURGE", color: "#34d399" },
  dream: { stage: "INFER", color: "#c084fc" },
};

function logRank(totalAffected: number): { rank: string; color: string } {
  if (totalAffected >= 50) return { rank: "S", color: "#fbbf24" };
  if (totalAffected >= 20) return { rank: "A", color: "#f87171" };
  if (totalAffected >= 10) return { rank: "B", color: "#818cf8" };
  if (totalAffected >= 3) return { rank: "C", color: "#34d399" };
  return { rank: "D", color: "#7a7a94" };
}

function formatTimeAgo(ts: number): string {
  const diff = Date.now() - ts;
  if (diff < 60_000) return "just now";
  if (diff < 3_600_000) return `${Math.floor(diff / 60_000)}m ago`;
  if (diff < 86_400_000) return `${Math.floor(diff / 3_600_000)}h ago`;
  return `${Math.floor(diff / 86_400_000)}d ago`;
}

export function PlasticityLog() {
  const cycles = useEngramStore((s) => s.cycles);
  const loadCycles = useEngramStore((s) => s.loadCycles);
  const synapticEvents = useEngramStore((s) => s.synapticEvents);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    void loadCycles();
  }, [loadCycles]);

  return (
    <div style={{ padding: 20, height: "100%", overflow: "auto" }}>
      <h2 className="display" style={{ fontSize: 22, marginBottom: 20, color: "#67e8f9" }}>
        Plasticity Log
      </h2>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: 20, maxWidth: 1000 }}>
        {/* Structural history */}
        <div className="stagger">
          <div className="label" style={{ marginBottom: 12 }}>Structural Remodeling History</div>
          {cycles.length === 0 ? (
            <div className="card" style={{ padding: 24, textAlign: "center" }}>
              <div style={{ fontSize: 32, marginBottom: 8, opacity: 0.3 }}>{"\u2699"}</div>
              <span style={{ color: "var(--text-muted)", fontSize: 13 }}>No plasticity cycles recorded</span>
              <p style={{ color: "var(--text-ghost)", fontSize: 11, marginTop: 4 }}>
                Trigger a consolidation cycle to initiate structural remodeling
              </p>
            </div>
          ) : (
            cycles.map((cycle) => {
              const totalAffected = cycle.phases.reduce((s: number, p: ConsolidationPhaseResult) => s + p.items_affected, 0);
              const { rank, color: rankColor } = logRank(totalAffected);
              const expanded = expandedId === cycle.id;
              const activePhases = cycle.phases.filter((p: ConsolidationPhaseResult) => p.status !== "skipped");

              return (
                <div key={cycle.id} className="card" style={{ marginBottom: 8, overflow: "hidden" }}>
                  {/* Header - clickable */}
                  <div
                    onClick={() => setExpandedId(expanded ? null : cycle.id)}
                    style={{
                      padding: "12px 14px", cursor: "pointer",
                      display: "flex", alignItems: "center", gap: 10,
                      borderBottom: expanded ? "1px solid var(--border)" : "none",
                    }}
                  >
                    {/* Rank badge */}
                    <div style={{
                      width: 28, height: 28, borderRadius: 4,
                      border: `2px solid ${rankColor}`,
                      display: "flex", alignItems: "center", justifyContent: "center",
                      fontFamily: "var(--font-mono)", fontSize: 14, fontWeight: 700,
                      color: rankColor, background: `${rankColor}10`,
                      flexShrink: 0,
                    }}>
                      {rank}
                    </div>

                    <div style={{ flex: 1 }}>
                      <div style={{ color: "var(--text-primary)", fontSize: 13, fontWeight: 500 }}>
                        {cycle.trigger === "pressure" ? "High Pressure Cycle" : cycle.trigger === "scheduled" ? "Scheduled Maintenance" : "Manual Optimization"}
                      </div>
                      <div className="mono" style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 1 }}>
                        {activePhases.length} phases {"\u00b7"} {totalAffected} affected {"\u00b7"} {Math.round(cycle.total_duration_ms)}ms
                      </div>
                    </div>

                    <span className="pill" style={{
                      background: cycle.status === "completed" ? "rgba(52, 211, 153, 0.1)" : "rgba(248, 113, 113, 0.1)",
                      color: cycle.status === "completed" ? "var(--success)" : "var(--danger)",
                    }}>
                      {cycle.status}
                    </span>

                    <span style={{
                      color: "var(--text-muted)", fontSize: 10,
                      transform: expanded ? "rotate(180deg)" : "none",
                      transition: "transform 0.2s",
                    }}>
                      {"\u25BC"}
                    </span>
                  </div>

                  {/* Expanded phases */}
                  {expanded && (
                    <div style={{ padding: "10px 14px" }}>
                      {cycle.phases.map((p: ConsolidationPhaseResult) => {
                        const info = NEURAL_STAGES[p.phase] ?? { stage: p.phase.toUpperCase(), color: "var(--text-muted)" };
                        const isActive = p.status !== "skipped";
                        return (
                          <div key={p.phase} style={{
                            display: "flex", alignItems: "center", gap: 8,
                            padding: "5px 0", opacity: isActive ? 1 : 0.35,
                          }}>
                            {/* Phase dot */}
                            <div style={{
                              width: 8, height: 8, borderRadius: "50%",
                              background: isActive ? info.color : "var(--text-ghost)",
                              boxShadow: isActive ? `0 0 6px ${info.color}40` : "none",
                              flexShrink: 0,
                            }} />
                            <span style={{
                              fontFamily: "var(--font-mono)", fontSize: 10, fontWeight: 600,
                              color: info.color, width: 80, letterSpacing: "0.05em",
                            }}>
                              {info.stage}
                            </span>
                            <span className="mono" style={{ fontSize: 10, color: "var(--text-secondary)", flex: 1 }}>
                              {p.items_processed > 0 && `${p.items_processed} processed`}
                              {p.items_affected > 0 && ` \u2192 ${p.items_affected} affected`}
                            </span>
                            <span className="mono" style={{ fontSize: 9, color: "var(--text-muted)" }}>
                              {p.duration_ms > 0 && `${Math.round(p.duration_ms)}ms`}
                            </span>
                            {p.status === "error" && (
                              <span style={{ color: "var(--danger)", fontSize: 10 }}>{"\u26A0"}</span>
                            )}
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>

        {/* Synaptic log */}
        <div>
          <div className="label" style={{ marginBottom: 12 }}>Synaptic Event Log</div>
          <div className="card" style={{ padding: 14 }}>
            {synapticEvents.length === 0 ? (
              <div style={{ textAlign: "center", padding: 16 }}>
                <span style={{ color: "var(--text-muted)", fontSize: 12, fontStyle: "italic" }}>
                  Sensorium is quiet...
                </span>
              </div>
            ) : (
              <div style={{ position: "relative" }}>
                {/* Timeline line */}
                <div style={{
                  position: "absolute", left: 5, top: 8, bottom: 8, width: 1,
                  background: "var(--border)",
                }} />
                {synapticEvents.slice(0, 30).map((ev) => (
                  <div key={ev.id} style={{
                    display: "flex", gap: 12, padding: "6px 0",
                    position: "relative",
                  }}>
                    {/* Timeline dot */}
                    <div style={{
                      width: 11, height: 11, borderRadius: "50%",
                      border: "2px solid var(--border-hover)",
                      background: ev.plasticity > 0 ? "#67e8f9" : "var(--surface-solid)",
                      flexShrink: 0, marginTop: 2, zIndex: 1,
                    }} />
                    <div style={{ flex: 1 }}>
                      <div style={{ color: "var(--text-primary)", fontSize: 12, lineHeight: 1.4 }}>
                        {ev.text}
                      </div>
                      <div style={{ display: "flex", gap: 8, marginTop: 2 }}>
                        {ev.plasticity > 0 && (
                          <span className="mono" style={{ fontSize: 10, color: "#67e8f9", fontWeight: 600 }}>
                            +{ev.plasticity} Plasticity
                          </span>
                        )}
                        <span className="mono" style={{ fontSize: 9, color: "var(--text-ghost)" }}>
                          {formatTimeAgo(ev.timestamp)}
                        </span>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
