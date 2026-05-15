import { useEffect, useState } from "react";
import { useEngramStore } from "../store";
import type { ConsolidationPhaseResult } from "../store/types";

const QUEST_STAGES: Record<string, { stage: string; color: string }> = {
  triage: { stage: "SCOUT", color: "#4ecdc4" },
  merge: { stage: "FORGE", color: "#ff6b6b" },
  infer: { stage: "DIVINE", color: "#818cf8" },
  calibrate: { stage: "ATTUNE", color: "#D4A84B" },
  evidence_adjudication: { stage: "JUDGE", color: "#f472b6" },
  edge_adjudication: { stage: "JUDGE", color: "#f472b6" },
  replay: { stage: "RELIVE", color: "#34d399" },
  prune: { stage: "PURGE", color: "#f87171" },
  compact: { stage: "CONDENSE", color: "#7a7a94" },
  mature: { stage: "EVOLVE", color: "#a78bfa" },
  semanticize: { stage: "TRANSCEND", color: "#67e8f9" },
  schema: { stage: "CRYSTALLIZE", color: "#fbbf24" },
  reindex: { stage: "CATALOG", color: "#7a7a94" },
  graph_embed: { stage: "WEAVE", color: "#818cf8" },
  microglia: { stage: "CLEANSE", color: "#34d399" },
  dream: { stage: "DREAM", color: "#c084fc" },
};

function questRank(totalAffected: number): { rank: string; color: string } {
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

export function QuestLog() {
  const cycles = useEngramStore((s) => s.cycles);
  const loadCycles = useEngramStore((s) => s.loadCycles);
  const questEvents = useEngramStore((s) => s.questEvents);
  const [expandedId, setExpandedId] = useState<string | null>(null);

  useEffect(() => {
    void loadCycles();
  }, [loadCycles]);

  return (
    <div style={{ padding: 20, height: "100%", overflow: "auto" }}>
      <h2 className="display" style={{ fontSize: 22, marginBottom: 20, color: "#D4A84B" }}>
        Quest Log
      </h2>

      <div style={{ display: "grid", gridTemplateColumns: "1fr 280px", gap: 20, maxWidth: 1000 }}>
        {/* Quests */}
        <div className="stagger">
          <div className="label" style={{ marginBottom: 12 }}>Completed Quests</div>
          {cycles.length === 0 ? (
            <div className="card" style={{ padding: 24, textAlign: "center" }}>
              <div style={{ fontSize: 32, marginBottom: 8, opacity: 0.3 }}>{"\u2721"}</div>
              <span style={{ color: "var(--text-muted)", fontSize: 13 }}>No quests undertaken yet</span>
              <p style={{ color: "var(--text-ghost)", fontSize: 11, marginTop: 4 }}>
                Trigger a consolidation cycle to begin your first quest
              </p>
            </div>
          ) : (
            cycles.map((cycle) => {
              const totalAffected = cycle.phases.reduce((s: number, p: ConsolidationPhaseResult) => s + p.items_affected, 0);
              const { rank, color: rankColor } = questRank(totalAffected);
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
                        {cycle.trigger === "pressure" ? "Urgent Quest" : cycle.trigger === "scheduled" ? "Daily Quest" : "Manual Quest"}
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
                        const info = QUEST_STAGES[p.phase] ?? { stage: p.phase.toUpperCase(), color: "var(--text-muted)" };
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

        {/* Adventure Log */}
        <div>
          <div className="label" style={{ marginBottom: 12 }}>Adventure Log</div>
          <div className="card" style={{ padding: 14 }}>
            {questEvents.length === 0 ? (
              <div style={{ textAlign: "center", padding: 16 }}>
                <span style={{ color: "var(--text-muted)", fontSize: 12, fontStyle: "italic" }}>
                  The page is blank...
                </span>
              </div>
            ) : (
              <div style={{ position: "relative" }}>
                {/* Timeline line */}
                <div style={{
                  position: "absolute", left: 5, top: 8, bottom: 8, width: 1,
                  background: "var(--border)",
                }} />
                {questEvents.slice(0, 30).map((ev) => (
                  <div key={ev.id} style={{
                    display: "flex", gap: 12, padding: "6px 0",
                    position: "relative",
                  }}>
                    {/* Timeline dot */}
                    <div style={{
                      width: 11, height: 11, borderRadius: "50%",
                      border: "2px solid var(--border-hover)",
                      background: ev.xp > 0 ? "#D4A84B" : "var(--surface-solid)",
                      flexShrink: 0, marginTop: 2, zIndex: 1,
                    }} />
                    <div style={{ flex: 1 }}>
                      <div style={{ color: "var(--text-primary)", fontSize: 12, lineHeight: 1.4 }}>
                        {ev.text}
                      </div>
                      <div style={{ display: "flex", gap: 8, marginTop: 2 }}>
                        {ev.xp > 0 && (
                          <span className="mono" style={{ fontSize: 10, color: "#D4A84B", fontWeight: 600 }}>
                            +{ev.xp} XP
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
