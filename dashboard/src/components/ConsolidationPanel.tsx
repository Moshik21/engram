import { useEffect, useState } from "react";
import { useEngramStore } from "../store";
import type { ConsolidationCycleSummary, ConsolidationPhaseResult } from "../store/types";

const STATUS_COLORS: Record<string, string> = {
  completed: "#34d399",
  failed: "#f87171",
  running: "#22d3ee",
  cancelled: "#fbbf24",
  pending: "#94a3b8",
  success: "#34d399",
  skipped: "#94a3b8",
  error: "#f87171",
};

function statusColor(status: string): string {
  return STATUS_COLORS[status] ?? "#94a3b8";
}

function formatTime(ts: number): string {
  return new Date(ts * 1000).toLocaleString(undefined, {
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

function formatDuration(ms: number): string {
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function PressureGauge({ value, threshold }: { value: number; threshold: number }) {
  const pct = Math.min((value / threshold) * 100, 100);
  const color = pct > 80 ? "#f87171" : pct > 50 ? "#fbbf24" : "#34d399";
  return (
    <div>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span className="label">Pressure</span>
        <span className="mono tabular-nums" style={{ fontSize: 10, color }}>
          {value.toFixed(2)} / {threshold}
        </span>
      </div>
      <div className="metric-bar" style={{ height: 6 }}>
        <div
          className="metric-bar-fill"
          style={{
            width: `${pct}%`,
            background: `linear-gradient(90deg, ${color}cc, ${color})`,
            transition: "width 0.3s ease",
          }}
        />
      </div>
    </div>
  );
}

function PhaseTimeline({ phases }: { phases: ConsolidationPhaseResult[] }) {
  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 2 }}>
      {phases.map((p) => (
        <div
          key={p.phase}
          style={{
            display: "flex",
            alignItems: "center",
            gap: 8,
            padding: "6px 10px",
            borderRadius: "var(--radius-xs)",
            background: p.status === "error" ? "rgba(248, 113, 113, 0.05)" : "transparent",
          }}
        >
          <span
            style={{
              width: 7,
              height: 7,
              borderRadius: "50%",
              background: statusColor(p.status),
              flexShrink: 0,
              boxShadow: p.status === "error" ? `0 0 6px ${statusColor(p.status)}60` : "none",
            }}
          />
          <span style={{ flex: 1, fontSize: 12, color: "var(--text-primary)", textTransform: "capitalize" }}>
            {p.phase}
          </span>
          <span className="mono tabular-nums" style={{ fontSize: 10, color: "var(--text-muted)" }}>
            {p.items_processed > 0 && `${p.items_affected}/${p.items_processed}`}
          </span>
          <span className="mono tabular-nums" style={{ fontSize: 10, color: "var(--text-muted)", minWidth: 40, textAlign: "right" }}>
            {p.duration_ms > 0 ? formatDuration(p.duration_ms) : ""}
          </span>
        </div>
      ))}
    </div>
  );
}

function CycleRow({
  cycle,
  isSelected,
  onClick,
}: {
  cycle: ConsolidationCycleSummary;
  isSelected: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 8,
        width: "100%",
        padding: "8px 12px",
        border: "none",
        borderRadius: "var(--radius-xs)",
        background: isSelected ? "rgba(34, 211, 238, 0.06)" : "transparent",
        cursor: "pointer",
        transition: "background 0.12s ease",
        textAlign: "left",
      }}
      onMouseEnter={(e) => {
        if (!isSelected) e.currentTarget.style.background = "rgba(255,255,255,0.02)";
      }}
      onMouseLeave={(e) => {
        if (!isSelected) e.currentTarget.style.background = "transparent";
      }}
    >
      <span
        style={{
          width: 7,
          height: 7,
          borderRadius: "50%",
          background: statusColor(cycle.status),
          flexShrink: 0,
          boxShadow: cycle.status === "running" ? `0 0 8px ${statusColor("running")}80` : "none",
          animation: cycle.status === "running" ? "glow-ring 2s ease-in-out infinite" : "none",
        }}
      />
      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
          <span style={{ fontSize: 12, color: isSelected ? "var(--accent)" : "var(--text-primary)", fontWeight: isSelected ? 500 : 400 }}>
            {cycle.trigger}
          </span>
          {cycle.dry_run && (
            <span className="pill" style={{ fontSize: 8, padding: "1px 5px" }}>DRY</span>
          )}
        </div>
        <div className="mono" style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 1 }}>
          {formatTime(cycle.started_at)}
        </div>
      </div>
      <span className="mono tabular-nums" style={{ fontSize: 10, color: "var(--text-muted)" }}>
        {formatDuration(cycle.total_duration_ms)}
      </span>
    </button>
  );
}

export function ConsolidationPanel() {
  const cycles = useEngramStore((s) => s.cycles);
  const isLoadingCycles = useEngramStore((s) => s.isLoadingCycles);
  const selectedCycleId = useEngramStore((s) => s.selectedCycleId);
  const cycleDetail = useEngramStore((s) => s.cycleDetail);
  const isLoadingDetail = useEngramStore((s) => s.isLoadingDetail);
  const isRunning = useEngramStore((s) => s.isRunning);
  const schedulerActive = useEngramStore((s) => s.schedulerActive);
  const pressure = useEngramStore((s) => s.pressure);
  const loadStatus = useEngramStore((s) => s.loadStatus);
  const loadCycles = useEngramStore((s) => s.loadCycles);
  const selectCycle = useEngramStore((s) => s.selectCycle);
  const triggerCycle = useEngramStore((s) => s.triggerCycle);
  const triggerDryRun = useEngramStore((s) => s.triggerDryRun);
  const setTriggerDryRun = useEngramStore((s) => s.setTriggerDryRun);

  useEffect(() => {
    loadStatus();
    loadCycles();
  }, [loadStatus, loadCycles]);

  return (
    <div
      className="animate-fade-in"
      style={{
        height: "100%",
        display: "flex",
        gap: 10,
        padding: "10px 14px",
        overflow: "hidden",
      }}
    >
      {/* Left column — Status + Cycle List */}
      <div
        className="card"
        style={{
          flex: "0 0 320px",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        {/* Status header */}
        <div
          style={{
            padding: "14px 16px 12px",
            borderBottom: "1px solid var(--border)",
          }}
        >
          <div className="label" style={{ marginBottom: 10 }}>
            Consolidation Status
          </div>

          {/* Running + Scheduler badges */}
          <div style={{ display: "flex", gap: 8, marginBottom: 10 }}>
            <span
              className="pill"
              style={
                isRunning
                  ? { borderColor: "var(--border-active)", background: "rgba(34, 211, 238, 0.06)", color: "var(--accent)" }
                  : {}
              }
            >
              <span
                style={{
                  width: 5,
                  height: 5,
                  borderRadius: "50%",
                  background: isRunning ? "var(--accent)" : "var(--text-muted)",
                  boxShadow: isRunning ? "0 0 6px var(--accent-glow-strong)" : "none",
                }}
              />
              {isRunning ? "RUNNING" : "IDLE"}
            </span>
            <span
              className="pill"
              style={
                schedulerActive
                  ? { borderColor: "rgba(52, 211, 153, 0.3)", background: "rgba(52, 211, 153, 0.06)", color: "#34d399" }
                  : {}
              }
            >
              {schedulerActive ? "Scheduler ON" : "Scheduler OFF"}
            </span>
          </div>

          {/* Pressure gauge */}
          {pressure && (
            <div style={{ marginBottom: 10 }}>
              <PressureGauge value={pressure.value} threshold={pressure.threshold} />
              <div className="mono" style={{ fontSize: 9, color: "var(--text-muted)", marginTop: 3 }}>
                {pressure.episodes_since_last} episodes, {pressure.entities_created} entities since last
              </div>
            </div>
          )}

          {/* Trigger controls */}
          <div style={{ display: "flex", gap: 6, alignItems: "center" }}>
            <button
              className="pill"
              onClick={() => triggerCycle(triggerDryRun)}
              disabled={isRunning}
              style={{
                borderColor: "var(--border-active)",
                background: isRunning ? "transparent" : "rgba(34, 211, 238, 0.08)",
                color: isRunning ? "var(--text-muted)" : "var(--accent)",
                cursor: isRunning ? "not-allowed" : "pointer",
                opacity: isRunning ? 0.5 : 1,
              }}
            >
              Trigger Cycle
            </button>
            <label
              style={{
                display: "flex",
                alignItems: "center",
                gap: 4,
                fontSize: 10,
                color: "var(--text-muted)",
                cursor: "pointer",
              }}
            >
              <input
                type="checkbox"
                checked={triggerDryRun}
                onChange={(e) => setTriggerDryRun(e.target.checked)}
                style={{ width: 12, height: 12, accentColor: "var(--accent)" }}
              />
              Dry run
            </label>
          </div>
        </div>

        {/* Cycle list */}
        <div style={{ flex: 1, overflowY: "auto", padding: "4px 0" }}>
          {isLoadingCycles && cycles.length === 0 ? (
            <div style={{ padding: 32, textAlign: "center" }}>
              <div className="skeleton" style={{ width: 120, height: 14, borderRadius: 4, margin: "0 auto" }} />
            </div>
          ) : cycles.length === 0 ? (
            <div style={{ padding: 32, textAlign: "center" }}>
              <span className="label">No cycles yet</span>
            </div>
          ) : (
            cycles.map((c) => (
              <CycleRow
                key={c.id}
                cycle={c}
                isSelected={selectedCycleId === c.id}
                onClick={() => selectCycle(c.id)}
              />
            ))
          )}
        </div>
      </div>

      {/* Right column — Cycle Detail */}
      <div
        className="card"
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        {!selectedCycleId ? (
          <div
            style={{
              flex: 1,
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexDirection: "column",
              gap: 12,
            }}
          >
            <div style={{ position: "relative", width: 64, height: 64 }}>
              <div
                style={{
                  position: "absolute",
                  inset: 0,
                  borderRadius: "50%",
                  border: "1px solid var(--border)",
                  animation: "glow-ring 4s ease-in-out infinite",
                }}
              />
              <div
                style={{
                  position: "absolute",
                  inset: 10,
                  borderRadius: "50%",
                  border: "1px solid var(--border-hover)",
                  animation: "glow-ring 4s ease-in-out infinite 0.5s",
                }}
              />
              <div
                style={{
                  position: "absolute",
                  top: "50%",
                  left: "50%",
                  transform: "translate(-50%, -50%)",
                  width: 8,
                  height: 8,
                  borderRadius: "50%",
                  background: "var(--accent-dim)",
                  boxShadow: "0 0 12px var(--accent-glow)",
                }}
              />
            </div>
            <span className="label">Select a cycle to view details</span>
          </div>
        ) : isLoadingDetail && !cycleDetail ? (
          <div style={{ flex: 1, display: "flex", alignItems: "center", justifyContent: "center" }}>
            <div className="skeleton" style={{ width: 140, height: 14, borderRadius: 4 }} />
          </div>
        ) : cycleDetail ? (
          <div style={{ display: "flex", flexDirection: "column", height: "100%", overflow: "hidden" }}>
            {/* Detail header */}
            <div style={{ padding: "14px 16px 12px", borderBottom: "1px solid var(--border)" }}>
              <div style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 4 }}>
                <span
                  style={{
                    width: 8,
                    height: 8,
                    borderRadius: "50%",
                    background: statusColor(cycleDetail.status),
                    boxShadow: `0 0 6px ${statusColor(cycleDetail.status)}60`,
                  }}
                />
                <span style={{ fontSize: 14, color: "var(--text-primary)", fontWeight: 500, textTransform: "capitalize" }}>
                  {cycleDetail.trigger} cycle
                </span>
                <span
                  className="pill"
                  style={{
                    fontSize: 9,
                    borderColor: statusColor(cycleDetail.status) + "40",
                    color: statusColor(cycleDetail.status),
                  }}
                >
                  {cycleDetail.status}
                </span>
                {cycleDetail.dry_run && (
                  <span className="pill" style={{ fontSize: 9, borderColor: "var(--border-hover)" }}>DRY RUN</span>
                )}
              </div>
              <div className="mono" style={{ fontSize: 10, color: "var(--text-muted)" }}>
                {formatTime(cycleDetail.started_at)} &middot; {formatDuration(cycleDetail.total_duration_ms)}
                {cycleDetail.error && (
                  <span style={{ color: "#f87171", marginLeft: 8 }}>{cycleDetail.error}</span>
                )}
              </div>
            </div>

            {/* Scrollable detail content */}
            <div style={{ flex: 1, overflowY: "auto", padding: "12px 16px" }}>
              {/* Phase timeline */}
              <div className="label" style={{ marginBottom: 8 }}>
                Phase Timeline
              </div>
              <PhaseTimeline phases={cycleDetail.phases} />

              {/* Audit sections */}
              {cycleDetail.merges.length > 0 && (
                <AuditSection title="Merges" count={cycleDetail.merges.length}>
                  {cycleDetail.merges.map((m) => (
                    <div key={m.id} style={{ display: "flex", gap: 6, alignItems: "center", padding: "3px 0" }}>
                      <span style={{ fontSize: 11, color: "#34d399" }}>{m.keep_name}</span>
                      <span style={{ fontSize: 9, color: "var(--text-muted)" }}>&larr;</span>
                      <span style={{ fontSize: 11, color: "#f87171", textDecoration: "line-through", opacity: 0.7 }}>{m.remove_name}</span>
                      <span className="mono tabular-nums" style={{ fontSize: 9, color: "var(--text-muted)", marginLeft: "auto" }}>
                        {(m.similarity * 100).toFixed(0)}%
                      </span>
                    </div>
                  ))}
                </AuditSection>
              )}

              {cycleDetail.inferred_edges.length > 0 && (
                <AuditSection title="Inferred Edges" count={cycleDetail.inferred_edges.length}>
                  {cycleDetail.inferred_edges.map((e) => (
                    <div key={e.id} style={{ display: "flex", gap: 6, alignItems: "center", padding: "3px 0" }}>
                      <span style={{ fontSize: 11, color: "var(--text-primary)" }}>{e.source_name}</span>
                      <span style={{ fontSize: 9, color: "var(--text-muted)" }}>&rarr;</span>
                      <span style={{ fontSize: 11, color: "var(--text-primary)" }}>{e.target_name}</span>
                      <span className="mono tabular-nums" style={{ fontSize: 9, color: "var(--text-muted)", marginLeft: "auto" }}>
                        {(e.confidence * 100).toFixed(0)}% ({e.co_occurrence_count}x)
                      </span>
                    </div>
                  ))}
                </AuditSection>
              )}

              {cycleDetail.prunes.length > 0 && (
                <AuditSection title="Pruned Entities" count={cycleDetail.prunes.length}>
                  {cycleDetail.prunes.map((p) => (
                    <div key={p.id} style={{ display: "flex", gap: 6, alignItems: "center", padding: "3px 0" }}>
                      <span style={{ fontSize: 11, color: "#f87171" }}>{p.entity_name}</span>
                      <span className="mono" style={{ fontSize: 9, color: "var(--text-muted)" }}>{p.entity_type}</span>
                      <span style={{ fontSize: 9, color: "var(--text-muted)", marginLeft: "auto" }}>{p.reason}</span>
                    </div>
                  ))}
                </AuditSection>
              )}

              {cycleDetail.replays?.length > 0 && (
                <AuditSection title="Episode Replays" count={cycleDetail.replays.length}>
                  {cycleDetail.replays.map((rp) => (
                    <div key={rp.id} style={{ display: "flex", gap: 6, alignItems: "center", padding: "3px 0" }}>
                      <span className="mono" style={{ fontSize: 10, color: "var(--text-secondary)" }}>
                        {rp.episode_id.slice(0, 12)}
                      </span>
                      {rp.skipped_reason ? (
                        <span style={{ fontSize: 9, color: "var(--text-muted)", fontStyle: "italic" }}>{rp.skipped_reason}</span>
                      ) : (
                        <>
                          <span style={{ fontSize: 9, color: "#34d399" }}>+{rp.new_entities_found} ent</span>
                          <span style={{ fontSize: 9, color: "#22d3ee" }}>+{rp.new_relationships_found} rel</span>
                          <span style={{ fontSize: 9, color: "var(--text-muted)" }}>{rp.entities_updated} updated</span>
                        </>
                      )}
                    </div>
                  ))}
                </AuditSection>
              )}

              {cycleDetail.reindexes?.length > 0 && (
                <AuditSection title="Reindexed Entities" count={cycleDetail.reindexes.length}>
                  {cycleDetail.reindexes.map((r) => (
                    <div key={r.id} style={{ display: "flex", gap: 6, alignItems: "center", padding: "3px 0" }}>
                      <span style={{ fontSize: 11, color: "var(--text-primary)" }}>{r.entity_name}</span>
                      <span className="mono" style={{ fontSize: 9, color: "var(--text-muted)", marginLeft: "auto" }}>
                        from {r.source_phase}
                      </span>
                    </div>
                  ))}
                </AuditSection>
              )}

              {cycleDetail.dreams.length > 0 && (
                <AuditSection title="Dream Boosts" count={cycleDetail.dreams.length}>
                  {cycleDetail.dreams.map((d) => (
                    <div key={d.id} style={{ display: "flex", gap: 6, alignItems: "center", padding: "3px 0" }}>
                      <span className="mono" style={{ fontSize: 10, color: "var(--text-secondary)" }}>
                        {d.source_entity_id.slice(0, 8)}
                      </span>
                      <span style={{ fontSize: 9, color: "var(--text-muted)" }}>&rarr;</span>
                      <span className="mono" style={{ fontSize: 10, color: "var(--text-secondary)" }}>
                        {d.target_entity_id.slice(0, 8)}
                      </span>
                      <span
                        className="mono tabular-nums"
                        style={{
                          fontSize: 10,
                          color: d.weight_delta > 0 ? "#34d399" : "#f87171",
                          marginLeft: "auto",
                        }}
                      >
                        {d.weight_delta > 0 ? "+" : ""}{d.weight_delta.toFixed(3)}
                      </span>
                    </div>
                  ))}
                </AuditSection>
              )}
            </div>
          </div>
        ) : null}
      </div>
    </div>
  );
}

function AuditSection({
  title,
  count,
  children,
}: {
  title: string;
  count: number;
  children: React.ReactNode;
}) {
  const [open, setOpen] = useState(true);
  return (
    <div style={{ marginTop: 16 }}>
      <button
        onClick={() => setOpen(!open)}
        style={{
          display: "flex",
          alignItems: "center",
          gap: 6,
          background: "none",
          border: "none",
          cursor: "pointer",
          padding: 0,
          marginBottom: 6,
        }}
      >
        <span style={{ fontSize: 10, color: "var(--text-muted)", transition: "transform 0.15s", transform: open ? "rotate(90deg)" : "rotate(0)" }}>
          &#9654;
        </span>
        <span className="label">{title}</span>
        <span className="mono tabular-nums" style={{ fontSize: 9, color: "var(--text-muted)" }}>
          ({count})
        </span>
      </button>
      {open && (
        <div style={{ paddingLeft: 16 }}>{children}</div>
      )}
    </div>
  );
}
