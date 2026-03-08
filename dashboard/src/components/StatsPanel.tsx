import { useEffect } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import { useEngramStore } from "../store";
import { entityColor, entityColorDim } from "../lib/colors";

const EMPTY_CUE_METRICS = {
  cueCount: 0,
  episodesWithoutCues: 0,
  cueCoverage: 0,
  cueHitCount: 0,
  cueHitEpisodeCount: 0,
  cueHitEpisodeRate: 0,
  cueSurfacedCount: 0,
  cueSelectedCount: 0,
  cueUsedCount: 0,
  cueNearMissCount: 0,
  avgPolicyScore: 0,
  avgProjectionAttempts: 0,
  projectedCueCount: 0,
  cueToProjectionConversionRate: 0,
};

const EMPTY_PROJECTION_METRICS = {
  stateCounts: {
    queued: 0,
    cued: 0,
    cueOnly: 0,
    scheduled: 0,
    projecting: 0,
    projected: 0,
    failed: 0,
    deadLetter: 0,
  },
  attemptedEpisodeCount: 0,
  totalAttempts: 0,
  failureCount: 0,
  deadLetterCount: 0,
  failureRate: 0,
  avgProcessingDurationMs: 0,
  avgTimeToProjectionMs: 0,
  yield: {
    linkedEntityCount: 0,
    relationshipCount: 0,
    avgLinkedEntitiesPerProjectedEpisode: 0,
    avgRelationshipsPerProjectedEpisode: 0,
  },
};

function formatRate(value: number) {
  const percentage = value * 100;
  const rounded = Number(percentage.toFixed(1));
  const digits = Math.abs(rounded - Math.round(rounded)) < 0.05 ? 0 : 1;
  return `${percentage.toFixed(digits)}%`;
}

function formatNumber(value: number, digits = 2) {
  if (!Number.isFinite(value)) return "0";
  if (Math.abs(value - Math.round(value)) < 0.001) {
    return Math.round(value).toLocaleString();
  }
  return value.toFixed(digits);
}

function formatDuration(value: number) {
  if (!Number.isFinite(value) || value <= 0) return "0 ms";
  if (value >= 1000) return `${formatNumber(value / 1000)} s`;
  return `${formatNumber(value, value >= 100 ? 0 : 1)} ms`;
}

function MetricCard({
  label,
  value,
  accent,
}: {
  label: string;
  value: number;
  accent?: string;
}) {
  return (
    <div
      className="card card-glow"
      style={{
        padding: "18px 20px",
        position: "relative",
        overflow: "hidden",
        minWidth: 0,
      }}
    >
      <div
        style={{
          position: "absolute",
          top: 0,
          left: 16,
          right: 16,
          height: 1,
          background: accent
            ? `linear-gradient(90deg, transparent, ${accent}40, transparent)`
            : "linear-gradient(90deg, transparent, var(--accent-dim), transparent)",
          opacity: 0.6,
        }}
      />
      <div className="label" style={{ marginBottom: 8 }}>
        {label}
      </div>
      <div
        className="mono tabular-nums"
        style={{
          fontSize: 34,
          fontWeight: 300,
          color: "#fff",
          lineHeight: 1,
          letterSpacing: "-0.02em",
        }}
      >
        {value.toLocaleString()}
      </div>
    </div>
  );
}

function SectionCard({
  title,
  subtitle,
  children,
}: {
  title: string;
  subtitle: string;
  children: React.ReactNode;
}) {
  return (
    <div className="card" style={{ padding: 16, minWidth: 0 }}>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          gap: 12,
          marginBottom: 12,
          alignItems: "baseline",
          flexWrap: "wrap",
        }}
      >
        <div>
          <div className="label" style={{ marginBottom: 4 }}>
            {title}
          </div>
          <div
            style={{
              fontSize: 12,
              color: "var(--text-muted)",
            }}
          >
            {subtitle}
          </div>
        </div>
      </div>
      {children}
    </div>
  );
}

function MiniMetric({
  label,
  value,
  accent,
  helper,
}: {
  label: string;
  value: string;
  accent: string;
  helper?: string;
}) {
  return (
    <div
      style={{
        minWidth: 0,
        padding: "10px 12px",
        borderRadius: "var(--radius-sm)",
        border: `1px solid ${accent}22`,
        background: `linear-gradient(180deg, ${accent}14, transparent)`,
      }}
    >
      <div
        style={{
          fontSize: 10,
          color: "var(--text-muted)",
          marginBottom: 6,
          textTransform: "uppercase",
          letterSpacing: "0.08em",
        }}
      >
        {label}
      </div>
      <div
        className="mono tabular-nums"
        style={{
          fontSize: 20,
          color: "#fff",
          lineHeight: 1.05,
          marginBottom: helper ? 4 : 0,
        }}
      >
        {value}
      </div>
      {helper ? (
        <div
          style={{
            fontSize: 11,
            color: "var(--text-muted)",
          }}
        >
          {helper}
        </div>
      ) : null}
    </div>
  );
}

function RatioBar({
  label,
  value,
  accent,
  caption,
}: {
  label: string;
  value: number;
  accent: string;
  caption: string;
}) {
  const width = `${Math.max(0, Math.min(100, value * 100))}%`;
  return (
    <div>
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "center",
          marginBottom: 5,
          gap: 12,
        }}
      >
        <span
          style={{
            fontSize: 11,
            color: "var(--text-secondary)",
          }}
        >
          {label}
        </span>
        <span
          className="mono tabular-nums"
          style={{
            fontSize: 11,
            color: accent,
          }}
        >
          {caption}
        </span>
      </div>
      <div className="metric-bar">
        <div
          className="metric-bar-fill"
          style={{
            width,
            background: `linear-gradient(90deg, ${accent}, ${accent}88)`,
          }}
        />
      </div>
    </div>
  );
}

export function StatsPanel() {
  const stats = useEngramStore((s) => s.stats);
  const isLoading = useEngramStore((s) => s.isLoadingStats);
  const loadStats = useEngramStore((s) => s.loadStats);

  useEffect(() => {
    loadStats();
  }, [loadStats]);

  if (isLoading && !stats) {
    return (
      <div
        style={{
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          gap: 8,
        }}
      >
        <div
          className="skeleton"
          style={{ width: 120, height: 14, borderRadius: 4 }}
        />
      </div>
    );
  }

  if (!stats) {
    return (
      <div
        style={{
          height: "100%",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
        }}
      >
        <span className="label">No stats available</span>
      </div>
    );
  }

  const cueMetrics = stats.cueMetrics ?? EMPTY_CUE_METRICS;
  const projectionMetrics = stats.projectionMetrics ?? EMPTY_PROJECTION_METRICS;
  const typeEntries = Object.entries(stats.entityTypeCounts).sort(
    ([, a], [, b]) => b - a,
  );
  const maxTypeCount = typeEntries.length > 0 ? typeEntries[0][1] : 1;

  return (
    <div
      className="animate-fade-in"
      style={{
        height: "100%",
        overflowY: "auto",
        padding: "10px 14px 20px",
        display: "flex",
        flexDirection: "column",
        gap: 12,
      }}
    >
      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(160px, 1fr))",
          gap: 10,
        }}
      >
        <MetricCard
          label="Entities"
          value={stats.totalEntities}
          accent="#22d3ee"
        />
        <MetricCard
          label="Relationships"
          value={stats.totalRelationships}
          accent="#818cf8"
        />
        <MetricCard
          label="Episodes"
          value={stats.totalEpisodes}
          accent="#f97316"
        />
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(280px, 1fr))",
          gap: 10,
        }}
      >
        <SectionCard
          title="Cue Layer"
          subtitle="Immediate recallability before full projection"
        >
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
              gap: 8,
              marginBottom: 12,
            }}
          >
            <MiniMetric
              label="Coverage"
              value={formatRate(cueMetrics.cueCoverage)}
              accent="#22d3ee"
              helper={`${cueMetrics.cueCount} cued`}
            />
            <MiniMetric
              label="Conversion"
              value={formatRate(cueMetrics.cueToProjectionConversionRate)}
              accent="#818cf8"
              helper={`${cueMetrics.projectedCueCount} projected`}
            />
            <MiniMetric
              label="Cue Hits"
              value={cueMetrics.cueHitCount.toLocaleString()}
              accent="#f97316"
              helper={`${cueMetrics.cueHitEpisodeCount} episodes`}
            />
            <MiniMetric
              label="Used"
              value={cueMetrics.cueUsedCount.toLocaleString()}
              accent="#34d399"
              helper={`${cueMetrics.cueSelectedCount} selected`}
            />
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
              gap: 10,
            }}
          >
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 10,
              }}
            >
              <RatioBar
                label="Cue coverage"
                value={cueMetrics.cueCoverage}
                accent="#22d3ee"
                caption={`${cueMetrics.episodesWithoutCues} without cues`}
              />
              <RatioBar
                label="Episodes ever hit"
                value={cueMetrics.cueHitEpisodeRate}
                accent="#f97316"
                caption={formatRate(cueMetrics.cueHitEpisodeRate)}
              />
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
                gap: 8,
              }}
            >
              <MiniMetric
                label="Near misses"
                value={cueMetrics.cueNearMissCount.toLocaleString()}
                accent="#fb7185"
              />
              <MiniMetric
                label="Policy"
                value={formatNumber(cueMetrics.avgPolicyScore)}
                accent="#67e8f9"
              />
              <MiniMetric
                label="Surfaced"
                value={cueMetrics.cueSurfacedCount.toLocaleString()}
                accent="#c084fc"
              />
              <MiniMetric
                label="Avg attempts"
                value={formatNumber(cueMetrics.avgProjectionAttempts)}
                accent="#facc15"
              />
            </div>
          </div>
        </SectionCard>

        <SectionCard
          title="Projection Health"
          subtitle="Queue pressure, failure rate, and applied yield"
        >
          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(120px, 1fr))",
              gap: 8,
              marginBottom: 12,
            }}
          >
            <MiniMetric
              label="Projected"
              value={projectionMetrics.stateCounts.projected.toLocaleString()}
              accent="#22d3ee"
            />
            <MiniMetric
              label="Scheduled"
              value={projectionMetrics.stateCounts.scheduled.toLocaleString()}
              accent="#818cf8"
            />
            <MiniMetric
              label="Failed"
              value={projectionMetrics.stateCounts.failed.toLocaleString()}
              accent="#fb7185"
            />
            <MiniMetric
              label="Dead Letter"
              value={projectionMetrics.stateCounts.deadLetter.toLocaleString()}
              accent="#f97316"
            />
          </div>

          <div
            style={{
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(180px, 1fr))",
              gap: 10,
            }}
          >
            <div
              style={{
                display: "flex",
                flexDirection: "column",
                gap: 10,
              }}
            >
              <RatioBar
                label="Failure rate"
                value={projectionMetrics.failureRate}
                accent="#fb7185"
                caption={formatRate(projectionMetrics.failureRate)}
              />
              <RatioBar
                label="Projection queue pressure"
                value={
                  stats.totalEpisodes > 0
                    ? (projectionMetrics.stateCounts.scheduled
                        + projectionMetrics.stateCounts.projecting)
                      / stats.totalEpisodes
                    : 0
                }
                accent="#818cf8"
                caption={`${projectionMetrics.stateCounts.scheduled + projectionMetrics.stateCounts.projecting} active`}
              />
            </div>
            <div
              style={{
                display: "grid",
                gridTemplateColumns: "repeat(2, minmax(0, 1fr))",
                gap: 8,
              }}
            >
              <MiniMetric
                label="Attempts"
                value={projectionMetrics.totalAttempts.toLocaleString()}
                accent="#67e8f9"
                helper={`${projectionMetrics.attemptedEpisodeCount} attempted episodes`}
              />
              <MiniMetric
                label="Avg duration"
                value={formatDuration(projectionMetrics.avgProcessingDurationMs)}
                accent="#34d399"
                helper={
                  projectionMetrics.avgTimeToProjectionMs > 0
                    ? `to project ${formatDuration(projectionMetrics.avgTimeToProjectionMs)}`
                    : undefined
                }
              />
              <MiniMetric
                label="Entities / ep"
                value={formatNumber(
                  projectionMetrics.yield.avgLinkedEntitiesPerProjectedEpisode,
                )}
                accent="#facc15"
                helper={`${projectionMetrics.yield.linkedEntityCount} linked`}
              />
              <MiniMetric
                label="Rels / ep"
                value={formatNumber(
                  projectionMetrics.yield.avgRelationshipsPerProjectedEpisode,
                )}
                accent="#c084fc"
                helper={`${projectionMetrics.yield.relationshipCount} written`}
              />
            </div>
          </div>
        </SectionCard>
      </div>

      <div
        style={{
          display: "grid",
          gridTemplateColumns: "repeat(auto-fit, minmax(320px, 1fr))",
          gap: 10,
          flex: 1,
          minHeight: 0,
        }}
      >
        <div
          className="card"
          style={{ padding: "16px 16px 8px", minWidth: 0 }}
        >
          <div className="label" style={{ marginBottom: 14 }}>
            Growth Timeline
          </div>
          {stats.growthTimeline.length > 0 ? (
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={stats.growthTimeline}>
                <defs>
                  <linearGradient
                    id="grad-ent"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop
                      offset="0%"
                      stopColor="#22d3ee"
                      stopOpacity={0.25}
                    />
                    <stop
                      offset="100%"
                      stopColor="#22d3ee"
                      stopOpacity={0}
                    />
                  </linearGradient>
                  <linearGradient
                    id="grad-ep"
                    x1="0"
                    y1="0"
                    x2="0"
                    y2="1"
                  >
                    <stop
                      offset="0%"
                      stopColor="#f97316"
                      stopOpacity={0.2}
                    />
                    <stop
                      offset="100%"
                      stopColor="#f97316"
                      stopOpacity={0}
                    />
                  </linearGradient>
                </defs>
                <XAxis
                  dataKey="date"
                  stroke="var(--text-ghost)"
                  tick={{
                    fontSize: 9,
                    fontFamily: "var(--font-mono)",
                    fill: "var(--text-muted)",
                  }}
                  tickLine={false}
                  axisLine={false}
                  interval="preserveStartEnd"
                />
                <YAxis
                  stroke="var(--text-ghost)"
                  tick={{
                    fontSize: 9,
                    fontFamily: "var(--font-mono)",
                    fill: "var(--text-muted)",
                  }}
                  tickLine={false}
                  axisLine={false}
                  width={28}
                />
                <Tooltip
                  contentStyle={{
                    background: "var(--surface-solid)",
                    border: "1px solid var(--border-hover)",
                    borderRadius: "var(--radius-sm)",
                    fontFamily: "var(--font-mono)",
                    fontSize: 10,
                    color: "var(--text-primary)",
                    boxShadow: "var(--shadow-elevated)",
                    padding: "6px 10px",
                  }}
                  cursor={{ stroke: "var(--text-ghost)", strokeWidth: 1 }}
                />
                <Area
                  type="monotone"
                  dataKey="entities"
                  stroke="#22d3ee"
                  fill="url(#grad-ent)"
                  strokeWidth={1.5}
                  dot={false}
                  activeDot={{
                    r: 3,
                    fill: "#22d3ee",
                    stroke: "none",
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="episodes"
                  stroke="#f97316"
                  fill="url(#grad-ep)"
                  strokeWidth={1.5}
                  dot={false}
                  activeDot={{
                    r: 3,
                    fill: "#f97316",
                    stroke: "none",
                  }}
                />
              </AreaChart>
            </ResponsiveContainer>
          ) : (
            <div
              style={{
                height: 220,
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <span className="label">No timeline data</span>
            </div>
          )}
        </div>

        <div
          style={{
            display: "flex",
            flexDirection: "column",
            gap: 10,
            minWidth: 0,
          }}
        >
          {typeEntries.length > 0 && (
            <div className="card" style={{ padding: 16, flex: 1 }}>
              <div className="label" style={{ marginBottom: 12 }}>
                Entity Types
              </div>
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 7,
                }}
              >
                {typeEntries.map(([type, count]) => {
                  const color = entityColor(type);
                  const pct = (count / maxTypeCount) * 100;
                  return (
                    <div key={type}>
                      <div
                        style={{
                          display: "flex",
                          justifyContent: "space-between",
                          alignItems: "center",
                          marginBottom: 3,
                        }}
                      >
                        <div
                          style={{
                            display: "flex",
                            alignItems: "center",
                            gap: 6,
                          }}
                        >
                          <span
                            style={{
                              width: 6,
                              height: 6,
                              borderRadius: "50%",
                              background: color,
                              flexShrink: 0,
                            }}
                          />
                          <span
                            style={{
                              fontSize: 12,
                              color: "var(--text-secondary)",
                              textTransform: "capitalize",
                            }}
                          >
                            {type}
                          </span>
                        </div>
                        <span
                          className="mono tabular-nums"
                          style={{
                            fontSize: 11,
                            color: "var(--text-primary)",
                          }}
                        >
                          {count}
                        </span>
                      </div>
                      <div className="metric-bar">
                        <div
                          className="metric-bar-fill"
                          style={{
                            width: `${pct}%`,
                            background: `linear-gradient(90deg, ${color}, ${color}88)`,
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {stats.topConnected.length > 0 && (
            <div className="card" style={{ padding: 16, flex: 1 }}>
              <div className="label" style={{ marginBottom: 12 }}>
                Most Connected
              </div>
              <div
                style={{
                  display: "flex",
                  flexDirection: "column",
                  gap: 4,
                }}
              >
                {stats.topConnected.slice(0, 8).map((item, i) => {
                  const color = entityColor(item.entityType);
                  return (
                    <div
                      key={item.id}
                      style={{
                        display: "flex",
                        alignItems: "center",
                        gap: 8,
                        padding: "5px 8px",
                        borderRadius: "var(--radius-xs)",
                        background:
                          i === 0
                            ? entityColorDim(item.entityType, 0.06)
                            : "transparent",
                        transition: "background 0.15s ease",
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.background =
                          entityColorDim(item.entityType, 0.08);
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.background =
                          i === 0
                            ? entityColorDim(item.entityType, 0.06)
                            : "transparent";
                      }}
                    >
                      <span
                        className="mono tabular-nums"
                        style={{
                          fontSize: 9,
                          color: "var(--text-muted)",
                          width: 14,
                          textAlign: "right",
                        }}
                      >
                        {i + 1}
                      </span>
                      <span
                        style={{
                          width: 5,
                          height: 5,
                          borderRadius: "50%",
                          background: color,
                          flexShrink: 0,
                        }}
                      />
                      <span
                        style={{
                          flex: 1,
                          fontSize: 12,
                          color: "var(--text-primary)",
                          overflow: "hidden",
                          textOverflow: "ellipsis",
                          whiteSpace: "nowrap",
                        }}
                      >
                        {item.name}
                      </span>
                      <span
                        className="mono tabular-nums"
                        style={{
                          fontSize: 11,
                          color: color,
                          fontWeight: 500,
                        }}
                      >
                        {item.connectionCount}
                      </span>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
