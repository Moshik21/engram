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
        flex: 1,
        minWidth: 140,
        position: "relative",
        overflow: "hidden",
      }}
    >
      {/* Subtle top accent line */}
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

  const typeEntries = Object.entries(stats.entityTypeCounts).sort(
    ([, a], [, b]) => b - a
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
      {/* Metric cards */}
      <div style={{ display: "flex", gap: 10 }}>
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

      {/* Two-column layout for chart + types */}
      <div style={{ display: "flex", gap: 10, flex: 1, minHeight: 0 }}>
        {/* Growth chart */}
        <div
          className="card"
          style={{ flex: 1.4, padding: "16px 16px 8px", minWidth: 0 }}
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

        {/* Right column: Types + Top Connected */}
        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            gap: 10,
            minWidth: 0,
          }}
        >
          {/* Entity type distribution */}
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

          {/* Top connected leaderboard */}
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
