import { useEffect, useCallback } from "react";
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
} from "recharts";
import { useEngramStore } from "../store";
import { activationColor, activationGlow, entityColor } from "../lib/colors";
import { sendWsCommand } from "../hooks/useWebSocket";

export function ActivationMonitor() {
  const leaderboard = useEngramStore((s) => s.activationLeaderboard);
  const selectedEntity = useEngramStore((s) => s.selectedActivationEntity);
  const decayCurve = useEngramStore((s) => s.decayCurve);
  const decayFormula = useEngramStore((s) => s.decayFormula);
  const accessEvents = useEngramStore((s) => s.accessEvents);
  const isLoadingCurve = useEngramStore((s) => s.isLoadingCurve);
  const isSubscribed = useEngramStore((s) => s.isActivationSubscribed);
  const selectEntity = useEngramStore((s) => s.selectActivationEntity);
  const loadCurve = useEngramStore((s) => s.loadDecayCurve);
  const setIsSubscribed = useEngramStore((s) => s.setIsActivationSubscribed);
  const loadSnapshot = useEngramStore((s) => s.setActivationLeaderboard);

  useEffect(() => {
    import("../api/client").then(({ api }) => {
      api
        .getActivationSnapshot(50)
        .then((data) => loadSnapshot(data.topActivated))
        .catch(() => {});
    });
  }, [loadSnapshot]);

  const toggleSubscription = useCallback(() => {
    if (isSubscribed) {
      sendWsCommand({
        type: "command",
        command: "unsubscribe.activation_monitor",
      });
      setIsSubscribed(false);
    } else {
      sendWsCommand({
        type: "command",
        command: "subscribe.activation_monitor",
        interval_ms: 2000,
      });
      setIsSubscribed(true);
    }
  }, [isSubscribed, setIsSubscribed]);

  useEffect(() => {
    return () => {
      if (isSubscribed) {
        sendWsCommand({
          type: "command",
          command: "unsubscribe.activation_monitor",
        });
      }
    };
  }, [isSubscribed]);

  const handleEntityClick = useCallback(
    (entityId: string) => {
      selectEntity(entityId);
      loadCurve(entityId);
    },
    [selectEntity, loadCurve]
  );

  const maxActivation =
    leaderboard.length > 0
      ? Math.max(...leaderboard.map((i) => i.currentActivation), 0.01)
      : 1;

  const chartData = decayCurve.map((p) => {
    const d = new Date(p.timestamp);
    const hoursAgo = (Date.now() - d.getTime()) / 3600000;
    return {
      time:
        hoursAgo > 1
          ? `${hoursAgo.toFixed(0)}h ago`
          : `${(hoursAgo * 60).toFixed(0)}m ago`,
      activation: p.activation,
    };
  });

  const selectedItem = leaderboard.find(
    (i) => i.entityId === selectedEntity
  );

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
      {/* Left — Leaderboard */}
      <div
        className="card"
        style={{
          flex: "0 0 320px",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        {/* Header */}
        <div
          style={{
            padding: "14px 16px 10px",
            borderBottom: "1px solid var(--border)",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
            }}
          >
            <div className="label">Activation Leaderboard</div>
            <button
              onClick={toggleSubscription}
              className="pill"
              style={
                isSubscribed
                  ? {
                      borderColor: "var(--border-active)",
                      background: "rgba(34, 211, 238, 0.06)",
                      color: "var(--accent)",
                    }
                  : {}
              }
            >
              <span
                style={{
                  width: 5,
                  height: 5,
                  borderRadius: "50%",
                  background: isSubscribed
                    ? "var(--accent)"
                    : "var(--text-muted)",
                  boxShadow: isSubscribed
                    ? "0 0 6px var(--accent-glow-strong)"
                    : "none",
                }}
              />
              {isSubscribed ? "LIVE" : "PAUSED"}
            </button>
          </div>
          <div
            className="mono"
            style={{ fontSize: 10, color: "var(--text-muted)", marginTop: 3 }}
          >
            Top {leaderboard.length} entities
          </div>
        </div>

        {/* List */}
        <div style={{ flex: 1, overflowY: "auto", padding: "4px 0" }}>
          {leaderboard.length === 0 ? (
            <div
              style={{
                padding: 32,
                textAlign: "center",
              }}
            >
              <span className="label">No activated entities</span>
            </div>
          ) : (
            leaderboard.map((item, idx) => {
              const isSelected = selectedEntity === item.entityId;
              const typeColor = entityColor(item.entityType);
              const barWidth = (item.currentActivation / maxActivation) * 100;
              const actColor = activationColor(item.currentActivation);

              return (
                <button
                  key={item.entityId}
                  onClick={() => handleEntityClick(item.entityId)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    width: "100%",
                    padding: "5px 14px",
                    border: "none",
                    background: isSelected
                      ? "rgba(34, 211, 238, 0.06)"
                      : "transparent",
                    cursor: "pointer",
                    transition: "background 0.12s ease",
                    textAlign: "left",
                    position: "relative",
                  }}
                  onMouseEnter={(e) => {
                    if (!isSelected)
                      e.currentTarget.style.background =
                        "rgba(255,255,255,0.02)";
                  }}
                  onMouseLeave={(e) => {
                    if (!isSelected)
                      e.currentTarget.style.background = "transparent";
                  }}
                >
                  {/* Rank */}
                  <span
                    className="mono tabular-nums"
                    style={{
                      fontSize: 9,
                      color:
                        idx < 3
                          ? "var(--text-secondary)"
                          : "var(--text-muted)",
                      width: 16,
                      textAlign: "right",
                      flexShrink: 0,
                    }}
                  >
                    {idx + 1}
                  </span>
                  {/* Type dot */}
                  <span
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: "50%",
                      background: typeColor,
                      flexShrink: 0,
                      boxShadow: isSelected
                        ? `0 0 6px ${typeColor}60`
                        : "none",
                    }}
                  />
                  {/* Name */}
                  <span
                    style={{
                      flex: 1,
                      fontSize: 12,
                      color: isSelected
                        ? "var(--accent)"
                        : "var(--text-primary)",
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      fontWeight: isSelected ? 500 : 400,
                    }}
                  >
                    {item.name}
                  </span>
                  {/* Bar + value */}
                  <div
                    style={{
                      width: 76,
                      display: "flex",
                      alignItems: "center",
                      gap: 6,
                    }}
                  >
                    <div className="metric-bar" style={{ flex: 1 }}>
                      <div
                        className="metric-bar-fill"
                        style={{
                          width: `${barWidth}%`,
                          background: `linear-gradient(90deg, ${actColor}cc, ${actColor})`,
                          boxShadow: `0 0 4px ${activationGlow(item.currentActivation, 0.2)}`,
                        }}
                      />
                    </div>
                    <span
                      className="mono tabular-nums"
                      style={{
                        fontSize: 10,
                        color: actColor,
                        minWidth: 26,
                        textAlign: "right",
                        fontWeight: 500,
                      }}
                    >
                      {item.currentActivation.toFixed(2)}
                    </span>
                  </div>
                </button>
              );
            })
          )}
        </div>
      </div>

      {/* Right — Decay Curve */}
      <div
        className="card"
        style={{
          flex: 1,
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
        }}
      >
        {!selectedEntity ? (
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
            {/* Decorative rings */}
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
            <span className="label">Select an entity to view decay curve</span>
          </div>
        ) : (
          <>
            {/* Header */}
            <div
              style={{
                padding: "14px 16px 10px",
                borderBottom: "1px solid var(--border)",
              }}
            >
              <div
                style={{
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "space-between",
                }}
              >
                <div className="label">Activation Decay Curve</div>
                {selectedItem && (
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
                        background: entityColor(selectedItem.entityType),
                      }}
                    />
                    <span
                      style={{
                        fontSize: 12,
                        color: "var(--text-primary)",
                        fontWeight: 500,
                      }}
                    >
                      {selectedItem.name}
                    </span>
                    <span
                      className="mono tabular-nums"
                      style={{
                        fontSize: 11,
                        color: activationColor(
                          selectedItem.currentActivation
                        ),
                        fontWeight: 500,
                      }}
                    >
                      {selectedItem.currentActivation.toFixed(3)}
                    </span>
                  </div>
                )}
              </div>
              {decayFormula && (
                <div
                  className="mono"
                  style={{
                    fontSize: 10,
                    color: "var(--text-muted)",
                    marginTop: 3,
                  }}
                >
                  {decayFormula}
                </div>
              )}
            </div>

            {/* Chart */}
            <div style={{ flex: 1, padding: "12px 8px 4px 0" }}>
              {isLoadingCurve ? (
                <div
                  style={{
                    height: "100%",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                  }}
                >
                  <div
                    className="skeleton"
                    style={{ width: 140, height: 14 }}
                  />
                </div>
              ) : (
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartData}>
                    <defs>
                      <linearGradient
                        id="grad-decay"
                        x1="0"
                        y1="0"
                        x2="0"
                        y2="1"
                      >
                        <stop
                          offset="0%"
                          stopColor="#22d3ee"
                          stopOpacity={0.2}
                        />
                        <stop
                          offset="100%"
                          stopColor="#22d3ee"
                          stopOpacity={0}
                        />
                      </linearGradient>
                    </defs>
                    <XAxis
                      dataKey="time"
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
                      domain={[0, 1]}
                      stroke="var(--text-ghost)"
                      tick={{
                        fontSize: 9,
                        fontFamily: "var(--font-mono)",
                        fill: "var(--text-muted)",
                      }}
                      tickLine={false}
                      axisLine={false}
                      tickFormatter={(v: number) => v.toFixed(1)}
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
                      cursor={{
                        stroke: "var(--text-ghost)",
                        strokeWidth: 1,
                      }}
                      formatter={(value: number | undefined) => [
                        value !== undefined ? value.toFixed(4) : "0",
                        "Activation",
                      ]}
                    />
                    {accessEvents.map((evt) => {
                      const evtDate = new Date(evt);
                      const hoursAgo =
                        (Date.now() - evtDate.getTime()) / 3600000;
                      const label =
                        hoursAgo > 1
                          ? `${hoursAgo.toFixed(0)}h ago`
                          : `${(hoursAgo * 60).toFixed(0)}m ago`;
                      return (
                        <ReferenceLine
                          key={evt}
                          x={label}
                          stroke="var(--warm)"
                          strokeDasharray="2 4"
                          strokeOpacity={0.3}
                          strokeWidth={1}
                        />
                      );
                    })}
                    <Area
                      type="monotone"
                      dataKey="activation"
                      stroke="#22d3ee"
                      fill="url(#grad-decay)"
                      strokeWidth={1.5}
                      dot={false}
                      activeDot={{
                        r: 3,
                        fill: "#22d3ee",
                        stroke: "#22d3ee",
                        strokeWidth: 4,
                        strokeOpacity: 0.2,
                      }}
                    />
                  </AreaChart>
                </ResponsiveContainer>
              )}
            </div>
          </>
        )}
      </div>
    </div>
  );
}
