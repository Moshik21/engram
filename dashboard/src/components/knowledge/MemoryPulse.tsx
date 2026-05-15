import { useEngramStore } from "../../store";
import { entityColor, entityColorDim, activationColor } from "../../lib/colors";

export function MemoryPulse() {
  const pulseEntities = useEngramStore((s) => s.pulseEntities);
  const isPulseLoading = useEngramStore((s) => s.isPulseLoading);
  const openDrawer = useEngramStore((s) => s.openDrawer);
  const lifecycleDrilldownStage = useEngramStore((s) => s.lifecycleDrilldownStage);
  const isRecallDrilldown = lifecycleDrilldownStage === "recall";

  if (!isPulseLoading && pulseEntities.length === 0 && !isRecallDrilldown) return null;

  return (
    <div
      className="animate-fade-in"
      data-lifecycle-focus={isRecallDrilldown ? "true" : undefined}
      style={{
        flexShrink: 0,
        borderBottom: `1px solid ${isRecallDrilldown ? "#34d39966" : "var(--border)"}`,
        padding: "8px 16px",
        display: "flex",
        alignItems: "center",
        gap: 10,
        overflow: "hidden",
        background: isRecallDrilldown
          ? "linear-gradient(90deg, rgba(52, 211, 153, 0.08), transparent)"
          : undefined,
      }}
    >
      <span
        className="label"
        style={{
          fontSize: 9,
          letterSpacing: "0.1em",
          flexShrink: 0,
          color: isRecallDrilldown ? "#34d399" : undefined,
        }}
      >
        {isRecallDrilldown ? "Recall Context" : "PULSE"}
      </span>

      <div
        style={{
          display: "flex",
          gap: 6,
          overflow: "auto",
          flex: 1,
          scrollbarWidth: "none",
        }}
      >
        {isPulseLoading ? (
          Array.from({ length: 3 }).map((_, i) => (
            <div
              key={i}
              className="skeleton"
              style={{
                width: 100,
                height: 26,
                borderRadius: 99,
                flexShrink: 0,
              }}
            />
          ))
        ) : pulseEntities.length === 0 ? (
          <span
            style={{
              fontSize: 11,
              color: "var(--text-muted)",
            }}
          >
            No active entities loaded
          </span>
        ) : (
          pulseEntities.map((entity) => {
              const color = entityColor(entity.entityType);
              const pct = Math.min(entity.currentActivation * 100, 100);
              return (
                <button
                  key={entity.entityId}
                  onClick={() => openDrawer(entity.entityId)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 6,
                    padding: "4px 10px 4px 8px",
                    background: entityColorDim(entity.entityType, 0.06),
                    border: `1px solid ${color}18`,
                    borderRadius: 99,
                    cursor: "pointer",
                    flexShrink: 0,
                    transition: "all 0.15s ease",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.borderColor = `${color}35`;
                    e.currentTarget.style.background = entityColorDim(entity.entityType, 0.12);
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.borderColor = `${color}18`;
                    e.currentTarget.style.background = entityColorDim(entity.entityType, 0.06);
                  }}
                >
                  <span
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: "50%",
                      background: color,
                      boxShadow: `0 0 4px ${color}40`,
                      flexShrink: 0,
                    }}
                  />
                  <span
                    style={{
                      fontSize: 11,
                      color: "var(--text-primary)",
                      maxWidth: 100,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                      fontFamily: "var(--font-body)",
                    }}
                  >
                    {entity.name}
                  </span>
                  {/* Micro activation bar */}
                  <div
                    style={{
                      width: 24,
                      height: 3,
                      borderRadius: 2,
                      background: "rgba(255,255,255,0.04)",
                      overflow: "hidden",
                      flexShrink: 0,
                    }}
                  >
                    <div
                      style={{
                        width: `${pct}%`,
                        height: "100%",
                        borderRadius: 2,
                        background: activationColor(entity.currentActivation),
                      }}
                    />
                  </div>
                </button>
              );
            })
        )}
      </div>
    </div>
  );
}
