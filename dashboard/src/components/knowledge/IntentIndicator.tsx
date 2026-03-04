import { useEngramStore } from "../../store";
import type { IntentMode } from "../../store/types";

const INTENT_CONFIG: Record<Exclude<IntentMode, null>, { label: string; color: string }> = {
  asking: { label: "Recalling...", color: "var(--accent)" },
  remembering: { label: "Remembering...", color: "#34d399" },
  observing: { label: "Observing...", color: "#818cf8" },
  forgetting: { label: "Forgetting...", color: "#f87171" },
};

export function IntentIndicator() {
  const intentMode = useEngramStore((s) => s.intentMode);

  if (!intentMode) return null;

  const config = INTENT_CONFIG[intentMode];

  return (
    <div
      className="animate-slide-up"
      style={{
        display: "flex",
        alignItems: "center",
        gap: 6,
        padding: "4px 14px",
      }}
    >
      <span
        style={{
          width: 5,
          height: 5,
          borderRadius: "50%",
          background: config.color,
          boxShadow: `0 0 6px ${config.color}`,
          animation: "pulse-soft 1.5s ease-in-out infinite",
          flexShrink: 0,
        }}
      />
      <span
        style={{
          fontSize: 10,
          color: config.color,
          fontFamily: "var(--font-mono)",
          fontWeight: 500,
          letterSpacing: "0.04em",
        }}
      >
        {config.label}
      </span>
    </div>
  );
}
