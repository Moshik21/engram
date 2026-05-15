import { useState, useCallback } from "react";
import { api } from "../api/client";

interface FeedbackButtonsProps {
  entityId: string;
  size?: "sm" | "md";
}

export function FeedbackButtons({ entityId, size = "sm" }: FeedbackButtonsProps) {
  const [state, setState] = useState<"idle" | "loading" | "done">("idle");
  const [lastRating, setLastRating] = useState<number | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const handleFeedback = useCallback(async (rating: number) => {
    setState("loading");
    try {
      await api.submitFeedback({ entityId, rating });
      setLastRating(rating);
      setState("done");
      const msg = rating >= 4
        ? "Noted! Future recalls will prioritize this."
        : "Noted! Future recalls will deprioritize this.";
      setToast(msg);
      setTimeout(() => {
        setState("idle");
        setToast(null);
      }, 2500);
    } catch {
      setState("idle");
    }
  }, [entityId]);

  const btnSize = size === "sm" ? 24 : 30;
  const fontSize = size === "sm" ? 11 : 14;

  return (
    <span style={{ display: "inline-flex", gap: 3, alignItems: "center", position: "relative" }}>
      <button
        onClick={() => handleFeedback(5)}
        disabled={state === "loading"}
        title="Prefer this entity"
        style={{
          width: btnSize, height: btnSize,
          border: `1px solid ${lastRating === 5 ? "rgba(52, 211, 153, 0.4)" : "var(--border)"}`,
          borderRadius: "var(--radius-xs)",
          background: lastRating === 5 ? "rgba(52, 211, 153, 0.12)" : "transparent",
          color: lastRating === 5 ? "var(--success)" : "var(--text-muted)",
          cursor: state === "loading" ? "wait" : "pointer",
          fontSize, display: "flex", alignItems: "center", justifyContent: "center",
          transition: "all 0.2s ease", fontFamily: "var(--font-body)",
          boxShadow: lastRating === 5 ? "0 0 8px rgba(52, 211, 153, 0.2)" : "none",
        }}
        onMouseEnter={(e) => {
          if (lastRating !== 5) e.currentTarget.style.borderColor = "rgba(52, 211, 153, 0.3)";
        }}
        onMouseLeave={(e) => {
          if (lastRating !== 5) e.currentTarget.style.borderColor = "var(--border)";
        }}
      >
        {"\u25B2"}
      </button>
      <button
        onClick={() => handleFeedback(1)}
        disabled={state === "loading"}
        title="Avoid this entity"
        style={{
          width: btnSize, height: btnSize,
          border: `1px solid ${lastRating === 1 ? "rgba(248, 113, 113, 0.4)" : "var(--border)"}`,
          borderRadius: "var(--radius-xs)",
          background: lastRating === 1 ? "rgba(248, 113, 113, 0.12)" : "transparent",
          color: lastRating === 1 ? "var(--danger)" : "var(--text-muted)",
          cursor: state === "loading" ? "wait" : "pointer",
          fontSize, display: "flex", alignItems: "center", justifyContent: "center",
          transition: "all 0.2s ease", fontFamily: "var(--font-body)",
          boxShadow: lastRating === 1 ? "0 0 8px rgba(248, 113, 113, 0.2)" : "none",
        }}
        onMouseEnter={(e) => {
          if (lastRating !== 1) e.currentTarget.style.borderColor = "rgba(248, 113, 113, 0.3)";
        }}
        onMouseLeave={(e) => {
          if (lastRating !== 1) e.currentTarget.style.borderColor = "var(--border)";
        }}
      >
        {"\u25BC"}
      </button>

      {/* Toast */}
      {toast && (
        <div
          className="animate-slide-up"
          style={{
            position: "absolute", bottom: "calc(100% + 6px)", left: "50%",
            transform: "translateX(-50%)", whiteSpace: "nowrap",
            padding: "4px 10px", borderRadius: "var(--radius-sm)",
            background: "var(--surface-solid)", border: "1px solid var(--border-hover)",
            color: "var(--text-secondary)", fontSize: 10, fontFamily: "var(--font-mono)",
            boxShadow: "var(--shadow-elevated)", zIndex: 50,
            pointerEvents: "none",
          }}
        >
          {toast}
        </div>
      )}
    </span>
  );
}
