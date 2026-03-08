import { useEffect, useState } from "react";
import { api, type ServerHealthStatus } from "../api/client";
import { useEngramStore } from "../store";
import type { WsReadyState } from "../store/types";

type IndicatorState =
  | "checking"
  | "live"
  | "syncing"
  | "server_ok"
  | "degraded"
  | "offline";

const STATE_CONFIG: Record<
  IndicatorState,
  { dotColor: string; glow: string; label: string; pulse: boolean; textColor: string }
> = {
  checking: {
    dotColor: "var(--text-muted)",
    glow: "rgba(148, 163, 184, 0.25)",
    label: "Checking",
    pulse: true,
    textColor: "var(--text-muted)",
  },
  live: {
    dotColor: "var(--accent)",
    glow: "var(--accent-glow-strong)",
    label: "Live",
    pulse: false,
    textColor: "var(--text-muted)",
  },
  syncing: {
    dotColor: "var(--warm)",
    glow: "rgba(251, 146, 60, 0.4)",
    label: "Syncing",
    pulse: true,
    textColor: "var(--warm)",
  },
  server_ok: {
    dotColor: "var(--accent)",
    glow: "var(--accent-glow-strong)",
    label: "Server OK",
    pulse: false,
    textColor: "var(--accent)",
  },
  degraded: {
    dotColor: "var(--warm)",
    glow: "rgba(251, 146, 60, 0.4)",
    label: "Degraded",
    pulse: false,
    textColor: "var(--warm)",
  },
  offline: {
    dotColor: "#ef4444",
    glow: "rgba(239, 68, 68, 0.4)",
    label: "Offline",
    pulse: false,
    textColor: "#ef4444",
  },
};

const HEALTH_POLL_INTERVAL_MS = 15_000;

function getIndicatorState(
  readyState: WsReadyState,
  healthStatus: ServerHealthStatus | "checking",
): IndicatorState {
  if (healthStatus === "healthy") {
    if (readyState === "connected") return "live";
    if (readyState === "connecting" || readyState === "reconnecting") return "syncing";
    return "server_ok";
  }
  if (healthStatus === "degraded") return "degraded";
  if (healthStatus === "unhealthy") return "offline";
  if (readyState === "connected") return "live";
  return "checking";
}

function getIndicatorTitle(state: IndicatorState): string {
  switch (state) {
    case "live":
      return "Server healthy. Live WebSocket updates are connected.";
    case "syncing":
      return "Server healthy. Live WebSocket updates are reconnecting.";
    case "server_ok":
      return "Server healthy. Live WebSocket updates are disconnected.";
    case "degraded":
      return "Server reachable, but one or more backend services are degraded.";
    case "offline":
      return "Health checks failed. The Engram server appears offline.";
    case "checking":
    default:
      return "Checking Engram server health.";
  }
}

export function ConnectionStatus() {
  const readyState = useEngramStore((s) => s.readyState);
  const [healthStatus, setHealthStatus] = useState<ServerHealthStatus | "checking">("checking");

  useEffect(() => {
    let isMounted = true;

    async function loadHealth() {
      try {
        const health = await api.getHealth();
        if (isMounted) {
          setHealthStatus(health.status);
        }
      } catch {
        if (isMounted) {
          setHealthStatus("unhealthy");
        }
      }
    }

    void loadHealth();
    const intervalId = window.setInterval(() => {
      void loadHealth();
    }, HEALTH_POLL_INTERVAL_MS);

    return () => {
      isMounted = false;
      window.clearInterval(intervalId);
    };
  }, []);

  const indicatorState = getIndicatorState(readyState, healthStatus);
  const config = STATE_CONFIG[indicatorState];

  return (
    <div
      title={getIndicatorTitle(indicatorState)}
      style={{
        display: "flex",
        alignItems: "center",
        gap: 7,
        padding: "8px 14px",
      }}
    >
      <span
        className={config.pulse ? "animate-pulse-soft" : undefined}
        style={{
          width: 5,
          height: 5,
          borderRadius: "50%",
          background: config.dotColor,
          boxShadow: `0 0 6px ${config.glow}`,
          transition: "all 0.3s ease",
        }}
      />
      <span
        className="mono"
        style={{
          fontSize: 9,
          letterSpacing: "0.06em",
          textTransform: "uppercase",
          color: config.textColor,
          transition: "color 0.3s ease",
        }}
      >
        {config.label}
      </span>
    </div>
  );
}
