import { useEffect, useState } from "react";
import { api } from "../api/client";

type LoopStatus = Awaited<ReturnType<typeof api.getLoopStatus>>;

export function LoopStewardCard() {
  const [status, setStatus] = useState<LoopStatus | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const s = await api.getLoopStatus();
        if (!cancelled) {
          setStatus(s);
          setError(null);
        }
      } catch (err) {
        if (!cancelled) {
          setError(err instanceof Error ? err.message : "failed to load loop status");
          setStatus(null);
        }
      } finally {
        if (!cancelled) setLoading(false);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  if (loading) {
    return (
      <div className="card" style={{ padding: 16 }} data-testid="loop-steward-card">
        <div className="label" style={{ marginBottom: 8 }}>
          Loop Steward
        </div>
        <div className="skeleton" style={{ width: "50%", height: 12 }} />
      </div>
    );
  }

  if (error) {
    return (
      <div className="card" style={{ padding: 16 }} data-testid="loop-steward-card">
        <div className="label" style={{ marginBottom: 8 }}>
          Loop Steward
        </div>
        <span className="label" style={{ opacity: 0.7 }}>
          {error}
        </span>
      </div>
    );
  }

  const active = Boolean(status?.active);
  const adj = status?.adjustment;
  const regime = status?.regime ?? adj?.regime ?? "none";
  const reason = status?.reason ?? adj?.reason ?? "";
  const remaining = status?.remaining_ttl_seconds ?? 0;
  const expires = status?.expires_at ?? adj?.expires_at;
  const budgets = adj?.budgets ?? {};
  const boost = adj?.phase_boost ?? [];
  const defer = adj?.phase_defer ?? [];

  return (
    <div className="card" style={{ padding: 16 }} data-testid="loop-steward-card">
      <div
        style={{
          display: "flex",
          justifyContent: "space-between",
          alignItems: "baseline",
          marginBottom: 10,
        }}
      >
        <div className="label">Loop Steward</div>
        <span
          data-testid="loop-steward-active"
          style={{
            fontSize: 12,
            fontWeight: 600,
            color: active ? "#34d399" : "#94a3b8",
          }}
        >
          {active ? "active" : "none"}
        </span>
      </div>

      {!active ? (
        <p style={{ margin: 0, fontSize: 13, opacity: 0.75 }} data-testid="loop-steward-empty">
          No active adjustment. Harness can apply via{" "}
          <code>engram loop apply</code> / operator MCP.
        </p>
      ) : (
        <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
          <div data-testid="loop-steward-regime" style={{ fontSize: 14, fontWeight: 600 }}>
            regime: {regime}
          </div>
          {reason ? (
            <div data-testid="loop-steward-reason" style={{ fontSize: 12, opacity: 0.85 }}>
              {reason}
            </div>
          ) : null}
          <div style={{ fontSize: 12, opacity: 0.8 }} data-testid="loop-steward-ttl">
            TTL remaining: {Math.max(0, Math.round(remaining))}s
            {expires ? ` · expires ${expires}` : ""}
          </div>
          {Object.keys(budgets).length > 0 ? (
            <div style={{ fontSize: 12 }} data-testid="loop-steward-budgets">
              budgets:{" "}
              {Object.entries(budgets)
                .map(([k, v]) => `${k}=${v}`)
                .join(", ")}
            </div>
          ) : null}
          {(boost.length > 0 || defer.length > 0) && (
            <div style={{ fontSize: 12 }} data-testid="loop-steward-phases">
              {boost.length > 0 ? `boost: ${boost.join(", ")}` : null}
              {boost.length > 0 && defer.length > 0 ? " · " : null}
              {defer.length > 0 ? `defer: ${defer.join(", ")}` : null}
            </div>
          )}
          {adj?.created_by ? (
            <div style={{ fontSize: 11, opacity: 0.6 }}>by {adj.created_by}</div>
          ) : null}
        </div>
      )}
    </div>
  );
}
