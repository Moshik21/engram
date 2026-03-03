import { useCallback, useMemo, useRef } from "react";
import { useEngramStore } from "../store";

export function TimeScrubber() {
  const nodes = useEngramStore((s) => s.nodes);
  const timePosition = useEngramStore((s) => s.timePosition);
  const setTimePosition = useEngramStore((s) => s.setTimePosition);
  const setIsTimeScrubbing = useEngramStore((s) => s.setIsTimeScrubbing);
  const loadGraphAt = useEngramStore((s) => s.loadGraphAt);
  const loadInitialGraph = useEngramStore((s) => s.loadInitialGraph);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const timeRange = useMemo(() => {
    const timestamps = Object.values(nodes)
      .map((n) => new Date(n.createdAt).getTime())
      .filter((t) => !isNaN(t));
    if (timestamps.length === 0)
      return { min: Date.now() - 86400000, max: Date.now() };
    return {
      min: Math.min(...timestamps),
      max: Date.now(),
    };
  }, [nodes]);

  const currentValue = timePosition
    ? new Date(timePosition).getTime()
    : timeRange.max;

  const isLive = timePosition === null;

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const ts = Number(e.target.value);
      const iso = new Date(ts).toISOString();
      setTimePosition(iso);
      setIsTimeScrubbing(true);

      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => {
        loadGraphAt(iso);
        setIsTimeScrubbing(false);
      }, 300);
    },
    [setTimePosition, setIsTimeScrubbing, loadGraphAt],
  );

  const handleLive = useCallback(() => {
    setTimePosition(null);
    setIsTimeScrubbing(false);
    if (timerRef.current) clearTimeout(timerRef.current);
    loadInitialGraph();
  }, [setTimePosition, setIsTimeScrubbing, loadInitialGraph]);

  return (
    <div
      className="card"
      style={{
        borderRadius: "var(--radius-md)",
        padding: "4px 12px",
        display: "flex",
        alignItems: "center",
        gap: 8,
      }}
    >
      <input
        type="range"
        min={timeRange.min}
        max={timeRange.max}
        value={currentValue}
        onChange={handleChange}
        style={{
          width: 120,
          accentColor: "var(--accent)",
          cursor: "pointer",
        }}
      />
      <button
        onClick={handleLive}
        className={isLive ? "pill pill-active" : "pill"}
        style={{ fontSize: 9, fontWeight: 600, letterSpacing: "0.08em" }}
      >
        {isLive && (
          <span
            style={{
              width: 5,
              height: 5,
              borderRadius: "50%",
              background: "var(--accent)",
              boxShadow: "0 0 6px var(--accent-glow-strong)",
            }}
          />
        )}
        LIVE
      </button>
    </div>
  );
}
