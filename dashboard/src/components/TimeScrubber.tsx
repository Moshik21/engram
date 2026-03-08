import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useEngramStore } from "../store";

/**
 * Extracts only the min createdAt timestamp from nodes.
 * Only re-renders when node set changes (structural), not on activation updates.
 */
function computeMinTimestamp() {
  const nodes = useEngramStore.getState().nodes;
  const timestamps = Object.values(nodes)
    .map((n) => new Date(n.createdAt).getTime())
    .filter((t) => !isNaN(t));
  return timestamps.length > 0 ? Math.min(...timestamps) : Date.now() - 86400000;
}

function useNodeTimeRange() {
  const [minTs, setMinTs] = useState(() => computeMinTimestamp());
  const [maxTs, setMaxTs] = useState(() => Date.now());

  useEffect(() => {
    // Only recompute when structure changes (keys)
    let prevKeys = Object.keys(useEngramStore.getState().nodes).sort().join(",");
    const unsub = useEngramStore.subscribe((state) => {
      const keys = Object.keys(state.nodes).sort().join(",");
      if (keys !== prevKeys) {
        prevKeys = keys;
        setMinTs(computeMinTimestamp());
        setMaxTs(Date.now());
      }
    });

    const intervalId = setInterval(() => {
      setMaxTs(Date.now());
    }, 60000);

    return () => {
      unsub();
      clearInterval(intervalId);
    };
  }, []);

  return useMemo(() => ({ min: minTs, max: maxTs }), [maxTs, minTs]);
}

export function TimeScrubber() {
  const brainMapScope = useEngramStore((s) => s.brainMapScope);
  const activeRegionId = useEngramStore((s) => s.activeRegionId);
  const atlasHistory = useEngramStore((s) => s.atlasHistory);
  const atlasSnapshotId = useEngramStore((s) => s.atlasSnapshotId);
  const timeRange = useNodeTimeRange();
  const timePosition = useEngramStore((s) => s.timePosition);
  const setTimePosition = useEngramStore((s) => s.setTimePosition);
  const setIsTimeScrubbing = useEngramStore((s) => s.setIsTimeScrubbing);
  const setAtlasSnapshotId = useEngramStore((s) => s.setAtlasSnapshotId);
  const loadAtlas = useEngramStore((s) => s.loadAtlas);
  const loadRegion = useEngramStore((s) => s.loadRegion);
  const loadGraphAt = useEngramStore((s) => s.loadGraphAt);
  const loadInitialGraph = useEngramStore((s) => s.loadInitialGraph);
  const timerRef = useRef<ReturnType<typeof setTimeout> | null>(null);

  const isAtlasScope =
    brainMapScope === "atlas" || brainMapScope === "region";
  const orderedAtlasHistory = useMemo(
    () => [...atlasHistory].reverse(),
    [atlasHistory],
  );
  const atlasIndex = useMemo(() => {
    if (orderedAtlasHistory.length === 0) return 0;
    if (!atlasSnapshotId) return orderedAtlasHistory.length - 1;
    return Math.max(
      0,
      orderedAtlasHistory.findIndex((entry) => entry.id === atlasSnapshotId),
    );
  }, [atlasSnapshotId, orderedAtlasHistory]);

  const currentValue = timePosition
    ? new Date(timePosition).getTime()
    : timeRange.max;

  const isLive = isAtlasScope ? atlasSnapshotId === null : timePosition === null;

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (isAtlasScope) return;
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
    [isAtlasScope, setTimePosition, setIsTimeScrubbing, loadGraphAt],
  );

  const handleAtlasChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      if (!isAtlasScope || orderedAtlasHistory.length === 0) return;
      const index = Number(e.target.value);
      const snapshot = orderedAtlasHistory[index];
      if (!snapshot) return;

      setAtlasSnapshotId(snapshot.id);
      setIsTimeScrubbing(true);

      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => {
        if (brainMapScope === "region" && activeRegionId) {
          void loadRegion(activeRegionId, { snapshotId: snapshot.id });
        } else {
          void loadAtlas({ snapshotId: snapshot.id });
        }
        setIsTimeScrubbing(false);
      }, 180);
    },
    [
      activeRegionId,
      brainMapScope,
      isAtlasScope,
      loadAtlas,
      loadRegion,
      orderedAtlasHistory,
      setAtlasSnapshotId,
      setIsTimeScrubbing,
    ],
  );

  const handleLive = useCallback(() => {
    setIsTimeScrubbing(false);
    if (timerRef.current) clearTimeout(timerRef.current);
    if (isAtlasScope) {
      setAtlasSnapshotId(null);
      if (brainMapScope === "region" && activeRegionId) {
        void loadRegion(activeRegionId);
      } else {
        void loadAtlas();
      }
      return;
    }
    setTimePosition(null);
    loadInitialGraph();
  }, [
    activeRegionId,
    brainMapScope,
    isAtlasScope,
    loadAtlas,
    loadInitialGraph,
    loadRegion,
    setAtlasSnapshotId,
    setIsTimeScrubbing,
    setTimePosition,
  ]);

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearTimeout(timerRef.current);
      }
    };
  }, []);

  const currentAtlasLabel = useMemo(() => {
    if (!isAtlasScope || orderedAtlasHistory.length === 0) {
      return null;
    }
    const currentEntry = orderedAtlasHistory[atlasIndex];
    if (!currentEntry) return null;
    return new Date(currentEntry.generatedAt).toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      hour: "numeric",
      minute: "2-digit",
    });
  }, [atlasIndex, isAtlasScope, orderedAtlasHistory]);

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
        min={isAtlasScope ? 0 : timeRange.min}
        max={
          isAtlasScope
            ? Math.max(orderedAtlasHistory.length - 1, 0)
            : timeRange.max
        }
        value={isAtlasScope ? atlasIndex : currentValue}
        onChange={isAtlasScope ? handleAtlasChange : handleChange}
        style={{
          width: 120,
          accentColor: "var(--accent)",
          cursor: "pointer",
        }}
        disabled={isAtlasScope && orderedAtlasHistory.length <= 1}
      />
      {isAtlasScope && currentAtlasLabel && (
        <span
          className="mono"
          style={{ fontSize: 10, color: "var(--text-muted)", minWidth: 88 }}
        >
          {currentAtlasLabel}
        </span>
      )}
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
