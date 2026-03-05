import { useEngramStore } from "../store";

function TogglePill({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={active ? "pill pill-active" : "pill"}
    >
      {children}
    </button>
  );
}

export function GraphControls({
  onToggleStressTest,
  showStressTest,
}: {
  onToggleStressTest?: () => void;
  showStressTest?: boolean;
} = {}) {
  const renderMode = useEngramStore((s) => s.renderMode);
  const setRenderMode = useEngramStore((s) => s.setRenderMode);
  const showHeatmap = useEngramStore((s) => s.showActivationHeatmap);
  const toggleHeatmap = useEngramStore((s) => s.toggleActivationHeatmap);
  const showEdgeLabels = useEngramStore((s) => s.showEdgeLabels);
  const toggleEdgeLabels = useEngramStore((s) => s.toggleEdgeLabels);
  const showFpsOverlay = useEngramStore((s) => s.showFpsOverlay);
  const toggleFpsOverlay = useEngramStore((s) => s.toggleFpsOverlay);

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 5 }}>
      <TogglePill
        active={renderMode === "3d"}
        onClick={() => setRenderMode(renderMode === "3d" ? "2d" : "3d")}
      >
        {renderMode === "3d" ? "3D" : "2D"}
      </TogglePill>
      <TogglePill active={showHeatmap} onClick={toggleHeatmap}>
        Heatmap
      </TogglePill>
      <TogglePill active={showEdgeLabels} onClick={toggleEdgeLabels}>
        Labels
      </TogglePill>
      <TogglePill active={showFpsOverlay} onClick={toggleFpsOverlay}>
        FPS
      </TogglePill>
      {import.meta.env.DEV && onToggleStressTest && (
        <TogglePill active={!!showStressTest} onClick={onToggleStressTest}>
          Stress
        </TogglePill>
      )}
    </div>
  );
}
