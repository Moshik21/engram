import type { CameraBookmark } from "./useNavigationHistory";

interface Props {
  items: Array<CameraBookmark & { index: number }>;
  currentIndex: number;
  truncatedLeft: boolean;
  truncatedRight: boolean;
  onNavigate: (index: number) => void;
  nodeNames: Map<string, string>;
}

export function NavigationBreadcrumbs({
  items,
  currentIndex,
  truncatedLeft,
  truncatedRight,
  onNavigate,
  nodeNames,
}: Props) {
  if (items.length === 0) return null;

  return (
    <div
      style={{
        position: "absolute",
        bottom: 40,
        left: "50%",
        transform: "translateX(-50%)",
        zIndex: 40,
        display: "flex",
        alignItems: "center",
        gap: 4,
        padding: "3px 6px",
        borderRadius: 8,
        background: "rgba(3, 4, 8, 0.75)",
        backdropFilter: "blur(6px)",
        fontSize: 11,
        fontFamily: "var(--font-mono)",
        pointerEvents: "auto",
        maxWidth: "80vw",
      }}
    >
      {truncatedLeft && (
        <span style={{ color: "rgba(148, 163, 184, 0.4)", padding: "2px 4px" }}>
          ...
        </span>
      )}
      {items.map((item, i) => {
        const isCurrent = i === currentIndex;
        const label = item.nodeId
          ? nodeNames.get(item.nodeId) ?? item.nodeId.slice(0, 8)
          : "Overview";

        return (
          <button
            key={item.index}
            onClick={() => onNavigate(item.index)}
            title={item.nodeId ? nodeNames.get(item.nodeId) ?? item.nodeId : "Overview"}
            style={{
              padding: "2px 8px",
              borderRadius: 4,
              border: "none",
              cursor: "pointer",
              fontSize: 11,
              fontFamily: "var(--font-mono)",
              transition: "all 150ms",
              background: isCurrent
                ? "rgba(99, 102, 241, 0.4)"
                : "rgba(148, 163, 184, 0.1)",
              color: isCurrent
                ? "var(--accent, #a5b4fc)"
                : "rgba(148, 163, 184, 0.6)",
              fontWeight: isCurrent ? 600 : 400,
              maxWidth: 100,
              overflow: "hidden",
              textOverflow: "ellipsis",
              whiteSpace: "nowrap",
            }}
          >
            {label}
          </button>
        );
      })}
      {truncatedRight && (
        <span style={{ color: "rgba(148, 163, 184, 0.4)", padding: "2px 4px" }}>
          ...
        </span>
      )}
    </div>
  );
}
