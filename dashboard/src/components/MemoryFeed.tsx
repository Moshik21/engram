import { useCallback, useEffect, useState } from "react";
import { useEngramStore } from "../store";
import { EpisodeCard } from "./EpisodeCard";
import type { EpisodeStatus } from "../store/types";

const SOURCE_OPTIONS = ["all", "mcp", "api", "manual"] as const;
const STATUS_OPTIONS: Array<"all" | EpisodeStatus> = [
  "all",
  "queued",
  "processing",
  "completed",
  "failed",
];

function SkeletonCard() {
  return (
    <div
      className="card"
      style={{
        padding: 16,
        display: "flex",
        flexDirection: "column",
        gap: 10,
      }}
    >
      <div style={{ display: "flex", gap: 8 }}>
        <div className="skeleton" style={{ width: 60, height: 14 }} />
        <div className="skeleton" style={{ width: 40, height: 14 }} />
      </div>
      <div className="skeleton" style={{ height: 14, width: "90%" }} />
      <div className="skeleton" style={{ height: 14, width: "70%" }} />
    </div>
  );
}

export function MemoryFeed() {
  const episodes = useEngramStore((s) => s.episodes);
  const hasMore = useEngramStore((s) => s.hasMoreEpisodes);
  const isLoading = useEngramStore((s) => s.isLoadingEpisodes);
  const cursor = useEngramStore((s) => s.episodeCursor);
  const loadEpisodes = useEngramStore((s) => s.loadEpisodes);
  const selectNode = useEngramStore((s) => s.selectNode);
  const setCurrentView = useEngramStore((s) => s.setCurrentView);
  const loadNeighborhood = useEngramStore((s) => s.loadNeighborhood);

  const [sourceFilter, setSourceFilter] = useState<string>("all");
  const [statusFilter, setStatusFilter] = useState<string>("all");

  useEffect(() => {
    loadEpisodes();
  }, [loadEpisodes]);

  const handleLoadMore = useCallback(() => {
    if (cursor && hasMore && !isLoading) {
      loadEpisodes(cursor);
    }
  }, [cursor, hasMore, isLoading, loadEpisodes]);

  const handleEntityClick = useCallback(
    (entityId: string) => {
      selectNode(entityId);
      loadNeighborhood(entityId);
      setCurrentView("graph");
    },
    [selectNode, loadNeighborhood, setCurrentView],
  );

  const filtered = episodes.filter((ep) => {
    if (sourceFilter !== "all" && ep.source !== sourceFilter) return false;
    if (statusFilter !== "all" && ep.status !== statusFilter) return false;
    return true;
  });

  return (
    <div
      className="animate-fade-in"
      style={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
        padding: "10px 14px",
        overflow: "hidden",
      }}
    >
      {/* Filter bar */}
      <div
        className="card"
        style={{
          padding: "8px 12px",
          marginBottom: 10,
          display: "flex",
          alignItems: "center",
          gap: 10,
          flexShrink: 0,
        }}
      >
        <span className="label">Filter</span>

        <select
          value={sourceFilter}
          onChange={(e) => setSourceFilter(e.target.value)}
          style={{
            padding: "3px 8px",
            borderRadius: "var(--radius-xs)",
            border: "1px solid var(--border)",
            background: "var(--surface-hover)",
            color: "var(--text-secondary)",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            outline: "none",
            cursor: "pointer",
          }}
        >
          {SOURCE_OPTIONS.map((opt) => (
            <option key={opt} value={opt}>
              {opt === "all" ? "All sources" : opt}
            </option>
          ))}
        </select>

        <select
          value={statusFilter}
          onChange={(e) => setStatusFilter(e.target.value)}
          style={{
            padding: "3px 8px",
            borderRadius: "var(--radius-xs)",
            border: "1px solid var(--border)",
            background: "var(--surface-hover)",
            color: "var(--text-secondary)",
            fontFamily: "var(--font-mono)",
            fontSize: 11,
            outline: "none",
            cursor: "pointer",
          }}
        >
          {STATUS_OPTIONS.map((opt) => (
            <option key={opt} value={opt}>
              {opt === "all" ? "All statuses" : opt}
            </option>
          ))}
        </select>

        <span style={{ flex: 1 }} />

        <span
          className="mono tabular-nums"
          style={{ fontSize: 10, color: "var(--text-muted)" }}
        >
          {filtered.length} episode{filtered.length !== 1 ? "s" : ""}
        </span>
      </div>

      {/* Episode list */}
      <div
        style={{
          flex: 1,
          overflowY: "auto",
          display: "flex",
          flexDirection: "column",
          gap: 6,
          paddingBottom: 16,
        }}
      >
        {isLoading && episodes.length === 0 && (
          <>
            <SkeletonCard />
            <SkeletonCard />
            <SkeletonCard />
          </>
        )}

        {filtered.map((ep) => (
          <EpisodeCard
            key={ep.episodeId}
            episode={ep}
            onEntityClick={handleEntityClick}
          />
        ))}

        {!isLoading && filtered.length === 0 && episodes.length > 0 && (
          <div
            style={{
              textAlign: "center",
              padding: 32,
            }}
          >
            <span className="label">No episodes match current filters</span>
          </div>
        )}

        {!isLoading && episodes.length === 0 && (
          <div
            style={{
              textAlign: "center",
              padding: 32,
            }}
          >
            <span className="label">
              No episodes yet. Use MCP tools to create memories.
            </span>
          </div>
        )}

        {hasMore && !isLoading && filtered.length > 0 && (
          <button
            onClick={handleLoadMore}
            style={{
              alignSelf: "center",
              padding: "6px 16px",
              borderRadius: "var(--radius-sm)",
              border: "1px solid var(--border)",
              background: "transparent",
              color: "var(--text-secondary)",
              fontFamily: "var(--font-mono)",
              fontSize: 11,
              cursor: "pointer",
              transition: "all 0.15s",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.borderColor = "var(--border-hover)";
              e.currentTarget.style.color = "var(--text-primary)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.borderColor = "var(--border)";
              e.currentTarget.style.color = "var(--text-secondary)";
            }}
          >
            Load more
          </button>
        )}

        {isLoading && episodes.length > 0 && <SkeletonCard />}
      </div>
    </div>
  );
}
