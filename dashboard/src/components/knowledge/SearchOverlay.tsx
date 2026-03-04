import { useState, useRef, useEffect, useCallback } from "react";
import { useEngramStore } from "../../store";
import { api } from "../../api/client";
import { entityColor, entityColorDim } from "../../lib/colors";
import { debounce } from "../../lib/utils";
import type { SearchResult } from "../../store/types";
import { useChatContext } from "./ChatProvider";

export function SearchOverlay() {
  const setSearchOverlayOpen = useEngramStore((s) => s.setSearchOverlayOpen);
  const openDrawer = useEngramStore((s) => s.openDrawer);
  const { sendMessage } = useChatContext();

  const [query, setQuery] = useState("");
  const [results, setResults] = useState<SearchResult[]>([]);
  const [isSearching, setIsSearching] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  const doSearch = useCallback(
    debounce(async (q: string) => {
      if (!q.trim()) {
        setResults([]);
        setIsSearching(false);
        return;
      }
      setIsSearching(true);
      try {
        const data = await api.searchEntities({ q, limit: 10 });
        setResults(data);
      } catch {
        setResults([]);
      } finally {
        setIsSearching(false);
      }
    }, 200),
    [],
  );

  const handleChange = (value: string) => {
    setQuery(value);
    doSearch(value);
  };

  const close = () => setSearchOverlayOpen(false);

  return (
    <div
      onClick={close}
      style={{
        position: "absolute",
        inset: 0,
        zIndex: 50,
        background: "rgba(3, 4, 8, 0.7)",
        display: "flex",
        alignItems: "flex-start",
        justifyContent: "center",
        paddingTop: "15vh",
      }}
    >
      <div
        className="glass-elevated animate-slide-down"
        onClick={(e) => e.stopPropagation()}
        style={{
          width: "100%",
          maxWidth: 520,
          maxHeight: "60vh",
          borderRadius: "var(--radius-lg)",
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
        }}
      >
        {/* Search input */}
        <div style={{ padding: "12px 14px", borderBottom: "1px solid var(--border)" }}>
          <input
            ref={inputRef}
            value={query}
            onChange={(e) => handleChange(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Escape") close();
              if (e.key === "Enter" && query.trim()) {
                sendMessage({ text: query.trim() });
                close();
              }
            }}
            placeholder="Search entities..."
            style={{
              width: "100%",
              padding: "8px 12px",
              background: "rgba(255,255,255,0.03)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-sm)",
              color: "var(--text-primary)",
              fontSize: 14,
              fontFamily: "var(--font-body)",
              outline: "none",
            }}
          />
        </div>

        {/* Results */}
        <div style={{ overflowY: "auto", flex: 1 }}>
          {/* "Ask memory" option */}
          {query.trim() && (
            <button
              onClick={() => {
                sendMessage({ text: query.trim() });
                close();
              }}
              style={{
                display: "flex",
                alignItems: "center",
                gap: 8,
                width: "100%",
                padding: "10px 14px",
                background: "transparent",
                border: "none",
                borderBottom: "1px solid var(--border-subtle)",
                cursor: "pointer",
                textAlign: "left",
                color: "var(--accent)",
                fontSize: 13,
                fontFamily: "var(--font-body)",
                transition: "background 0.1s",
              }}
              onMouseEnter={(e) => { e.currentTarget.style.background = "var(--surface-hover)"; }}
              onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
            >
              <span style={{ fontSize: 11, color: "var(--text-muted)" }}>Ask memory:</span>
              <span>{query}</span>
            </button>
          )}

          {isSearching && (
            <div style={{ padding: 16, textAlign: "center" }}>
              <div className="skeleton" style={{ width: 120, height: 14, margin: "0 auto" }} />
            </div>
          )}

          {results.map((entity) => {
            const color = entityColor(entity.entityType);
            return (
              <button
                key={entity.id}
                onClick={() => {
                  openDrawer(entity.id);
                  close();
                }}
                style={{
                  display: "flex",
                  alignItems: "center",
                  gap: 8,
                  width: "100%",
                  padding: "10px 14px",
                  background: "transparent",
                  border: "none",
                  borderBottom: "1px solid var(--border-subtle)",
                  cursor: "pointer",
                  textAlign: "left",
                  color: "var(--text-primary)",
                  fontSize: 13,
                  fontFamily: "var(--font-body)",
                  transition: "background 0.1s",
                }}
                onMouseEnter={(e) => { e.currentTarget.style.background = "var(--surface-hover)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.background = "transparent"; }}
              >
                <span style={{ width: 8, height: 8, borderRadius: "50%", background: color, flexShrink: 0 }} />
                <span style={{ flex: 1, overflow: "hidden", textOverflow: "ellipsis", whiteSpace: "nowrap" }}>
                  {entity.name}
                </span>
                <span
                  className="pill"
                  style={{
                    borderColor: `${color}25`,
                    background: entityColorDim(entity.entityType, 0.08),
                    color,
                    fontSize: 9,
                    padding: "2px 6px",
                  }}
                >
                  {entity.entityType}
                </span>
              </button>
            );
          })}

          {!isSearching && query.trim() && results.length === 0 && (
            <div style={{ padding: 16, textAlign: "center", color: "var(--text-muted)", fontSize: 12 }}>
              No entities found
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
