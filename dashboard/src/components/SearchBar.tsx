import { useCallback, useEffect, useMemo, useRef } from "react";
import { useEngramStore } from "../store";
import { debounce } from "../lib/utils";
import { entityColor } from "../lib/colors";

export function SearchBar() {
  const searchQuery = useEngramStore((s) => s.searchQuery);
  const setSearchQuery = useEngramStore((s) => s.setSearchQuery);
  const executeSearch = useEngramStore((s) => s.executeSearch);
  const searchResults = useEngramStore((s) => s.searchResults);
  const isSearching = useEngramStore((s) => s.isSearching);
  const clearSearch = useEngramStore((s) => s.clearSearch);
  const loadNeighborhood = useEngramStore((s) => s.loadNeighborhood);
  const selectNode = useEngramStore((s) => s.selectNode);
  const inputRef = useRef<HTMLInputElement>(null);

  const debouncedSearch = useMemo(
    () => debounce((q: string) => executeSearch(q), 250),
    [executeSearch],
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const q = e.target.value;
      setSearchQuery(q);
      debouncedSearch(q);
    },
    [setSearchQuery, debouncedSearch],
  );

  const handleResultClick = useCallback(
    (id: string) => {
      clearSearch();
      loadNeighborhood(id);
      selectNode(id);
    },
    [clearSearch, loadNeighborhood, selectNode],
  );

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === "/" && e.target === document.body) {
        e.preventDefault();
        inputRef.current?.focus();
      }
      if (e.key === "Escape") {
        inputRef.current?.blur();
        clearSearch();
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [clearSearch]);

  return (
    <div style={{ position: "relative" }}>
      <div style={{ position: "relative" }}>
        <input
          ref={inputRef}
          type="text"
          placeholder="Search entities..."
          value={searchQuery}
          onChange={handleChange}
          style={{
            width: "100%",
            padding: "7px 32px 7px 12px",
            borderRadius: "var(--radius-sm)",
            border: "1px solid var(--border)",
            background: "var(--surface-hover)",
            color: "var(--text-primary)",
            fontFamily: "var(--font-body)",
            fontSize: 12,
            outline: "none",
            transition: "border-color 0.15s, box-shadow 0.15s",
          }}
          onFocus={(e) => {
            e.currentTarget.style.borderColor = "var(--border-active)";
            e.currentTarget.style.boxShadow = "0 0 0 1px var(--accent-dim)";
          }}
          onBlur={(e) => {
            e.currentTarget.style.borderColor = "var(--border)";
            e.currentTarget.style.boxShadow = "none";
          }}
        />
        {!searchQuery && (
          <span
            className="mono"
            style={{
              position: "absolute",
              right: 8,
              top: "50%",
              transform: "translateY(-50%)",
              fontSize: 10,
              color: "var(--text-ghost)",
              border: "1px solid var(--border)",
              borderRadius: "var(--radius-xs)",
              padding: "1px 5px",
              lineHeight: 1.3,
            }}
          >
            /
          </span>
        )}
        {searchQuery && (
          <button
            onClick={clearSearch}
            style={{
              position: "absolute",
              right: 8,
              top: "50%",
              transform: "translateY(-50%)",
              background: "none",
              border: "none",
              color: "var(--text-muted)",
              cursor: "pointer",
              fontSize: 14,
              lineHeight: 1,
              padding: 0,
              transition: "color 0.15s",
            }}
            onMouseEnter={(e) => {
              e.currentTarget.style.color = "var(--text-primary)";
            }}
            onMouseLeave={(e) => {
              e.currentTarget.style.color = "var(--text-muted)";
            }}
          >
            &times;
          </button>
        )}
      </div>

      {isSearching && (
        <div
          className="mono"
          style={{
            marginTop: 4,
            fontSize: 11,
            color: "var(--text-muted)",
          }}
        >
          Searching...
        </div>
      )}

      {searchResults.length > 0 && (
        <ul
          className="card animate-fade-in"
          style={{
            position: "absolute",
            left: 0,
            right: 0,
            marginTop: 4,
            borderRadius: "var(--radius-sm)",
            padding: 4,
            maxHeight: 240,
            overflowY: "auto",
            zIndex: 40,
            listStyle: "none",
          }}
        >
          {searchResults.map((r) => {
            const color = entityColor(r.entityType);
            return (
              <li key={r.id}>
                <button
                  onClick={() => handleResultClick(r.id)}
                  style={{
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    width: "100%",
                    padding: "6px 8px",
                    border: "none",
                    borderRadius: "var(--radius-xs)",
                    background: "transparent",
                    color: "var(--text-primary)",
                    fontFamily: "var(--font-body)",
                    fontSize: 12,
                    cursor: "pointer",
                    textAlign: "left",
                    transition: "background 0.12s",
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.background = "var(--surface-hover)";
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.background = "transparent";
                  }}
                >
                  <span
                    style={{
                      width: 6,
                      height: 6,
                      borderRadius: "50%",
                      background: color,
                      flexShrink: 0,
                      boxShadow: `0 0 6px ${color}40`,
                    }}
                  />
                  <span
                    style={{
                      flex: 1,
                      overflow: "hidden",
                      textOverflow: "ellipsis",
                      whiteSpace: "nowrap",
                    }}
                  >
                    {r.name}
                  </span>
                  <span
                    className="mono"
                    style={{
                      fontSize: 9,
                      color: color,
                      letterSpacing: "0.06em",
                      textTransform: "uppercase",
                    }}
                  >
                    {r.entityType}
                  </span>
                </button>
              </li>
            );
          })}
        </ul>
      )}
    </div>
  );
}
