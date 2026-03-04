import { useRef, useEffect, useCallback } from "react";
import { useEngramStore } from "../../store";
import { entityColor } from "../../lib/colors";

const ENTITY_TYPES = ["Person", "Technology", "Project", "Organization", "Concept", "Location", "Event"];

export function KnowledgeSearchBar() {
  const query = useEngramStore((s) => s.knowledgeQuery);
  const setQuery = useEngramStore((s) => s.setKnowledgeQuery);
  const executeRecall = useEngramStore((s) => s.executeRecall);
  const isRecalling = useEngramStore((s) => s.isRecalling);
  const activeTypeFilter = useEngramStore((s) => s.activeTypeFilter);
  const setActiveTypeFilter = useEngramStore((s) => s.setActiveTypeFilter);
  const entityGroups = useEngramStore((s) => s.entityGroups);
  const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined);

  const handleChange = useCallback(
    (value: string) => {
      setQuery(value);
      if (timerRef.current) clearTimeout(timerRef.current);
      timerRef.current = setTimeout(() => {
        executeRecall(value);
      }, 300);
    },
    [setQuery, executeRecall],
  );

  useEffect(() => {
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, []);

  // Only show types that actually have entities
  const availableTypes = ENTITY_TYPES.filter((t) => entityGroups[t]?.length);

  return (
    <div>
      {/* Search input */}
      <div style={{ position: "relative" }}>
        <span
          style={{
            position: "absolute",
            left: 12,
            top: "50%",
            transform: "translateY(-50%)",
            color: "var(--text-muted)",
            fontSize: 13,
            pointerEvents: "none",
          }}
        >
          {"\u2315"}
        </span>
        <input
          type="text"
          value={query}
          onChange={(e) => handleChange(e.target.value)}
          placeholder="Search memories..."
          style={{
            width: "100%",
            padding: "10px 12px 10px 32px",
            background: "rgba(255, 255, 255, 0.03)",
            border: "1px solid var(--border)",
            borderRadius: "var(--radius-md)",
            color: "var(--text-primary)",
            fontSize: 13,
            fontFamily: "var(--font-body)",
            outline: "none",
            transition: "border-color 0.15s ease",
          }}
          onFocus={(e) => {
            e.currentTarget.style.borderColor = "var(--border-active)";
          }}
          onBlur={(e) => {
            e.currentTarget.style.borderColor = "var(--border)";
          }}
        />
        {isRecalling && (
          <span
            className="animate-pulse-soft"
            style={{
              position: "absolute",
              right: 12,
              top: "50%",
              transform: "translateY(-50%)",
              width: 6,
              height: 6,
              borderRadius: "50%",
              background: "var(--accent)",
            }}
          />
        )}
      </div>

      {/* Type filter pills */}
      {availableTypes.length > 0 && (
        <div style={{ display: "flex", flexWrap: "wrap", gap: 5, marginTop: 8 }}>
          {availableTypes.map((type) => {
            const isActive = activeTypeFilter === type;
            const color = entityColor(type);
            const count = entityGroups[type]?.length ?? 0;
            return (
              <button
                key={type}
                onClick={() => setActiveTypeFilter(isActive ? null : type)}
                className={isActive ? "pill pill-active" : "pill"}
                style={{
                  cursor: "pointer",
                  borderColor: isActive ? `${color}30` : undefined,
                  background: isActive ? `${color}10` : undefined,
                  color: isActive ? color : undefined,
                }}
              >
                <span
                  style={{
                    width: 6,
                    height: 6,
                    borderRadius: "50%",
                    background: color,
                    display: "inline-block",
                    flexShrink: 0,
                  }}
                />
                {type}
                <span className="tabular-nums" style={{ opacity: 0.6 }}>
                  {count}
                </span>
              </button>
            );
          })}
        </div>
      )}
    </div>
  );
}
