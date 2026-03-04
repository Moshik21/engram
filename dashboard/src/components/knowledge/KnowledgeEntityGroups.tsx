import { useEngramStore } from "../../store";
import { EntityGroupSection } from "./EntityGroupSection";

const TYPE_ORDER = ["Person", "Technology", "Project", "Organization", "Concept", "Location", "Event"];

export function KnowledgeEntityGroups() {
  const entityGroups = useEngramStore((s) => s.entityGroups);
  const isLoadingEntities = useEngramStore((s) => s.isLoadingEntities);
  const activeTypeFilter = useEngramStore((s) => s.activeTypeFilter);

  if (isLoadingEntities && Object.keys(entityGroups).length === 0) {
    return (
      <div style={{ padding: 20 }}>
        {[1, 2, 3].map((i) => (
          <div key={i} className="skeleton" style={{ height: 60, marginBottom: 8 }} />
        ))}
      </div>
    );
  }

  const allTypes = Object.keys(entityGroups);
  if (allTypes.length === 0) {
    return (
      <div style={{ textAlign: "center", padding: "60px 20px" }}>
        <p style={{ color: "var(--text-muted)", fontSize: 13 }}>No entities found</p>
        <p style={{ color: "var(--text-ghost)", fontSize: 11, marginTop: 4 }}>
          Use /remember or /observe to add memories
        </p>
      </div>
    );
  }

  // Sort types: known order first, then alphabetical
  const sortedTypes = activeTypeFilter
    ? allTypes.filter((t) => t === activeTypeFilter)
    : [...allTypes].sort((a, b) => {
        const ai = TYPE_ORDER.indexOf(a);
        const bi = TYPE_ORDER.indexOf(b);
        if (ai >= 0 && bi >= 0) return ai - bi;
        if (ai >= 0) return -1;
        if (bi >= 0) return 1;
        return a.localeCompare(b);
      });

  return (
    <div className="stagger">
      {sortedTypes.map((type) => (
        <EntityGroupSection
          key={type}
          type={type}
          entities={entityGroups[type] ?? []}
        />
      ))}
    </div>
  );
}
