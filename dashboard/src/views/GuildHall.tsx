import { lazy, Suspense } from "react";

const KnowledgePanel = lazy(() =>
  import("../components/knowledge/KnowledgePanel").then((m) => ({
    default: m.KnowledgePanel,
  })),
);

export function GuildHall() {
  return (
    <div style={{ height: "100%" }}>
      <Suspense
        fallback={
          <div style={{ display: "flex", alignItems: "center", justifyContent: "center", height: "100%" }}>
            <span className="label">Opening the Guild Hall...</span>
          </div>
        }
      >
        <KnowledgePanel />
      </Suspense>
    </div>
  );
}
