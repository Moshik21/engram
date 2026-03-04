import { useEngramStore } from "../../store";
import { KnowledgeEntityGroups } from "./KnowledgeEntityGroups";
import { KnowledgeRecallResults } from "./KnowledgeRecallResults";

export function KnowledgeMainArea() {
  const query = useEngramStore((s) => s.knowledgeQuery);
  const hasQuery = query.trim().length > 0;

  return (
    <div
      style={{
        flex: 1,
        overflowY: "auto",
        padding: "16px 20px",
        minWidth: 0,
      }}
    >
      {hasQuery ? <KnowledgeRecallResults /> : <KnowledgeEntityGroups />}
    </div>
  );
}
