import { KnowledgeMainArea } from "./KnowledgeMainArea";
import { KnowledgeChatPanel } from "./KnowledgeChatPanel";

export function KnowledgeContent() {
  return (
    <div style={{ flex: 1, display: "flex", gap: 10, overflow: "hidden", minHeight: 0 }}>
      <KnowledgeMainArea />
      <KnowledgeChatPanel />
    </div>
  );
}
