import { useEffect } from "react";
import { useEngramStore } from "../../store";
import { ChatProvider } from "./ChatProvider";
import { MemoryPulse } from "./MemoryPulse";
import { KnowledgeChatStream } from "./KnowledgeChatStream";
import { KnowledgeInputBar } from "./KnowledgeInputBar";
import { EntityDetailDrawer } from "./EntityDetailDrawer";
import { SearchOverlay } from "./SearchOverlay";
import { EntityBrowseOverlay } from "./EntityBrowseOverlay";
import { ConfirmDialog } from "./ConfirmDialog";

export function KnowledgePanel() {
  const loadPulseEntities = useEngramStore((s) => s.loadPulseEntities);
  const loadEntityGroups = useEngramStore((s) => s.loadEntityGroups);
  const drawerEntityId = useEngramStore((s) => s.drawerEntityId);
  const searchOverlayOpen = useEngramStore((s) => s.searchOverlayOpen);
  const browseOverlayOpen = useEngramStore((s) => s.browseOverlayOpen);
  const confirmDialog = useEngramStore((s) => s.confirmDialog);

  useEffect(() => {
    loadPulseEntities();
    loadEntityGroups();
  }, [loadPulseEntities, loadEntityGroups]);

  return (
    <ChatProvider>
      <div
        className="animate-fade-in"
        style={{
          display: "flex",
          flexDirection: "column",
          height: "100%",
          overflow: "hidden",
          position: "relative",
        }}
      >
        <MemoryPulse />
        <KnowledgeChatStream />
        <KnowledgeInputBar />

        {drawerEntityId && <EntityDetailDrawer />}
        {searchOverlayOpen && <SearchOverlay />}
        {browseOverlayOpen && <EntityBrowseOverlay />}
        {confirmDialog && <ConfirmDialog />}
      </div>
    </ChatProvider>
  );
}
