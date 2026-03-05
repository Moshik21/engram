import { useEffect } from "react";
import { useEngramStore } from "../../store";
import { ChatProvider } from "./ChatProvider";
import { ConversationSidebar } from "./ConversationSidebar";
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
  const loadConversations = useEngramStore((s) => s.loadConversations);
  const drawerEntityId = useEngramStore((s) => s.drawerEntityId);
  const searchOverlayOpen = useEngramStore((s) => s.searchOverlayOpen);
  const browseOverlayOpen = useEngramStore((s) => s.browseOverlayOpen);
  const confirmDialog = useEngramStore((s) => s.confirmDialog);
  const sidebarOpen = useEngramStore((s) => s.conversationSidebarOpen);

  useEffect(() => {
    loadPulseEntities();
    loadEntityGroups();
    loadConversations();
  }, [loadPulseEntities, loadEntityGroups, loadConversations]);

  return (
    <ChatProvider>
      <div
        className="animate-fade-in"
        style={{
          display: "flex",
          height: "100%",
          overflow: "hidden",
          position: "relative",
        }}
      >
        {sidebarOpen && <ConversationSidebar />}
        <div
          style={{
            flex: 1,
            display: "flex",
            flexDirection: "column",
            overflow: "hidden",
          }}
        >
          <MemoryPulse />
          <KnowledgeChatStream />
          <KnowledgeInputBar />
        </div>

        {drawerEntityId && <EntityDetailDrawer />}
        {searchOverlayOpen && <SearchOverlay />}
        {browseOverlayOpen && <EntityBrowseOverlay />}
        {confirmDialog && <ConfirmDialog />}
      </div>
    </ChatProvider>
  );
}
