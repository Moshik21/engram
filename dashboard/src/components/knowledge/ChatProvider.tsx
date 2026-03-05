import { useChat, type UseChatHelpers } from "@ai-sdk/react";
import { DefaultChatTransport, type UIMessage } from "ai";
import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  type ReactNode,
} from "react";
import { useEngramStore } from "../../store";
import { api } from "../../api/client";

type ChatContextType = UseChatHelpers<UIMessage>;

const ChatContext = createContext<ChatContextType | null>(null);

/** Extract plain text from a UIMessage's parts array. */
function textFromParts(msg: UIMessage): string {
  return msg.parts
    .filter((p): p is { type: "text"; text: string } => p.type === "text")
    .map((p) => p.text)
    .join("");
}

export function ChatProvider({ children }: { children: ReactNode }) {
  const activeConversationId = useEngramStore((s) => s.activeConversationId);
  const setConversationId = useEngramStore((s) => s.setConversationId);
  const loadConversations = useEngramStore((s) => s.loadConversations);
  const convIdRef = useRef(activeConversationId);
  convIdRef.current = activeConversationId;

  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: "/api/knowledge/chat",
        prepareSendMessagesRequest: async ({ messages }) => {
          const last = messages[messages.length - 1];
          const message = textFromParts(last);
          const history = messages.slice(0, -1).map((m) => ({
            role: m.role,
            content: textFromParts(m),
          }));
          const today = new Date().toISOString().slice(0, 10);
          return {
            body: {
              message,
              history,
              conversation_id: convIdRef.current ?? undefined,
              session_date: today,
            },
          };
        },
      }),
    [],
  );

  const chat = useChat({
    transport,
    onFinish: useCallback(() => {
      // Extract conversationId from the response metadata
      // The finish event includes it — but useChat doesn't expose SSE data directly.
      // We handle this via a side-channel: after assistant reply, persist via API.
    }, []),
  });

  const { messages, status, setMessages } = chat;
  const prevStatusRef = useRef(status);
  const prevMsgCountRef = useRef(messages.length);

  // When status transitions from streaming → ready with new messages, persist
  useEffect(() => {
    const wasStreaming =
      prevStatusRef.current === "streaming" || prevStatusRef.current === "submitted";
    const isReady = status === "ready";
    const hasNewMessages = messages.length > prevMsgCountRef.current;

    prevStatusRef.current = status;
    prevMsgCountRef.current = messages.length;

    if (wasStreaming && isReady && hasNewMessages && messages.length >= 2) {
      const lastAssistant = messages[messages.length - 1];
      const lastUser = messages[messages.length - 2];
      if (lastAssistant?.role !== "assistant" || lastUser?.role !== "user") return;

      const userText = textFromParts(lastUser);
      const assistantText = textFromParts(lastAssistant);

      // Persist messages to backend
      const convId = convIdRef.current;
      if (convId) {
        api
          .appendConversationMessages(convId, [
            { role: "user", content: userText },
            { role: "assistant", content: assistantText },
          ])
          .then(() => loadConversations())
          .catch(() => {});
      } else {
        // No conversation ID yet — the backend created one during /chat SSE.
        // We need to fetch conversations to discover it.
        loadConversations().then(() => {
          // Set the newest conversation as active
          const { conversations } = useEngramStore.getState();
          if (conversations.length > 0 && !convIdRef.current) {
            setConversationId(conversations[0].id);
          }
        });
      }
    }
  }, [status, messages, loadConversations, setConversationId]);

  // Load messages when activeConversationId changes
  const prevConvIdRef = useRef<string | null>(null);
  useEffect(() => {
    if (activeConversationId === prevConvIdRef.current) return;
    prevConvIdRef.current = activeConversationId;

    if (!activeConversationId) {
      setMessages([]);
      return;
    }

    api.getConversationMessages(activeConversationId).then((data) => {
      const uiMessages: UIMessage[] = data.messages.map((m) => ({
        id: m.id,
        role: m.role as "user" | "assistant",
        parts: [{ type: "text" as const, text: m.content }],
      }));
      setMessages(uiMessages);
    });
  }, [activeConversationId, setMessages]);

  return <ChatContext.Provider value={chat}>{children}</ChatContext.Provider>;
}

export function useChatContext(): ChatContextType {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error("useChatContext must be used inside ChatProvider");
  return ctx;
}
