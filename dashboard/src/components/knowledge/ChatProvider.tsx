import { useChat, type UseChatHelpers } from "@ai-sdk/react";
import { DefaultChatTransport, type UIMessage } from "ai";
import { createContext, useContext, useMemo, type ReactNode } from "react";

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
  const transport = useMemo(
    () =>
      new DefaultChatTransport({
        api: "/api/knowledge/chat",
        prepareSendMessagesRequest: async ({ messages }) => {
          // Convert AI SDK UIMessage[] → our backend's { message, history } format
          const last = messages[messages.length - 1];
          const message = textFromParts(last);
          const history = messages.slice(0, -1).map((m) => ({
            role: m.role,
            content: textFromParts(m),
          }));
          return { body: { message, history } };
        },
      }),
    [],
  );
  const chat = useChat({ transport });
  return <ChatContext.Provider value={chat}>{children}</ChatContext.Provider>;
}

export function useChatContext(): ChatContextType {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error("useChatContext must be used inside ChatProvider");
  return ctx;
}
