import { act, render } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";

const setMessagesSpy = vi.fn();

vi.mock("ai", () => ({
  DefaultChatTransport: class DefaultChatTransport {
    constructor() {}
  },
}));

vi.mock("@ai-sdk/react", () => ({
  useChat: vi.fn(() => ({
    messages: [],
    status: "ready",
    setMessages: setMessagesSpy,
    sendMessage: vi.fn(),
    error: undefined,
    id: "chat-test",
    stop: vi.fn(),
    regenerate: vi.fn(),
    resumeStream: vi.fn(),
    addToolResult: vi.fn(),
    addToolOutput: vi.fn(),
    addToolApprovalResponse: vi.fn(),
    clearError: vi.fn(),
  })),
}));

vi.mock("../api/client", async (importOriginal) => {
  const actual = await importOriginal<typeof import("../api/client")>();
  return {
    ...actual,
    api: {
      ...actual.api,
      getConversationMessages: vi.fn(),
      appendConversationMessages: vi.fn(),
      listConversations: vi.fn().mockResolvedValue({ conversations: [] }),
    },
  };
});

import { api } from "../api/client";
import { ChatProvider } from "../components/knowledge/ChatProvider";
import { useEngramStore } from "../store";

function resetConversationStore() {
  useEngramStore.setState({
    activeConversationId: null,
    conversations: [],
    isLoadingConversations: false,
  });
}

describe("ChatProvider", () => {
  beforeEach(() => {
    setMessagesSpy.mockReset();
    vi.mocked(api.getConversationMessages).mockReset();
    resetConversationStore();
  });

  it("ignores stale conversation loads after switching threads", async () => {
    let resolveFirst:
      | ((value: { messages: Array<{ id: string; role: "user" | "assistant"; content: string; partsJson: string | null; createdAt: string }> }) => void)
      | null = null;
    let resolveSecond:
      | ((value: { messages: Array<{ id: string; role: "user" | "assistant"; content: string; partsJson: string | null; createdAt: string }> }) => void)
      | null = null;

    vi.mocked(api.getConversationMessages)
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveFirst = resolve;
          }),
      )
      .mockImplementationOnce(
        () =>
          new Promise((resolve) => {
            resolveSecond = resolve;
          }),
      );

    render(
      <ChatProvider>
        <div>chat</div>
      </ChatProvider>,
    );

    act(() => {
      useEngramStore.getState().setActiveConversation("conv-1");
    });

    act(() => {
      useEngramStore.getState().setActiveConversation("conv-2");
    });

    await act(async () => {
      resolveFirst?.({
        messages: [
          {
            id: "msg-old",
            role: "assistant",
            content: "Old thread",
            partsJson: null,
            createdAt: "2026-03-06T00:00:00Z",
          },
        ],
      });
    });

    expect(setMessagesSpy).not.toHaveBeenCalled();

    await act(async () => {
      resolveSecond?.({
        messages: [
          {
            id: "msg-new",
            role: "assistant",
            content: "New thread",
            partsJson: null,
            createdAt: "2026-03-06T00:00:00Z",
          },
        ],
      });
    });

    expect(setMessagesSpy).toHaveBeenCalledTimes(1);
    expect(setMessagesSpy).toHaveBeenCalledWith([
      {
        id: "msg-new",
        role: "assistant",
        parts: [{ type: "text", text: "New thread" }],
      },
    ]);
  });
});
