"use client";

import React, {
  useState,
  useEffect,
  useRef,
  useCallback,
  FormEvent,
} from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Spinner } from "@/components/ui/spinner";
import { UserChatMessage } from "@/components/chat/user-chat";
import { AgentChatMessage } from "@/components/chat/agent-chat-message";
import { TypingIndicator } from "@/components/chat/typing-indicator";
import { Send, ChevronUp, MessageCircle } from "lucide-react";
import { cn } from "@/lib/utils";

// === Types ===

interface ChatMessage {
  message_id: number | string;
  role: "user" | "assistant";
  content: string;
  created_at: string;
  /** Client-side state for pending messages */
  isPending?: boolean;
  /** Client-side state for failed messages */
  hasFailed?: boolean;
  /** Error message if failed */
  errorMessage?: string;
  /** Temporary ID for optimistic updates */
  temporaryId?: string;
}

interface ChatInitializationResponse {
  user_id: number;
  onboarding_status: string;
  recent_message_history: ChatMessage[];
  has_more_history: boolean;
  user_display_name: string | null;
}

interface ChatHistoryPaginationResponse {
  messages: ChatMessage[];
  has_more_messages: boolean;
  next_cursor_message_id: number | null;
  total_message_count: number;
}

interface SendMessageResponse {
  user_message: ChatMessage;
  assistant_reply: ChatMessage;
  extracted_health_insights: string[] | null;
}

// === API Configuration ===

const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";
const API_VERSION_PREFIX = "/api/v1";

// === API Functions ===

async function initializeChatSession(
  deviceIdentifier?: string
): Promise<ChatInitializationResponse> {
  const response = await fetch(
    `${API_BASE_URL}${API_VERSION_PREFIX}/chat/initialize`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        device_identifier: deviceIdentifier || null,
      }),
    }
  );

  if (!response.ok) {
    throw new Error(`Failed to initialize chat: ${response.statusText}`);
  }

  return response.json();
}

async function fetchChatHistoryPage(
  userId: number,
  cursorMessageId?: number,
  pageSize: number = 20
): Promise<ChatHistoryPaginationResponse> {
  const queryParams = new URLSearchParams({
    user_id: userId.toString(),
    page_size: pageSize.toString(),
  });

  if (cursorMessageId) {
    queryParams.append("cursor_message_id", cursorMessageId.toString());
  }

  const response = await fetch(
    `${API_BASE_URL}${API_VERSION_PREFIX}/chat/history?${queryParams}`,
    {
      method: "GET",
      headers: { "Content-Type": "application/json" },
    }
  );

  if (!response.ok) {
    throw new Error(`Failed to fetch chat history: ${response.statusText}`);
  }

  return response.json();
}

async function sendMessageToAgent(
  userId: number,
  messageContent: string
): Promise<SendMessageResponse> {
  const response = await fetch(
    `${API_BASE_URL}${API_VERSION_PREFIX}/chat/message`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        user_id: userId,
        message_content: messageContent,
      }),
    }
  );

  if (!response.ok) {
    throw new Error(`Failed to send message: ${response.statusText}`);
  }

  return response.json();
}

// === Utility Functions ===

function generateTemporaryMessageId(): string {
  return `temp_${Date.now()}_${Math.random().toString(36).substring(2, 9)}`;
}

function getDeviceIdentifier(): string {
  if (typeof window === "undefined") return "";

  let storedDeviceId = localStorage.getItem("health_agent_device_id");

  if (!storedDeviceId) {
    storedDeviceId = `device_${Date.now()}_${Math.random()
      .toString(36)
      .substring(2, 15)}`;
    localStorage.setItem("health_agent_device_id", storedDeviceId);
  }

  return storedDeviceId;
}

// === Main Chat Component ===

export default function ChatPage() {
  // === State ===
  const [currentUserId, setCurrentUserId] = useState<number | null>(null);
  const [chatMessages, setChatMessages] = useState<ChatMessage[]>([]);
  const [userInputValue, setUserInputValue] = useState<string>("");
  const [isInitializing, setIsInitializing] = useState<boolean>(true);
  const [initializationError, setInitializationError] = useState<string | null>(
    null
  );
  const [isAgentProcessing, setIsAgentProcessing] = useState<boolean>(false);
  const [isLoadingOlderMessages, setIsLoadingOlderMessages] =
    useState<boolean>(false);
  const [hasMoreOlderMessages, setHasMoreOlderMessages] =
    useState<boolean>(false);
  const [oldestLoadedMessageId, setOldestLoadedMessageId] = useState<
    number | null
  >(null);
  // Track pending retry for potential UI feedback (e.g., showing retry banner)
  const [, setPendingRetryMessage] = useState<{
    content: string;
    temporaryId: string;
  } | null>(null);

  // === Refs ===
  const scrollAreaViewportRef = useRef<HTMLDivElement>(null);
  const messageInputRef = useRef<HTMLInputElement>(null);
  const shouldAutoScrollRef = useRef<boolean>(true);
  const previousScrollHeightRef = useRef<number>(0);

  // === Auto-scroll to bottom ===
  const scrollToLatestMessage = useCallback(() => {
    if (scrollAreaViewportRef.current && shouldAutoScrollRef.current) {
      const viewport = scrollAreaViewportRef.current;
      viewport.scrollTop = viewport.scrollHeight;
    }
  }, []);

  // === Check if user is near bottom (for auto-scroll logic) ===
  const checkIfNearBottomOfChat = useCallback(() => {
    if (!scrollAreaViewportRef.current) return true;

    const viewport = scrollAreaViewportRef.current;
    const distanceFromBottom =
      viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight;

    // Consider "near bottom" if within 100px
    return distanceFromBottom < 100;
  }, []);

  // === Load older messages (pagination) ===
  const loadOlderMessages = useCallback(async () => {
    if (!currentUserId || isLoadingOlderMessages || !hasMoreOlderMessages)
      return;

    setIsLoadingOlderMessages(true);

    // Store current scroll height to maintain position
    if (scrollAreaViewportRef.current) {
      previousScrollHeightRef.current =
        scrollAreaViewportRef.current.scrollHeight;
    }

    try {
      const historyResponse = await fetchChatHistoryPage(
        currentUserId,
        oldestLoadedMessageId || undefined,
        20
      );

      if (historyResponse.messages.length > 0) {
        setChatMessages((previousMessages) => [
          ...historyResponse.messages,
          ...previousMessages,
        ]);

        // Update oldest message cursor
        const newOldestMessage = historyResponse.messages[0];
        if (
          newOldestMessage &&
          typeof newOldestMessage.message_id === "number"
        ) {
          setOldestLoadedMessageId(newOldestMessage.message_id);
        }
      }

      setHasMoreOlderMessages(historyResponse.has_more_messages);

      // Maintain scroll position after loading older messages
      requestAnimationFrame(() => {
        if (scrollAreaViewportRef.current) {
          const newScrollHeight = scrollAreaViewportRef.current.scrollHeight;
          const scrollDifference =
            newScrollHeight - previousScrollHeightRef.current;
          scrollAreaViewportRef.current.scrollTop = scrollDifference;
        }
      });
    } catch (error) {
      console.error("Failed to load older messages:", error);
    } finally {
      setIsLoadingOlderMessages(false);
    }
  }, [currentUserId, isLoadingOlderMessages, hasMoreOlderMessages, oldestLoadedMessageId]);

  // === State for triggering load on scroll ===
  const [shouldLoadOlderOnScroll, setShouldLoadOlderOnScroll] = useState(false);

  // === Handle scroll for pagination ===
  const handleChatScroll = useCallback(() => {
    if (!scrollAreaViewportRef.current) return;

    const viewport = scrollAreaViewportRef.current;

    // Update auto-scroll flag based on scroll position
    shouldAutoScrollRef.current = checkIfNearBottomOfChat();

    // Check if scrolled to top for loading older messages
    if (
      viewport.scrollTop < 50 &&
      hasMoreOlderMessages &&
      !isLoadingOlderMessages &&
      currentUserId
    ) {
      setShouldLoadOlderOnScroll(true);
    }
  }, [
    hasMoreOlderMessages,
    isLoadingOlderMessages,
    currentUserId,
    checkIfNearBottomOfChat,
  ]);

  // === Effect to load older messages when triggered by scroll ===
  useEffect(() => {
    if (shouldLoadOlderOnScroll && !isLoadingOlderMessages) {
      loadOlderMessages();
      setShouldLoadOlderOnScroll(false);
    }
  }, [shouldLoadOlderOnScroll, isLoadingOlderMessages, loadOlderMessages]);

  // === Initialize chat session ===
  useEffect(() => {
    async function initializeChat() {
      setIsInitializing(true);
      setInitializationError(null);

      try {
        const deviceId = getDeviceIdentifier();
        const initResponse = await initializeChatSession(deviceId);

        setCurrentUserId(initResponse.user_id);
        setChatMessages(initResponse.recent_message_history);
        setHasMoreOlderMessages(initResponse.has_more_history);

        // Set oldest message cursor for pagination
        if (initResponse.recent_message_history.length > 0) {
          const oldestMessage = initResponse.recent_message_history[0];
          if (typeof oldestMessage.message_id === "number") {
            setOldestLoadedMessageId(oldestMessage.message_id);
          }
        }

        // Auto-scroll to latest message after initialization
        requestAnimationFrame(() => {
          scrollToLatestMessage();
        });
      } catch (error) {
        console.error("Chat initialization failed:", error);
        setInitializationError(
          error instanceof Error ? error.message : "Failed to initialize chat"
        );
      } finally {
        setIsInitializing(false);
      }
    }

    initializeChat();
  }, [scrollToLatestMessage]);

  // === Auto-scroll when new messages are added ===
  useEffect(() => {
    if (shouldAutoScrollRef.current) {
      scrollToLatestMessage();
    }
  }, [chatMessages, scrollToLatestMessage]);

  // === Send message handler ===
  const handleSendMessage = useCallback(
    async (messageContent: string, retryTemporaryId?: string) => {
      if (!currentUserId || !messageContent.trim()) return;

      const trimmedMessageContent = messageContent.trim();
      const temporaryMessageId =
        retryTemporaryId || generateTemporaryMessageId();

      // Remove any failed message with same temp ID (for retry)
      if (retryTemporaryId) {
        setChatMessages((prev) =>
          prev.filter((msg) => msg.temporaryId !== retryTemporaryId)
        );
        setPendingRetryMessage(null);
      }

      // Add optimistic user message
      const optimisticUserMessage: ChatMessage = {
        message_id: temporaryMessageId,
        role: "user",
        content: trimmedMessageContent,
        created_at: new Date().toISOString(),
        isPending: true,
        temporaryId: temporaryMessageId,
      };

      setChatMessages((prev) => [...prev, optimisticUserMessage]);
      setUserInputValue("");
      shouldAutoScrollRef.current = true;
      setIsAgentProcessing(true);

      try {
        const sendResponse = await sendMessageToAgent(
          currentUserId,
          trimmedMessageContent
        );

        // Replace optimistic message with actual message and add agent response
        setChatMessages((prev) => {
          const messagesWithoutOptimistic = prev.filter(
            (msg) => msg.temporaryId !== temporaryMessageId
          );
          return [
            ...messagesWithoutOptimistic,
            sendResponse.user_message,
            sendResponse.assistant_reply,
          ];
        });
      } catch (error) {
        console.error("Failed to send message:", error);

        // Mark user message as failed
        setChatMessages((prev) =>
          prev.map((msg) =>
            msg.temporaryId === temporaryMessageId
              ? {
                  ...msg,
                  isPending: false,
                  hasFailed: true,
                  errorMessage:
                    error instanceof Error
                      ? error.message
                      : "Failed to send message",
                }
              : msg
          )
        );

        // Store for retry
        setPendingRetryMessage({
          content: trimmedMessageContent,
          temporaryId: temporaryMessageId,
        });
      } finally {
        setIsAgentProcessing(false);
      }
    },
    [currentUserId]
  );

  // === Form submit handler ===
  const handleFormSubmit = useCallback(
    (event: FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      handleSendMessage(userInputValue);
    },
    [userInputValue, handleSendMessage]
  );

  // === Retry user message ===
  const handleRetryUserMessage = useCallback(
    (temporaryId: string, content: string) => {
      handleSendMessage(content, temporaryId);
    },
    [handleSendMessage]
  );

  // === Retry agent response ===
  const handleRetryAgentResponse = useCallback(
    (userMessageContent: string) => {
      // This would re-trigger the agent for the last user message
      if (currentUserId) {
        handleSendMessage(userMessageContent);
      }
    },
    [currentUserId, handleSendMessage]
  );

  // === Render loading state ===
  if (isInitializing) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-whatsapp-dark-bg">
        <MessageCircle className="size-16 text-whatsapp-teal mb-4 animate-pulse" />
        <Spinner className="size-8 text-whatsapp-teal" />
        <p className="text-gray-400 mt-4">Starting chat...</p>
      </div>
    );
  }

  // === Render error state ===
  if (initializationError) {
    return (
      <div className="flex flex-col items-center justify-center h-screen bg-whatsapp-dark-bg px-4">
        <MessageCircle className="size-16 text-red-400 mb-4" />
        <p className="text-red-400 text-center mb-4">{initializationError}</p>
        <Button
          onClick={() => window.location.reload()}
          className="bg-whatsapp-teal hover:bg-whatsapp-teal/80 text-white"
        >
          Retry Connection
        </Button>
      </div>
    );
  }

  // === Main chat render ===
  return (
    <div className="flex flex-col h-screen bg-whatsapp-dark-bg">
      {/* Chat Header */}
      <header className="flex items-center gap-3 px-4 py-3 bg-whatsapp-header-bg border-b border-whatsapp-border">
        <div className="flex items-center justify-center size-10 rounded-full bg-ios-green">
          <MessageCircle className="size-5 text-white" />
        </div>
        <div className="flex flex-col">
          <h1 className="text-white font-semibold text-lg">Health Agent</h1>
          <span className="text-xs text-gray-400">
            {isAgentProcessing ? "typing..." : "online"}
          </span>
        </div>
      </header>

      {/* Chat Messages Area */}
      <div
        ref={scrollAreaViewportRef}
        className="flex-1 overflow-y-auto whatsapp-chat-pattern chat-scrollbar"
        onScroll={handleChatScroll}
      >
          <div className="flex flex-col py-4 min-h-full">
            {/* Load More Button / Indicator */}
            {hasMoreOlderMessages && (
              <div className="flex justify-center py-2">
                {isLoadingOlderMessages ? (
                  <div className="flex items-center gap-2 text-gray-400">
                    <Spinner className="size-4" />
                    <span className="text-sm">Loading older messages...</span>
                  </div>
                ) : (
                  <Button
                    variant="ghost"
                    size="sm"
                    onClick={loadOlderMessages}
                    className="text-gray-400 hover:text-white hover:bg-whatsapp-header-bg"
                  >
                    <ChevronUp className="size-4 mr-1" />
                    Load older messages
                  </Button>
                )}
              </div>
            )}

            {/* Empty state */}
            {chatMessages.length === 0 && !isAgentProcessing && (
              <div className="flex-1 flex flex-col items-center justify-center text-center px-8">
                <div className="size-20 rounded-full bg-whatsapp-header-bg flex items-center justify-center mb-4">
                  <MessageCircle className="size-10 text-whatsapp-teal" />
                </div>
                <h2 className="text-white text-xl font-semibold mb-2">
                  Welcome to Health Agent
                </h2>
                <p className="text-gray-400 max-w-md">
                  Start a conversation to get personalized health guidance. Your
                  chat history will be saved for future reference.
                </p>
              </div>
            )}

            {/* Message List */}
            {chatMessages.map((message, index) => {
              const isUserMessage = message.role === "user";
              const messageKey = message.temporaryId || message.message_id;

              if (isUserMessage) {
                return (
                  <UserChatMessage
                    key={messageKey}
                    messageId={message.message_id}
                    messageContent={message.content}
                    createdAt={new Date(message.created_at)}
                    isSending={message.isPending}
                    hasSendFailed={message.hasFailed}
                    sendErrorMessage={message.errorMessage}
                    onRetryMessageSend={
                      message.hasFailed && message.temporaryId
                        ? () =>
                            handleRetryUserMessage(
                              message.temporaryId!,
                              message.content
                            )
                        : undefined
                    }
                  />
                );
              }

              return (
                <AgentChatMessage
                  key={messageKey}
                  messageId={message.message_id}
                  messageContent={message.content}
                  createdAt={new Date(message.created_at)}
                  hasResponseFailed={message.hasFailed}
                  responseErrorMessage={message.errorMessage}
                  onRetryAgentResponse={
                    message.hasFailed
                      ? () => {
                          // Find the user message before this agent message to retry
                          const previousUserMessage = chatMessages
                            .slice(0, index)
                            .reverse()
                            .find((m) => m.role === "user");
                          if (previousUserMessage) {
                            handleRetryAgentResponse(previousUserMessage.content);
                          }
                        }
                      : undefined
                  }
                />
              );
            })}

            {/* Typing Indicator */}
            {isAgentProcessing && (
              <TypingIndicator isVisible={true} displayText="typing" />
            )}
          </div>
      </div>

      {/* Message Input Area */}
      <div className="px-4 py-3 bg-whatsapp-header-bg border-t border-whatsapp-border">
        <form
          onSubmit={handleFormSubmit}
          className="flex items-center gap-3"
        >
          <Input
            ref={messageInputRef}
            type="text"
            placeholder="Type a message..."
            value={userInputValue}
            onChange={(e) => setUserInputValue(e.target.value)}
            disabled={isAgentProcessing}
            className={cn(
              "flex-1 h-11 px-4 rounded-full",
              "bg-whatsapp-input-bg border-whatsapp-border",
              "text-white placeholder:text-gray-500",
              "focus-visible:ring-whatsapp-teal focus-visible:border-whatsapp-teal"
            )}
            maxLength={4000}
          />
          <Button
            type="submit"
            disabled={!userInputValue.trim() || isAgentProcessing}
            className={cn(
              "size-11 rounded-full p-0",
              "bg-whatsapp-teal hover:bg-whatsapp-teal/80",
              "disabled:bg-gray-600 disabled:opacity-50"
            )}
          >
            {isAgentProcessing ? (
              <Spinner className="size-5 text-white" />
            ) : (
              <Send className="size-5 text-white" />
            )}
          </Button>
        </form>

        {/* Character count indicator */}
        {userInputValue.length > 3500 && (
          <div className="flex justify-end mt-1">
            <span
              className={cn(
                "text-xs",
                userInputValue.length > 3900 ? "text-red-400" : "text-gray-500"
              )}
            >
              {userInputValue.length}/4000
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
