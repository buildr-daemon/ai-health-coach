"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { AlertCircle, RotateCcw, Bot } from "lucide-react";

/**
 * Props for the AgentChatMessage component
 */
interface AgentChatMessageProps {
  /** Unique identifier for the message */
  messageId: number | string;
  /** The message text content from the agent */
  messageContent: string;
  /** Timestamp when message was created */
  createdAt: Date;
  /** Whether the agent response failed */
  hasResponseFailed?: boolean;
  /** Error message to display on failure */
  responseErrorMessage?: string;
  /** Callback function to retry getting agent response */
  onRetryAgentResponse?: () => void;
  /** Additional CSS classes */
  className?: string;
}

/**
 * AgentChatMessage - Displays an AI agent's message in iOS green bubble style
 * Includes failure state with retry functionality
 */
export function AgentChatMessage({
  messageId,
  messageContent,
  createdAt,
  hasResponseFailed = false,
  responseErrorMessage,
  onRetryAgentResponse,
  className,
}: AgentChatMessageProps) {
  const formattedMessageTime = new Intl.DateTimeFormat("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  }).format(new Date(createdAt));

  return (
    <div
      className={cn(
        "flex w-full justify-start px-4 py-1",
        className
      )}
      data-message-id={messageId}
    >
      <div className="flex items-end gap-2 max-w-[80%] md:max-w-[70%]">
        {/* Agent Avatar */}
        <Avatar className="size-8 shrink-0 mb-1">
          <AvatarFallback className="bg-ios-green text-white">
            <Bot className="size-4" />
          </AvatarFallback>
        </Avatar>

        <div className="flex flex-col items-start">
          {/* Message Bubble */}
          <div
            className={cn(
              "relative px-4 py-2 rounded-2xl rounded-bl-md",
              "bg-ios-green text-white",
              "shadow-sm",
              hasResponseFailed && "opacity-60"
            )}
          >
            {/* Message Content */}
            <p className="text-[15px] leading-relaxed whitespace-pre-wrap wrap-break-word">
              {messageContent}
            </p>

            {/* Timestamp */}
            <div className="flex items-center justify-end mt-1">
              <span className="text-[11px] text-white/70">
                {formattedMessageTime}
              </span>
            </div>

            {/* Tail for bubble (WhatsApp style) */}
            <div
              className="absolute bottom-0 -left-2 w-4 h-4 overflow-hidden"
              aria-hidden="true"
            >
              <div className="absolute bottom-0 left-2 w-4 h-4 bg-ios-green transform -rotate-45 origin-bottom-right" />
            </div>
          </div>

          {/* Failure State with Retry */}
          {hasResponseFailed && (
            <div className="flex items-center gap-2 mt-2 animate-in fade-in slide-in-from-top-1 duration-200">
              <div className="flex items-center gap-1.5 text-red-400">
                <AlertCircle className="size-4" />
                <span className="text-xs">
                  {responseErrorMessage || "Failed to get response"}
                </span>
              </div>
              {onRetryAgentResponse && (
                <Button
                  variant="ghost"
                  size="sm"
                  onClick={onRetryAgentResponse}
                  className="h-7 px-2 text-xs text-red-400 hover:text-red-300 hover:bg-red-500/10"
                >
                  <RotateCcw className="size-3 mr-1" />
                  Retry
                </Button>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default AgentChatMessage;

