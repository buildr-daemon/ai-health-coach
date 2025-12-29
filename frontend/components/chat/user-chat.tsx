"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { Button } from "@/components/ui/button";
import { AlertCircle, RotateCcw } from "lucide-react";

/**
 * Props for the UserChatMessage component
 */
interface UserChatMessageProps {
  /** Unique identifier for the message */
  messageId: number | string;
  /** The message text content */
  messageContent: string;
  /** Timestamp when message was created */
  createdAt: Date;
  /** Whether the message is currently being sent */
  isSending?: boolean;
  /** Whether the message failed to send */
  hasSendFailed?: boolean;
  /** Error message to display on failure */
  sendErrorMessage?: string;
  /** Callback function to retry sending the message */
  onRetryMessageSend?: () => void;
  /** Additional CSS classes */
  className?: string;
}

/**
 * UserChatMessage - Displays a user's message in iOS blue bubble style
 * Includes failure state with retry functionality
 */
export function UserChatMessage({
  messageId,
  messageContent,
  createdAt,
  isSending = false,
  hasSendFailed = false,
  sendErrorMessage,
  onRetryMessageSend,
  className,
}: UserChatMessageProps) {
  const formattedMessageTime = new Intl.DateTimeFormat("en-US", {
    hour: "numeric",
    minute: "2-digit",
    hour12: true,
  }).format(new Date(createdAt));

  return (
    <div
      className={cn(
        "flex w-full justify-end px-4 py-1",
        className
      )}
      data-message-id={messageId}
    >
      <div className="flex flex-col items-end max-w-[80%] md:max-w-[70%]">
        {/* Message Bubble */}
        <div
          className={cn(
            "relative px-4 py-2 rounded-2xl rounded-br-md",
            "bg-ios-blue text-white",
            "shadow-sm",
            "transition-opacity duration-200",
            isSending && "opacity-70",
            hasSendFailed && "opacity-60"
          )}
        >
          {/* Message Content */}
          <p className="text-[15px] leading-relaxed whitespace-pre-wrap wrap-break-word">
            {messageContent}
          </p>

          {/* Timestamp */}
          <div className="flex items-center justify-end gap-1 mt-1">
            <span className="text-[11px] text-white/70">
              {formattedMessageTime}
            </span>
            {isSending && (
              <span className="text-[11px] text-white/70">Sending...</span>
            )}
          </div>

          {/* Tail for bubble (WhatsApp style) */}
          <div
            className="absolute bottom-0 -right-2 w-4 h-4 overflow-hidden"
            aria-hidden="true"
          >
            <div className="absolute bottom-0 right-2 w-4 h-4 bg-ios-blue transform rotate-45 origin-bottom-left" />
          </div>
        </div>

        {/* Failure State with Retry */}
        {hasSendFailed && (
          <div className="flex items-center gap-2 mt-2 animate-in fade-in slide-in-from-top-1 duration-200">
            <div className="flex items-center gap-1.5 text-red-400">
              <AlertCircle className="size-4" />
              <span className="text-xs">
                {sendErrorMessage || "Failed to send"}
              </span>
            </div>
            {onRetryMessageSend && (
              <Button
                variant="ghost"
                size="sm"
                onClick={onRetryMessageSend}
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
  );
}

export default UserChatMessage;

