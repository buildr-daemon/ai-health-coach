"use client";

import React from "react";
import { cn } from "@/lib/utils";
import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Bot } from "lucide-react";

/**
 * Props for the TypingIndicator component
 */
interface TypingIndicatorProps {
  /** Whether to show the typing indicator */
  isVisible?: boolean;
  /** Display text like "typing..." or custom text */
  displayText?: string;
  /** Additional CSS classes */
  className?: string;
}

/**
 * TypingIndicator - WhatsApp-like animated typing indicator
 * Shows bouncing dots to indicate the agent is processing/typing
 */
export function TypingIndicator({
  isVisible = true,
  displayText = "typing",
  className,
}: TypingIndicatorProps) {
  if (!isVisible) {
    return null;
  }

  return (
    <div
      className={cn(
        "flex w-full justify-start px-4 py-1",
        "animate-in fade-in slide-in-from-bottom-2 duration-300",
        className
      )}
      role="status"
      aria-label="Agent is typing"
    >
      <div className="flex items-end gap-2">
        {/* Agent Avatar */}
        <Avatar className="size-8 shrink-0 mb-1">
          <AvatarFallback className="bg-ios-green text-white">
            <Bot className="size-4" />
          </AvatarFallback>
        </Avatar>

        {/* Typing Bubble */}
        <div
          className={cn(
            "relative px-4 py-3 rounded-2xl rounded-bl-md",
            "bg-whatsapp-message-incoming",
            "shadow-sm"
          )}
        >
          <div className="flex items-center gap-1.5">
            {/* Animated Dots */}
            <div className="flex items-center gap-1">
              <TypingDot delayClass="animation-delay-0" />
              <TypingDot delayClass="animation-delay-150" />
              <TypingDot delayClass="animation-delay-300" />
            </div>
            
            {/* Optional typing text */}
            {displayText && (
              <span className="text-[13px] text-gray-400 ml-1.5">
                {displayText}
              </span>
            )}
          </div>

          {/* Tail for bubble */}
          <div
            className="absolute bottom-0 -left-2 w-4 h-4 overflow-hidden"
            aria-hidden="true"
          >
            <div className="absolute bottom-0 left-2 w-4 h-4 bg-whatsapp-message-incoming transform -rotate-45 origin-bottom-right" />
          </div>
        </div>
      </div>
    </div>
  );
}

/**
 * Individual animated dot for the typing indicator
 */
function TypingDot({ delayClass }: { delayClass: string }) {
  return (
    <span
      className={cn(
        "inline-block size-2 rounded-full bg-gray-400",
        "animate-typing-bounce",
        delayClass
      )}
    />
  );
}

export default TypingIndicator;
