import { useEffect, useState } from "react";

import { Markdown } from "@/components/umbrella/markdown";
import { UmbrellaMark } from "@/components/umbrella/logo";
import { UserAvatar } from "@/components/umbrella/user-avatar";
import type { Message } from "@/lib/umbrella-types";
import { cn } from "@/lib/utils";

function useTypedText(text: string, enabled: boolean) {
  const [shown, setShown] = useState(enabled ? "" : text);

  useEffect(() => {
    if (!enabled) {
      setShown(text);
      return;
    }
    let index = 0;
    setShown("");
    const interval = setInterval(() => {
      index = Math.min(text.length, index + 6);
      setShown(text.slice(0, index));
      if (index >= text.length) clearInterval(interval);
    }, 16);
    return () => clearInterval(interval);
  }, [text, enabled]);

  return { shown, done: shown.length >= text.length };
}

export function ChatMessage({
  message,
  userName,
  streaming = false,
}: {
  message: Message;
  userName: string;
  streaming?: boolean;
}) {
  const isUser = message.sender === "user";
  const { shown, done } = useTypedText(message.content, streaming && !isUser);
  // The backend evicts old uploads, so the URL can 404 in a long-lived
  // conversation. Hide the image rather than leaving a broken-image icon -
  // the message text is still perfectly readable without it.
  const [imageFailed, setImageFailed] = useState(false);
  const showImage = Boolean(message.imageUrl) && !imageFailed;

  return (
    <div className={cn("flex w-full gap-3 animate-fade-up", isUser && "justify-end")}>
      {!isUser && (
        <span className="mt-1 flex size-8 shrink-0 items-center justify-center rounded-md border border-border bg-card">
          <UmbrellaMark className="h-4 w-4" />
        </span>
      )}

      <div className={cn("min-w-0", isUser ? "max-w-[80%]" : "flex-1")}>
        {isUser ? (
          <div className="flex flex-col items-end gap-2">
            {showImage && (
              <a
                href={message.imageUrl}
                target="_blank"
                rel="noreferrer"
                className="block overflow-hidden rounded-2xl rounded-tr-sm border border-border"
              >
                <img
                  src={message.imageUrl}
                  alt={message.imageName ?? "Attached image"}
                  onError={() => setImageFailed(true)}
                  className="max-h-64 w-auto max-w-full object-contain"
                />
              </a>
            )}
            {message.content && (
              <div className="rounded-2xl rounded-tr-sm bg-primary px-4 py-2.5 text-[0.95rem] leading-7 text-primary-foreground">
                {message.content}
              </div>
            )}
          </div>
        ) : (
          <div className={cn(!done && "typing-caret")}>
            <Markdown content={shown} />
          </div>
        )}
      </div>

      {isUser && <UserAvatar name={userName} className="mt-1" />}
    </div>
  );
}
