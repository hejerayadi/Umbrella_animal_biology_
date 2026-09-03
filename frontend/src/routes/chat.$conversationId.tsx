import { createFileRoute } from "@tanstack/react-router";
import { Compass } from "lucide-react";
import { useEffect, useRef } from "react";

import { AgentThinkingPanel } from "@/components/umbrella/agent-thinking-panel";
import { ChatComposer } from "@/components/umbrella/chat-composer";
import { ChatMessage } from "@/components/umbrella/chat-message";
import { UmbrellaMark } from "@/components/umbrella/logo";
import { SUGGESTED_PROMPTS } from "@/lib/mock-data";
import { useUmbrella } from "@/lib/umbrella-store";
import { useAuth } from "@/lib/auth-context";

export const Route = createFileRoute("/chat/$conversationId")({
  component: ConversationView,
});

function ConversationView() {
  const { conversationId } = Route.useParams();
  const {
    conversations,
    messagesFor,
    activitiesFor,
    liveAnswerFor,
    sendMessage,
    isThinking,
    streamingMessageId,
    hydrated,
  } = useUmbrella();
  const { user } = useAuth();

  const conversation = conversations.find((c) => c.id === conversationId);
  const messages = messagesFor(conversationId);
  const activities = activitiesFor(conversationId);
  const liveAnswer = liveAnswerFor(conversationId);
  const bottomRef = useRef<HTMLDivElement>(null);

  // A run in progress belongs with the reply it is about to become, so the
  // commentary is shown in the thread while it is live and only falls back to
  // the pinned panel above the composer once it is history.
  const live = isThinking || liveAnswer !== null;

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages.length, isThinking, conversationId]);

  // The answer arrives token by token; keep the newest line in view as it
  // grows, exactly as if the user were scrolling along with it.
  useEffect(() => {
    if (liveAnswer !== null) bottomRef.current?.scrollIntoView({ block: "end" });
  }, [liveAnswer]);

  if (hydrated && !conversation) {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 px-6 text-center">
        <Compass className="h-6 w-6 text-muted-foreground" />
        <p className="text-sm font-medium">This conversation no longer exists</p>
        <p className="text-sm text-muted-foreground">
          Pick another thread from the sidebar or start a new one.
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <header className="hidden h-14 shrink-0 items-center border-b border-border px-6 md:flex">
        <h1 className="truncate text-sm font-medium">{conversation?.title ?? "Conversation"}</h1>
      </header>

      <div className="min-h-0 flex-1 overflow-y-auto scroll-smooth">
        <div className="mx-auto max-w-3xl px-5 py-8">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center py-16 text-center">
              <UmbrellaMark className="h-9 w-9" />
              <h2 className="mt-5 text-2xl font-semibold">
                What are we investigating, {user?.full_name?.split(" ")[0] ?? "researcher"}?
              </h2>
              <p className="mt-2 max-w-md text-sm text-muted-foreground">
                Umbrella routes your question to the genome, biodiversity, trait, protein, and
                literature agents that fit it best.
              </p>
              <div className="mt-8 grid w-full gap-2 sm:grid-cols-2">
                {SUGGESTED_PROMPTS.map((prompt) => (
                  <button
                    key={prompt}
                    type="button"
                    onClick={() => sendMessage(conversationId, prompt)}
                    className="rounded-xl border border-border bg-card p-3.5 text-left text-sm transition-colors hover:border-primary/40 hover:bg-accent/40"
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="space-y-7">
              {messages.map((message) => (
                <ChatMessage
                  key={message.id}
                  message={message}
                  userName={user?.full_name ?? "Researcher"}
                  streaming={message.id === streamingMessageId}
                />
              ))}

              {live && (
                <div className="space-y-3">
                  <AgentThinkingPanel activities={activities} isThinking={isThinking} />
                  {liveAnswer !== null && (
                    <ChatMessage
                      message={{
                        id: "live",
                        conversationId,
                        sender: "assistant",
                        // Already arriving a token at a time, so the
                        // typewriter effect stays off - see `streaming`.
                        content: liveAnswer,
                        timestamp: new Date().toISOString(),
                      }}
                      userName={user?.full_name ?? "Researcher"}
                    />
                  )}
                </div>
              )}
            </div>
          )}
          <div ref={bottomRef} className="h-2" />
        </div>
      </div>

      <div className="shrink-0 border-t border-border bg-background/90 px-5 py-4 backdrop-blur">
        <div className="mx-auto max-w-3xl space-y-2">
          {!live && <AgentThinkingPanel activities={activities} isThinking={false} />}
          <ChatComposer
            focusKey={conversationId}
            // `live`, not `isThinking`: thinking ends at the first token of the
            // answer, and a second question sent while that answer is still
            // being written would take over the stream it is arriving on.
            disabled={live}
            onSend={(value, image) => sendMessage(conversationId, value, image)}
          />
          <p className="text-center text-[0.7rem] text-muted-foreground">
            Attach, paste or drop a photo to identify a species.
          </p>
        </div>
      </div>
    </div>
  );
}
