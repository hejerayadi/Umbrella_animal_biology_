import { createFileRoute, useNavigate } from "@tanstack/react-router";
import { Loader2 } from "lucide-react";
import { useEffect, useRef } from "react";

import { useUmbrella } from "@/lib/umbrella-store";
import { useAuth } from "@/lib/auth-context";

export const Route = createFileRoute("/chat/")({
  component: ChatIndex,
});

function ChatIndex() {
  const { conversations, createConversation, hydrated } = useUmbrella();
  const { user, loading } = useAuth();
  const navigate = useNavigate();
  const done = useRef(false);

  useEffect(() => {
    if (!hydrated || loading || !user || done.current) return;
    done.current = true;
    const sorted = [...conversations].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
    const target = sorted[0] ?? createConversation();
    navigate({
      to: "/chat/$conversationId",
      params: { conversationId: target.id },
      replace: true,
    });
  }, [hydrated, loading, user, conversations, createConversation, navigate]);

  return (
    <div className="flex h-full items-center justify-center">
      <Loader2 className="h-5 w-5 animate-spin text-primary" />
    </div>
  );
}
