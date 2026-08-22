import { Outlet, createFileRoute, useNavigate, useParams } from "@tanstack/react-router";
import { Menu, X } from "lucide-react";
import { useEffect, useState } from "react";

import { Button } from "@/components/ui/button";
import { ConversationSidebar } from "@/components/umbrella/conversation-sidebar";
import { UmbrellaLogo } from "@/components/umbrella/logo";
import { useAuth } from "@/lib/auth-context";
import { cn } from "@/lib/utils";

export const Route = createFileRoute("/chat")({
  head: () => ({
    meta: [
      { title: "Workspace — Umbrella" },
      {
        name: "description",
        content: "Your Umbrella workspace: multi-agent research conversations and orchestration.",
      },
      { name: "robots", content: "noindex" },
    ],
  }),
  component: ChatLayout,
});

function ChatLayout() {
  const { loading, user } = useAuth();
  const navigate = useNavigate();
  const [collapsed, setCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);

  useEffect(() => {
    if (!loading && !user) navigate({ to: "/signin" });
  }, [loading, user, navigate]);

  const params = useParams({ strict: false }) as { conversationId?: string };

  return (
    <div className="flex h-screen overflow-hidden bg-background">
      <div className="hidden md:flex">
        <ConversationSidebar
          activeId={params.conversationId}
          collapsed={collapsed}
          onToggleCollapsed={() => setCollapsed((v) => !v)}
        />
      </div>

      {mobileOpen && (
        <div className="fixed inset-0 z-40 flex md:hidden">
          <button
            type="button"
            aria-label="Close menu"
            className="absolute inset-0 bg-foreground/40"
            onClick={() => setMobileOpen(false)}
          />
          <div className="relative z-50" onClick={() => setMobileOpen(false)}>
            <ConversationSidebar
              activeId={params.conversationId}
              collapsed={false}
              onToggleCollapsed={() => setMobileOpen(false)}
            />
          </div>
        </div>
      )}

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-14 items-center gap-2 border-b border-border px-3 md:hidden">
          <Button
            variant="ghost"
            size="icon"
            aria-label={mobileOpen ? "Close conversations" : "Open conversations"}
            onClick={() => setMobileOpen((v) => !v)}
          >
            {mobileOpen ? <X className="h-4 w-4" /> : <Menu className="h-4 w-4" />}
          </Button>
          <UmbrellaLogo />
        </header>
        <div className={cn("min-h-0 flex-1")}>
          <Outlet />
        </div>
      </div>
    </div>
  );
}
