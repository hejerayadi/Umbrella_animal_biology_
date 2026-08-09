import { Link, useNavigate } from "@tanstack/react-router";
import {
  Check,
  LogOut,
  MessageSquarePlus,
  PanelLeftClose,
  PanelLeftOpen,
  Pencil,
  Search,
  Trash2,
  X,
} from "lucide-react";
import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { UmbrellaLogo, UmbrellaMark } from "@/components/umbrella/logo";
import { ThemeToggle } from "@/components/umbrella/theme-toggle";
import { UserAvatar } from "@/components/umbrella/user-avatar";
import { useUmbrella } from "@/lib/umbrella-store";
import { cn } from "@/lib/utils";

export function ConversationSidebar({
  activeId,
  collapsed,
  onToggleCollapsed,
}: {
  activeId?: string;
  collapsed: boolean;
  onToggleCollapsed: () => void;
}) {
  const { conversations, user, createConversation, renameConversation, deleteConversation, signOut } =
    useUmbrella();
  const navigate = useNavigate();
  const [query, setQuery] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState("");

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase();
    const sorted = [...conversations].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
    return q ? sorted.filter((c) => c.title.toLowerCase().includes(q)) : sorted;
  }, [conversations, query]);

  const startNew = () => {
    const conversation = createConversation();
    navigate({ to: "/chat/$conversationId", params: { conversationId: conversation.id } });
  };

  if (collapsed) {
    return (
      <aside className="hidden w-14 shrink-0 flex-col items-center gap-2 border-r border-sidebar-border bg-sidebar py-3 md:flex">
        <Link to="/chat" aria-label="Umbrella home" className="mb-1">
          <UmbrellaMark className="h-6 w-6" />
        </Link>
        <Button variant="ghost" size="icon" onClick={onToggleCollapsed} aria-label="Expand sidebar">
          <PanelLeftOpen className="h-4 w-4" />
        </Button>
        <Button variant="ghost" size="icon" onClick={startNew} aria-label="New conversation">
          <MessageSquarePlus className="h-4 w-4" />
        </Button>
        <div className="mt-auto flex flex-col items-center gap-2">
          <ThemeToggle />
          {user && <UserAvatar name={user.name} className="size-7" />}
        </div>
      </aside>
    );
  }

  return (
    <aside className="flex w-72 shrink-0 flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground">
      <div className="flex items-center justify-between px-3 py-3">
        <Link to="/chat">
          <UmbrellaLogo />
        </Link>
        <Button
          variant="ghost"
          size="icon"
          onClick={onToggleCollapsed}
          aria-label="Collapse sidebar"
          className="hidden md:inline-flex"
        >
          <PanelLeftClose className="h-4 w-4" />
        </Button>
      </div>

      <div className="space-y-2 px-3 pb-3">
        <Button onClick={startNew} className="w-full justify-start gap-2">
          <MessageSquarePlus className="h-4 w-4" />
          New conversation
        </Button>
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search conversations"
            className="h-9 pl-8 text-sm"
          />
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
        <p className="px-2 py-2 text-[0.7rem] font-medium uppercase tracking-wider text-muted-foreground">
          History
        </p>
        {filtered.length === 0 && (
          <p className="px-2 py-6 text-center text-xs text-muted-foreground">
            {query ? "No conversations match that search." : "No conversations yet."}
          </p>
        )}
        <ul className="space-y-0.5">
          {filtered.map((conversation) => {
            const isActive = conversation.id === activeId;
            const isEditing = editingId === conversation.id;
            return (
              <li
                key={conversation.id}
                className={cn(
                  "group flex items-center gap-1 rounded-md px-1.5 py-1 transition-colors",
                  isActive ? "bg-sidebar-accent" : "hover:bg-sidebar-accent/60",
                )}
              >
                {isEditing ? (
                  <>
                    <Input
                      autoFocus
                      value={draftTitle}
                      onChange={(e) => setDraftTitle(e.target.value)}
                      onKeyDown={(e) => {
                        if (e.key === "Enter") {
                          renameConversation(conversation.id, draftTitle);
                          setEditingId(null);
                        }
                        if (e.key === "Escape") setEditingId(null);
                      }}
                      className="h-7 text-sm"
                    />
                    <Button
                      variant="ghost"
                      size="icon"
                      className="size-7"
                      aria-label="Save name"
                      onClick={() => {
                        renameConversation(conversation.id, draftTitle);
                        setEditingId(null);
                      }}
                    >
                      <Check className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="size-7"
                      aria-label="Cancel rename"
                      onClick={() => setEditingId(null)}
                    >
                      <X className="h-3.5 w-3.5" />
                    </Button>
                  </>
                ) : (
                  <>
                    <Link
                      to="/chat/$conversationId"
                      params={{ conversationId: conversation.id }}
                      className="min-w-0 flex-1 truncate rounded px-1.5 py-1.5 text-sm"
                      title={conversation.title}
                    >
                      {conversation.title}
                    </Link>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="size-7 opacity-0 transition-opacity group-hover:opacity-100 focus-visible:opacity-100"
                      aria-label={`Rename ${conversation.title}`}
                      onClick={() => {
                        setEditingId(conversation.id);
                        setDraftTitle(conversation.title);
                      }}
                    >
                      <Pencil className="h-3.5 w-3.5" />
                    </Button>
                    <Button
                      variant="ghost"
                      size="icon"
                      className="size-7 text-muted-foreground opacity-0 transition-opacity hover:text-destructive group-hover:opacity-100 focus-visible:opacity-100"
                      aria-label={`Delete ${conversation.title}`}
                      onClick={() => {
                        deleteConversation(conversation.id);
                        if (isActive) navigate({ to: "/chat" });
                      }}
                    >
                      <Trash2 className="h-3.5 w-3.5" />
                    </Button>
                  </>
                )}
              </li>
            );
          })}
        </ul>
      </div>

      <div className="border-t border-sidebar-border p-2">
        <div className="flex items-center gap-2 rounded-md px-1.5 py-1.5">
          {user && <UserAvatar name={user.name} />}
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium">{user?.name ?? "Guest"}</p>
            <p className="truncate text-xs text-muted-foreground">{user?.role ?? "Not signed in"}</p>
          </div>
          <ThemeToggle />
          <Button
            variant="ghost"
            size="icon"
            aria-label="Sign out"
            onClick={() => {
              signOut();
              navigate({ to: "/" });
            }}
          >
            <LogOut className="h-4 w-4" />
          </Button>
        </div>
      </div>
    </aside>
  );
}