import { Link, useNavigate } from "@tanstack/react-router";
import {
  Check,
  LogOut,
  MessageSquare,
  MessageSquarePlus,
  PanelLeftClose,
  PanelLeftOpen,
  Pencil,
  Search,
  Shield,
  Trash2,
  X,
} from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useMemo, useState } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { IconTooltip } from "@/components/umbrella/icon-tooltip";
import { UmbrellaLogo, UmbrellaMark } from "@/components/umbrella/logo";
import { ThemeToggle } from "@/components/umbrella/theme-toggle";
import { UserAvatar } from "@/components/umbrella/user-avatar";
import { useAuth } from "@/lib/auth-context";
import { cn } from "@/lib/utils";
import { useUmbrella } from "@/lib/umbrella-store";

const EASE = [0.22, 1, 0.36, 1] as const;

export function ConversationSidebar({
  activeId,
  collapsed,
  onToggleCollapsed,
}: {
  activeId?: string;
  collapsed: boolean;
  onToggleCollapsed: () => void;
}) {
  const { conversations, createConversation, renameConversation, deleteConversation } =
    useUmbrella();
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const reduceMotion = useReducedMotion();
  const [query, setQuery] = useState("");
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draftTitle, setDraftTitle] = useState("");

  const filtered = useMemo(() => {
    const normalizedQuery = query.trim().toLowerCase();
    const sorted = [...conversations].sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
    return normalizedQuery
      ? sorted.filter((conversation) => conversation.title.toLowerCase().includes(normalizedQuery))
      : sorted;
  }, [conversations, query]);

  function startNew() {
    const conversation = createConversation();
    navigate({ to: "/chat/$conversationId", params: { conversationId: conversation.id } });
  }

  function signOut() {
    void logout().finally(() => navigate({ to: "/" }));
  }

  function saveTitle(conversationId: string) {
    renameConversation(conversationId, draftTitle);
    setEditingId(null);
  }

  if (collapsed) {
    return (
      <motion.aside
        className="chat-sidebar-collapsed hidden w-14 shrink-0 flex-col items-center border-r border-sidebar-border bg-sidebar py-3 md:flex"
        initial={reduceMotion ? false : { opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.25 }}
      >
        <IconTooltip label="Umbrella workspace">
          <Link to="/chat" aria-label="Umbrella workspace" className="mb-1 rounded-md p-1">
            <UmbrellaMark className="h-6 w-6" />
          </Link>
        </IconTooltip>
        <IconTooltip label="Expand sidebar">
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggleCollapsed}
            aria-label="Expand sidebar"
          >
            <PanelLeftOpen className="h-4 w-4" />
          </Button>
        </IconTooltip>
        <IconTooltip label="New conversation">
          <Button variant="ghost" size="icon" onClick={startNew} aria-label="New conversation">
            <MessageSquarePlus className="h-4 w-4" />
          </Button>
        </IconTooltip>

        <div className="my-2 h-px w-7 bg-sidebar-border" />
        <nav className="flex min-h-0 flex-1 flex-col items-center gap-1 overflow-y-auto px-1">
          {filtered.slice(0, 8).map((conversation) => (
            <IconTooltip key={conversation.id} label={conversation.title}>
              <Link
                to="/chat/$conversationId"
                params={{ conversationId: conversation.id }}
                aria-label={conversation.title}
                aria-current={conversation.id === activeId ? "page" : undefined}
                className={cn(
                  "relative grid size-9 shrink-0 place-items-center rounded-md text-muted-foreground transition-colors hover:bg-sidebar-accent hover:text-sidebar-accent-foreground",
                  conversation.id === activeId &&
                    "bg-sidebar-accent text-sidebar-accent-foreground before:absolute before:left-0 before:h-4 before:w-0.5 before:bg-primary",
                )}
              >
                <MessageSquare className="h-4 w-4" />
              </Link>
            </IconTooltip>
          ))}
        </nav>

        <div className="mt-auto flex flex-col items-center gap-1 border-t border-sidebar-border pt-2">
          {user?.role === "ADMIN" && (
            <IconTooltip label="Administration">
              <Button asChild variant="ghost" size="icon">
                <Link to="/admin" search={{ view: "overview" }} aria-label="Administration">
                  <Shield className="h-4 w-4" />
                </Link>
              </Button>
            </IconTooltip>
          )}
          <IconTooltip label="Change color theme">
            <ThemeToggle />
          </IconTooltip>
          {user && (
            <IconTooltip label={`${user.full_name} · ${user.role}`}>
              <span className="grid size-9 place-items-center rounded-md">
                <UserAvatar name={user.full_name} className="size-7" />
              </span>
            </IconTooltip>
          )}
          <IconTooltip label="Sign out">
            <Button variant="ghost" size="icon" aria-label="Sign out" onClick={signOut}>
              <LogOut className="h-4 w-4" />
            </Button>
          </IconTooltip>
        </div>
      </motion.aside>
    );
  }

  return (
    <motion.aside
      className="chat-sidebar flex w-72 shrink-0 flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground"
      initial={reduceMotion ? false : { opacity: 0, x: -14 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.35, ease: EASE }}
    >
      <div className="flex items-center justify-between px-3 py-3">
        <Link to="/chat" aria-label="Umbrella workspace">
          <UmbrellaLogo />
        </Link>
        <IconTooltip label="Collapse sidebar" side="left">
          <Button
            variant="ghost"
            size="icon"
            onClick={onToggleCollapsed}
            aria-label="Collapse sidebar"
            className="hidden md:inline-flex"
          >
            <PanelLeftClose className="h-4 w-4" />
          </Button>
        </IconTooltip>
      </div>

      <div className="space-y-2 px-3 pb-3">
        <Button onClick={startNew} className="w-full justify-start gap-2 shadow-sm">
          <MessageSquarePlus className="h-4 w-4" />
          New conversation
        </Button>
        <div className="relative">
          <Search className="pointer-events-none absolute left-2.5 top-1/2 h-3.5 w-3.5 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search conversations"
            aria-label="Search conversations"
            className="h-9 bg-background/70 pl-8 text-sm shadow-none"
          />
        </div>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto px-2 pb-2">
        <div className="flex items-center justify-between px-2 py-2">
          <p className="text-[0.7rem] font-semibold uppercase text-muted-foreground">History</p>
          <span className="text-[0.68rem] text-muted-foreground">{filtered.length}</span>
        </div>
        {filtered.length === 0 && (
          <p className="px-2 py-8 text-center text-xs leading-5 text-muted-foreground">
            {query
              ? "No conversations match that search."
              : "Your research threads will appear here."}
          </p>
        )}
        <motion.ul layout className="space-y-1">
          <AnimatePresence initial={false}>
            {filtered.map((conversation) => {
              const isActive = conversation.id === activeId;
              const isEditing = editingId === conversation.id;
              return (
                <motion.li
                  layout
                  key={conversation.id}
                  initial={reduceMotion ? false : { opacity: 0, y: 6 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={reduceMotion ? undefined : { opacity: 0, x: -12 }}
                  transition={{ duration: 0.2, ease: EASE }}
                  className={cn(
                    "group relative flex min-h-11 items-center gap-1 rounded-md px-1.5 py-1 transition-colors",
                    isActive ? "bg-sidebar-accent" : "hover:bg-sidebar-accent/60",
                  )}
                >
                  {isActive && (
                    <motion.span
                      layoutId="active-conversation"
                      className="absolute inset-y-2 left-0 w-0.5 bg-primary"
                    />
                  )}
                  {isEditing ? (
                    <>
                      <Input
                        autoFocus
                        value={draftTitle}
                        onChange={(event) => setDraftTitle(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Enter") saveTitle(conversation.id);
                          if (event.key === "Escape") setEditingId(null);
                        }}
                        className="h-8 text-sm"
                      />
                      <IconTooltip label="Save name" side="top">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="size-7"
                          aria-label="Save name"
                          onClick={() => saveTitle(conversation.id)}
                        >
                          <Check className="h-3.5 w-3.5" />
                        </Button>
                      </IconTooltip>
                      <IconTooltip label="Cancel rename" side="top">
                        <Button
                          variant="ghost"
                          size="icon"
                          className="size-7"
                          aria-label="Cancel rename"
                          onClick={() => setEditingId(null)}
                        >
                          <X className="h-3.5 w-3.5" />
                        </Button>
                      </IconTooltip>
                    </>
                  ) : (
                    <>
                      <MessageSquare
                        className={cn(
                          "ml-1 h-3.5 w-3.5 shrink-0",
                          isActive ? "text-primary" : "text-muted-foreground",
                        )}
                      />
                      <Link
                        to="/chat/$conversationId"
                        params={{ conversationId: conversation.id }}
                        className="min-w-0 flex-1 px-1.5 py-1"
                        aria-current={isActive ? "page" : undefined}
                      >
                        <span className="block truncate text-sm">{conversation.title}</span>
                        <span className="mt-0.5 block text-[0.66rem] text-muted-foreground">
                          {formatConversationDate(conversation.updatedAt)}
                        </span>
                      </Link>
                      <IconTooltip label="Rename conversation" side="top">
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
                      </IconTooltip>
                      <IconTooltip label="Delete conversation" side="top">
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
                      </IconTooltip>
                    </>
                  )}
                </motion.li>
              );
            })}
          </AnimatePresence>
        </motion.ul>
      </div>

      <div className="border-t border-sidebar-border p-2">
        {user?.role === "ADMIN" && (
          <Button asChild variant="ghost" className="mb-1 w-full justify-start gap-2">
            <Link to="/admin" search={{ view: "overview" }}>
              <Shield className="h-4 w-4" /> Administration
            </Link>
          </Button>
        )}
        <div className="flex items-center gap-2 rounded-md px-1.5 py-1.5">
          {user && <UserAvatar name={user.full_name} />}
          <div className="min-w-0 flex-1">
            <p className="truncate text-sm font-medium">{user?.full_name ?? "Guest"}</p>
            <p className="truncate text-xs text-muted-foreground">
              {user?.role ?? "Not signed in"}
            </p>
          </div>
          <IconTooltip label="Change color theme" side="top">
            <ThemeToggle />
          </IconTooltip>
          <IconTooltip label="Sign out" side="top">
            <Button variant="ghost" size="icon" aria-label="Sign out" onClick={signOut}>
              <LogOut className="h-4 w-4" />
            </Button>
          </IconTooltip>
        </div>
      </div>
    </motion.aside>
  );
}

function formatConversationDate(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Recently updated";
  const elapsed = Date.now() - date.getTime();
  if (elapsed < 60_000) return "Just now";
  if (elapsed < 3_600_000) return `${Math.floor(elapsed / 60_000)}m ago`;
  if (elapsed < 86_400_000) return `${Math.floor(elapsed / 3_600_000)}h ago`;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}
