import { Link, useNavigate } from "@tanstack/react-router";
import {
  Bell,
  ClipboardCheck,
  LayoutDashboard,
  LogOut,
  MailPlus,
  MessagesSquare,
  PanelLeftClose,
  PanelLeftOpen,
  ScrollText,
  Users,
} from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";

import { Button } from "@/components/ui/button";
import { IconTooltip } from "@/components/umbrella/icon-tooltip";
import { UmbrellaLogo, UmbrellaMark } from "@/components/umbrella/logo";
import { ThemeToggle } from "@/components/umbrella/theme-toggle";
import { UserAvatar } from "@/components/umbrella/user-avatar";
import { useAuth } from "@/lib/auth-context";
import { cn } from "@/lib/utils";

export type AdminView =
  "overview" | "applications" | "users" | "invite" | "notifications" | "audit";

const NAVIGATION: Array<{
  id: AdminView;
  label: string;
  icon: typeof LayoutDashboard;
  countKey?: "applications" | "notifications";
}> = [
  { id: "overview", label: "Overview", icon: LayoutDashboard },
  { id: "applications", label: "Applications", icon: ClipboardCheck, countKey: "applications" },
  { id: "users", label: "Users", icon: Users },
  { id: "invite", label: "Invite biologist", icon: MailPlus },
  { id: "notifications", label: "Notifications", icon: Bell, countKey: "notifications" },
  { id: "audit", label: "Audit trail", icon: ScrollText },
];

export function AdminSidebar({
  activeView,
  collapsed,
  onToggleCollapsed,
  onNavigate,
  counts,
}: {
  activeView: AdminView;
  collapsed: boolean;
  onToggleCollapsed: () => void;
  onNavigate?: () => void;
  counts: { applications: number; notifications: number };
}) {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const reduceMotion = useReducedMotion();

  function signOut() {
    void logout().finally(() => navigate({ to: "/" }));
  }

  return (
    <motion.aside
      className={cn(
        "admin-sidebar flex h-full shrink-0 flex-col border-r border-sidebar-border bg-sidebar text-sidebar-foreground",
        collapsed ? "w-16" : "w-64",
      )}
      initial={reduceMotion ? false : { opacity: 0, x: -18 }}
      animate={{ opacity: 1, x: 0 }}
      transition={{ duration: 0.35, ease: [0.22, 1, 0.36, 1] }}
    >
      <div
        className={cn(
          "flex h-16 items-center border-b border-sidebar-border px-3",
          collapsed ? "justify-center" : "justify-between",
        )}
      >
        {collapsed ? (
          <IconTooltip label="Umbrella administration">
            <Link to="/admin" search={{ view: "overview" }} aria-label="Umbrella administration">
              <UmbrellaMark className="size-7" />
            </Link>
          </IconTooltip>
        ) : (
          <Link to="/admin" search={{ view: "overview" }} aria-label="Umbrella administration">
            <UmbrellaLogo />
          </Link>
        )}
        {!collapsed && (
          <IconTooltip label="Collapse sidebar" side="left">
            <Button
              variant="ghost"
              size="icon"
              className="hidden md:inline-flex"
              onClick={onToggleCollapsed}
              aria-label="Collapse sidebar"
            >
              <PanelLeftClose className="size-4" />
            </Button>
          </IconTooltip>
        )}
      </div>

      {collapsed && (
        <div className="flex justify-center py-2">
          <IconTooltip label="Expand sidebar">
            <Button
              variant="ghost"
              size="icon"
              onClick={onToggleCollapsed}
              aria-label="Expand sidebar"
            >
              <PanelLeftOpen className="size-4" />
            </Button>
          </IconTooltip>
        </div>
      )}

      <nav
        className="min-h-0 flex-1 space-y-1 overflow-y-auto px-2 py-3"
        aria-label="Administration"
      >
        {!collapsed && (
          <p className="px-2 pb-2 text-[0.68rem] font-semibold uppercase text-muted-foreground">
            Management
          </p>
        )}
        {NAVIGATION.map((item) => {
          const Icon = item.icon;
          const active = activeView === item.id;
          const count = item.countKey ? counts[item.countKey] : 0;
          const link = (
            <Link
              to="/admin"
              search={{ view: item.id }}
              onClick={onNavigate}
              aria-current={active ? "page" : undefined}
              className={cn(
                "admin-sidebar-link relative flex h-10 items-center rounded-md text-sm transition-colors",
                collapsed ? "justify-center px-2" : "gap-3 px-3",
                active
                  ? "bg-sidebar-accent font-medium text-sidebar-accent-foreground"
                  : "text-muted-foreground hover:bg-sidebar-accent/65 hover:text-sidebar-accent-foreground",
              )}
            >
              {active && (
                <motion.span
                  layoutId="admin-active-navigation"
                  className="absolute inset-y-2 left-0 w-0.5 bg-primary"
                  transition={{ duration: 0.24 }}
                />
              )}
              <Icon className={cn("size-4 shrink-0", active && "text-primary")} />
              {!collapsed && <span className="min-w-0 flex-1 truncate">{item.label}</span>}
              {!collapsed && count > 0 && (
                <span className="grid min-w-5 place-items-center rounded-full bg-primary px-1.5 py-0.5 text-[0.65rem] font-semibold text-primary-foreground">
                  {count > 99 ? "99+" : count}
                </span>
              )}
              {collapsed && count > 0 && (
                <span className="absolute right-1 top-1 size-1.5 rounded-full bg-primary" />
              )}
            </Link>
          );
          return collapsed ? (
            <IconTooltip key={item.id} label={item.label}>
              {link}
            </IconTooltip>
          ) : (
            <div key={item.id}>{link}</div>
          );
        })}
      </nav>

      <div className="border-t border-sidebar-border p-2">
        {collapsed ? (
          <div className="flex flex-col items-center gap-1">
            <IconTooltip label="Research workspace">
              <Button asChild variant="ghost" size="icon">
                <Link to="/chat" aria-label="Research workspace" onClick={onNavigate}>
                  <MessagesSquare className="size-4" />
                </Link>
              </Button>
            </IconTooltip>
            <IconTooltip label="Change color theme">
              <ThemeToggle />
            </IconTooltip>
            {user && (
              <IconTooltip label={`${user.full_name} · Administrator`}>
                <span className="grid size-9 place-items-center rounded-md">
                  <UserAvatar name={user.full_name} className="size-7" />
                </span>
              </IconTooltip>
            )}
            <IconTooltip label="Sign out">
              <Button variant="ghost" size="icon" aria-label="Sign out" onClick={signOut}>
                <LogOut className="size-4" />
              </Button>
            </IconTooltip>
          </div>
        ) : (
          <>
            <Button asChild variant="ghost" className="w-full justify-start gap-3">
              <Link to="/chat" onClick={onNavigate}>
                <MessagesSquare className="size-4" /> Research workspace
              </Link>
            </Button>
            <div className="mt-2 flex items-center gap-2 rounded-md border-t border-sidebar-border px-1.5 pt-3">
              {user && <UserAvatar name={user.full_name} />}
              <div className="min-w-0 flex-1">
                <p className="truncate text-sm font-medium">{user?.full_name ?? "Administrator"}</p>
                <p className="truncate text-xs text-muted-foreground">{user?.email}</p>
              </div>
              <IconTooltip label="Change color theme" side="top">
                <ThemeToggle />
              </IconTooltip>
              <IconTooltip label="Sign out" side="top">
                <Button variant="ghost" size="icon" aria-label="Sign out" onClick={signOut}>
                  <LogOut className="size-4" />
                </Button>
              </IconTooltip>
            </div>
          </>
        )}
      </div>
    </motion.aside>
  );
}
