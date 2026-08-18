import { Link, createFileRoute, useNavigate } from "@tanstack/react-router";
import {
  Activity,
  AlertTriangle,
  Ban,
  Bell,
  Check,
  CheckCheck,
  ClipboardCheck,
  KeyRound,
  Loader2,
  MailPlus,
  Menu,
  RefreshCw,
  RotateCcw,
  Search,
  ShieldCheck,
  Trash2,
  UserCheck,
  Users,
  X,
} from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useCallback, useEffect, useMemo, useState, type ReactElement } from "react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import {
  AlertDialog,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog";
import { AdminInvitation } from "@/components/umbrella/admin-invitation";
import { AdminAuditDashboard, type AuditEvent } from "@/components/umbrella/admin-audit-dashboard";
import { AdminPagination } from "@/components/umbrella/admin-pagination";
import { AdminSidebar, type AdminView } from "@/components/umbrella/admin-sidebar";
import { AdminStatCards, type AdminStatCard } from "@/components/umbrella/admin-stat-cards";
import { AdminUserTable } from "@/components/umbrella/admin-user-table";
import { IconTooltip } from "@/components/umbrella/icon-tooltip";
import { apiRequest, apiRequestWithMeta } from "@/lib/api-client";
import { useAuth } from "@/lib/auth-context";
import type { User, UserRole, UserStatus } from "@/lib/umbrella-types";
import { cn } from "@/lib/utils";

const ADMIN_VIEWS: AdminView[] = [
  "overview",
  "applications",
  "users",
  "invite",
  "notifications",
  "audit",
];
const EASE = [0.22, 1, 0.36, 1] as const;
const DEFAULT_PAGE_SIZE = 5;

interface Notification {
  id: string;
  kind?: string;
  title: string;
  message: string;
  read_at?: string | null;
  created_at: string;
}

interface AdminStats {
  total_users: number;
  active_users: number;
  active_biologists: number;
  pending_applications: number;
  rejected_applications: number;
  invited_users: number;
  disabled_users: number;
  status_counts: Partial<Record<UserStatus, number>>;
  role_counts: Partial<Record<UserRole, number>>;
  unread_notifications: number;
  total_notifications: number;
  total_audit_events: number;
  failed_audit_events: number;
}

const EMPTY_STATS: AdminStats = {
  total_users: 0,
  active_users: 0,
  active_biologists: 0,
  pending_applications: 0,
  rejected_applications: 0,
  invited_users: 0,
  disabled_users: 0,
  status_counts: {},
  role_counts: {},
  unread_notifications: 0,
  total_notifications: 0,
  total_audit_events: 0,
  failed_audit_events: 0,
};

const VIEW_COPY: Record<AdminView, { title: string; description: string }> = {
  overview: {
    title: "Administration overview",
    description: "Monitor access, account health, and recent security activity.",
  },
  applications: {
    title: "Biologist applications",
    description: "Review and decide who can access the Umbrella research workspace.",
  },
  users: {
    title: "User management",
    description: "Manage roles, account status, sessions, passwords, and MFA.",
  },
  invite: {
    title: "Invite a biologist",
    description: "Issue a secure, time-limited invitation to a research collaborator.",
  },
  notifications: {
    title: "Admin notifications",
    description: "Stay current on access requests and account events requiring attention.",
  },
  audit: {
    title: "Security audit trail",
    description: "Inspect administrative and authentication events across the platform.",
  },
};

export const Route = createFileRoute("/admin")({
  validateSearch: (search: Record<string, unknown>): { view: AdminView } => ({
    view:
      typeof search.view === "string" && ADMIN_VIEWS.includes(search.view as AdminView)
        ? (search.view as AdminView)
        : "overview",
  }),
  head: () => ({
    meta: [
      { title: "Administration - Umbrella Animal BioHub" },
      { name: "description", content: "Umbrella user and security administration dashboard." },
      { name: "robots", content: "noindex" },
    ],
  }),
  component: Admin,
});

function Admin() {
  const { view } = Route.useSearch();
  const { user, loading } = useAuth();
  const navigate = useNavigate();
  const reduceMotion = useReducedMotion();
  const [applications, setApplications] = useState<User[]>([]);
  const [users, setUsers] = useState<User[]>([]);
  const [notifications, setNotifications] = useState<Notification[]>([]);
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [adminStats, setAdminStats] = useState<AdminStats>(EMPTY_STATS);
  const [applicationPage, setApplicationPage] = useState(1);
  const [applicationPageSize, setApplicationPageSize] = useState(DEFAULT_PAGE_SIZE);
  const [applicationTotal, setApplicationTotal] = useState(0);
  const [applicationSearchInput, setApplicationSearchInput] = useState("");
  const [userPage, setUserPage] = useState(1);
  const [userPageSize, setUserPageSize] = useState(DEFAULT_PAGE_SIZE);
  const [userTotal, setUserTotal] = useState(0);
  const [userSearchInput, setUserSearchInput] = useState("");
  const [userRole, setUserRole] = useState<"ALL" | UserRole>("ALL");
  const [userStatus, setUserStatus] = useState<"ALL" | UserStatus>("ALL");
  const [auditPage, setAuditPage] = useState(1);
  const [auditPageSize, setAuditPageSize] = useState(DEFAULT_PAGE_SIZE);
  const [auditTotal, setAuditTotal] = useState(0);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);
  const [mobileOpen, setMobileOpen] = useState(false);
  const [deleteCandidate, setDeleteCandidate] = useState<User | null>(null);
  const [deleting, setDeleting] = useState(false);
  const [selectedUsers, setSelectedUsers] = useState<Map<string, User>>(new Map());
  const [bulkBusy, setBulkBusy] = useState(false);
  const [bulkDeleteOpen, setBulkDeleteOpen] = useState(false);
  const applicationSearch = useDebouncedValue(applicationSearchInput, 300);
  const userSearch = useDebouncedValue(userSearchInput, 300);

  const refresh = useCallback(async () => {
    setBusy(true);
    setError("");
    try {
      const applicationQuery = new URLSearchParams({
        page: String(applicationPage),
        page_size: String(applicationPageSize),
      });
      if (applicationSearch) applicationQuery.set("search", applicationSearch);
      const userQuery = new URLSearchParams({
        page: String(userPage),
        page_size: String(userPageSize),
      });
      if (userSearch) userQuery.set("search", userSearch);
      if (userRole !== "ALL") userQuery.set("role", userRole);
      if (userStatus !== "ALL") userQuery.set("state", userStatus);
      const auditQuery = new URLSearchParams({
        page: String(auditPage),
        page_size: String(auditPageSize),
      });
      const [statsRow, applicationResult, userResult, notificationRows, auditResult] =
        await Promise.all([
          apiRequest<AdminStats>("/api/v1/admin/stats"),
          apiRequestWithMeta<User[]>(`/api/v1/admin/applications?${applicationQuery}`),
          apiRequestWithMeta<User[]>(`/api/v1/admin/users?${userQuery}`),
          apiRequest<Notification[]>("/api/v1/admin/notifications"),
          apiRequestWithMeta<AuditEvent[]>(`/api/v1/admin/audit-events?${auditQuery}`),
        ]);
      setAdminStats(statsRow);
      setApplications(applicationResult.data);
      setApplicationTotal(
        applicationResult.meta.pagination?.total ?? applicationResult.data.length,
      );
      setUsers(userResult.data);
      setUserTotal(userResult.meta.pagination?.total ?? userResult.data.length);
      setNotifications(notificationRows);
      setEvents(auditResult.data);
      setAuditTotal(auditResult.meta.pagination?.total ?? auditResult.data.length);
    } catch (value) {
      setError(value instanceof Error ? value.message : "Admin data could not be loaded.");
    } finally {
      setBusy(false);
    }
  }, [
    applicationPage,
    applicationPageSize,
    applicationSearch,
    auditPage,
    auditPageSize,
    userPage,
    userPageSize,
    userRole,
    userSearch,
    userStatus,
  ]);

  useEffect(() => setApplicationPage(1), [applicationSearch]);
  useEffect(() => setUserPage(1), [userRole, userSearch, userStatus]);

  useEffect(() => {
    if (!loading && (!user || user.role !== "ADMIN")) {
      navigate({ to: "/chat", replace: true });
    } else if (!loading && user) {
      void refresh();
    }
  }, [loading, navigate, refresh, user]);

  const performAction = useCallback(
    async (path: string, body?: object) => {
      setError("");
      try {
        await apiRequest(path, {
          method: "POST",
          body: body ? JSON.stringify(body) : undefined,
        });
        setSelectedUsers(new Map());
        await refresh();
      } catch (value) {
        setError(value instanceof Error ? value.message : "Action failed.");
      }
    },
    [refresh],
  );

  const deleteUser = useCallback(async () => {
    if (!deleteCandidate) return;
    setDeleting(true);
    setError("");
    try {
      await apiRequest(`/api/v1/admin/users/${deleteCandidate.id}`, { method: "DELETE" });
      setSelectedUsers((current) => {
        const next = new Map(current);
        next.delete(deleteCandidate.id);
        return next;
      });
      setDeleteCandidate(null);
      await refresh();
    } catch (value) {
      setError(value instanceof Error ? value.message : "The account could not be deleted.");
    } finally {
      setDeleting(false);
    }
  }, [deleteCandidate, refresh]);

  const selectedIds = useMemo(() => new Set(selectedUsers.keys()), [selectedUsers]);
  const allSelectedDisabled =
    selectedUsers.size > 0 &&
    [...selectedUsers.values()].every((account) => account.status === "DISABLED");

  const toggleSelectedUser = useCallback((account: User, selected: boolean) => {
    setSelectedUsers((current) => {
      const next = new Map(current);
      if (selected) next.set(account.id, account);
      else next.delete(account.id);
      return next;
    });
  }, []);

  const toggleSelectedPage = useCallback((accounts: User[], selected: boolean) => {
    setSelectedUsers((current) => {
      const next = new Map(current);
      for (const account of accounts) {
        if (selected) next.set(account.id, account);
        else next.delete(account.id);
      }
      return next;
    });
  }, []);

  const bulkReactivate = useCallback(async () => {
    if (!selectedUsers.size) return;
    setBulkBusy(true);
    setError("");
    try {
      await apiRequest("/api/v1/admin/bulk-users/reactivate", {
        method: "POST",
        body: JSON.stringify({ user_ids: [...selectedUsers.keys()] }),
      });
      setSelectedUsers(new Map());
      await refresh();
    } catch (value) {
      setError(
        value instanceof Error ? value.message : "Selected accounts could not be reactivated.",
      );
    } finally {
      setBulkBusy(false);
    }
  }, [refresh, selectedUsers]);

  const bulkDelete = useCallback(async () => {
    if (!selectedUsers.size) return;
    setBulkBusy(true);
    setError("");
    try {
      await apiRequest("/api/v1/admin/bulk-users/delete", {
        method: "POST",
        body: JSON.stringify({ user_ids: [...selectedUsers.keys()] }),
      });
      setBulkDeleteOpen(false);
      setSelectedUsers(new Map());
      await refresh();
    } catch (value) {
      setError(value instanceof Error ? value.message : "Selected accounts could not be deleted.");
    } finally {
      setBulkBusy(false);
    }
  }, [refresh, selectedUsers]);

  const pendingApplications = applications.filter(
    (application) => application.status === "PENDING_APPROVAL",
  );
  const unreadNotifications = notifications.filter((notification) => !notification.read_at);
  const stats = {
    totalUsers: adminStats.total_users,
    activeUsers: adminStats.active_users,
    pendingApplications: adminStats.pending_applications,
    unreadNotifications: adminStats.unread_notifications,
  };

  useEffect(() => {
    setApplicationPage((current) =>
      Math.min(current, Math.max(1, Math.ceil(applicationTotal / applicationPageSize))),
    );
  }, [applicationPageSize, applicationTotal]);
  useEffect(() => {
    setUserPage((current) => Math.min(current, Math.max(1, Math.ceil(userTotal / userPageSize))));
  }, [userPageSize, userTotal]);
  useEffect(() => {
    setAuditPage((current) =>
      Math.min(current, Math.max(1, Math.ceil(auditTotal / auditPageSize))),
    );
  }, [auditPageSize, auditTotal]);

  const renderApplicationActions = useCallback(
    (item: User) => (
      <>
        {item.status === "PENDING_APPROVAL" ? (
          <>
            <ActionButton
              title="Approve application"
              onClick={() => void performAction(`/api/v1/admin/applications/${item.id}/approve`)}
            >
              <Check />
            </ActionButton>
            <ActionButton
              title="Reject application"
              danger
              onClick={() => {
                const reason = window.prompt("Provide a rejection reason (minimum 5 characters)");
                if (reason?.trim()) {
                  void performAction(`/api/v1/admin/applications/${item.id}/reject`, {
                    reason: reason.trim(),
                  });
                }
              }}
            >
              <X />
            </ActionButton>
          </>
        ) : (
          <ActionButton
            title="Reopen application"
            onClick={() => void performAction(`/api/v1/admin/applications/${item.id}/reopen`)}
          >
            <RotateCcw />
          </ActionButton>
        )}
      </>
    ),
    [performAction],
  );

  const renderUserActions = useCallback(
    (item: User) => (
      <>
        {item.status === "INVITED" && (
          <ActionButton
            title="Resend invitation"
            onClick={() => void performAction(`/api/v1/admin/invitations/${item.id}/resend`)}
          >
            <MailPlus />
          </ActionButton>
        )}
        {item.status === "DISABLED" ? (
          <ActionButton
            title="Reactivate account"
            onClick={() => void performAction(`/api/v1/admin/users/${item.id}/reactivate`)}
          >
            <UserCheck />
          </ActionButton>
        ) : (
          <ActionButton
            title="Disable account"
            danger
            onClick={() => void performAction(`/api/v1/admin/users/${item.id}/disable`)}
          >
            <Ban />
          </ActionButton>
        )}
        <ActionButton
          title="Revoke active sessions"
          onClick={() => void performAction(`/api/v1/admin/users/${item.id}/revoke-sessions`)}
        >
          <ShieldCheck />
        </ActionButton>
        <ActionButton
          title="Send password reset"
          onClick={() => void performAction(`/api/v1/admin/users/${item.id}/force-password-reset`)}
        >
          <KeyRound />
        </ActionButton>
        {item.role === "ADMIN" && (
          <ActionButton
            title="Reset administrator MFA"
            onClick={() => void performAction(`/api/v1/admin/users/${item.id}/reset-mfa`)}
          >
            <RotateCcw />
          </ActionButton>
        )}
        <ActionButton
          title={item.id === user?.id ? "You cannot delete your own account" : "Delete account"}
          danger
          disabled={item.id === user?.id}
          onClick={() => setDeleteCandidate(item)}
        >
          <Trash2 />
        </ActionButton>
      </>
    ),
    [performAction, user?.id],
  );

  const applicationCards: AdminStatCard[] = [
    {
      label: "Awaiting review",
      value: adminStats.pending_applications,
      note: adminStats.pending_applications ? "Decision required" : "Review queue is clear",
      icon: ClipboardCheck,
      tone: "amber",
    },
    {
      label: "Rejected",
      value: adminStats.rejected_applications,
      note: "Applications currently closed",
      icon: X,
      tone: "red",
    },
    {
      label: "Active biologists",
      value: adminStats.active_biologists,
      note: "Approved research accounts",
      icon: UserCheck,
      tone: "green",
    },
    {
      label: "Approval rate",
      value: `${percentage(
        adminStats.active_biologists,
        adminStats.active_biologists + adminStats.rejected_applications,
      )}%`,
      note: "Active versus completed reviews",
      icon: Activity,
      tone: "blue",
    },
  ];
  const userCards: AdminStatCard[] = [
    {
      label: "Total accounts",
      value: adminStats.total_users,
      note: "Across all roles and states",
      icon: Users,
      tone: "primary",
    },
    {
      label: "Active users",
      value: adminStats.active_users,
      note: `${percentage(adminStats.active_users, adminStats.total_users)}% account health`,
      icon: UserCheck,
      tone: "green",
    },
    {
      label: "Administrators",
      value: adminStats.role_counts.ADMIN ?? 0,
      note: "Protected operational access",
      icon: ShieldCheck,
      tone: "blue",
    },
    {
      label: "Disabled",
      value: adminStats.disabled_users,
      note: "Accounts ready for review",
      icon: Ban,
      tone: "red",
    },
  ];

  const copy = VIEW_COPY[view];
  const sidebarCounts = {
    applications: adminStats.pending_applications,
    notifications: adminStats.unread_notifications,
  };

  return (
    <div className="admin-dashboard flex h-screen overflow-hidden bg-background">
      <div className="hidden md:flex">
        <AdminSidebar
          activeView={view}
          collapsed={sidebarCollapsed}
          onToggleCollapsed={() => setSidebarCollapsed((current) => !current)}
          counts={sidebarCounts}
        />
      </div>

      <AnimatePresence>
        {mobileOpen && (
          <motion.div
            className="fixed inset-0 z-50 flex md:hidden"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
          >
            <button
              type="button"
              className="absolute inset-0 bg-foreground/35"
              aria-label="Close administration menu"
              onClick={() => setMobileOpen(false)}
            />
            <motion.div
              className="relative z-10 h-full"
              initial={{ x: -260 }}
              animate={{ x: 0 }}
              exit={{ x: -260 }}
              transition={{ duration: 0.28, ease: EASE }}
            >
              <AdminSidebar
                activeView={view}
                collapsed={false}
                onToggleCollapsed={() => setMobileOpen(false)}
                onNavigate={() => setMobileOpen(false)}
                counts={sidebarCounts}
              />
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>

      <div className="flex min-w-0 flex-1 flex-col">
        <header className="admin-topbar flex h-16 shrink-0 items-center gap-3 border-b border-border bg-background/90 px-4 backdrop-blur md:px-6">
          <Button
            variant="ghost"
            size="icon"
            className="md:hidden"
            aria-label="Open administration menu"
            onClick={() => setMobileOpen(true)}
          >
            <Menu className="size-4" />
          </Button>
          <div className="min-w-0 flex-1">
            <h1 className="truncate text-base font-semibold md:text-lg">{copy.title}</h1>
            <p className="hidden truncate text-xs text-muted-foreground sm:block">
              {copy.description}
            </p>
          </div>
          <div className="hidden items-center gap-2 text-xs text-muted-foreground sm:flex">
            <span className="size-1.5 rounded-full bg-emerald-500" />
            System operational
          </div>
          <IconTooltip label="Refresh dashboard" side="bottom">
            <Button
              variant="outline"
              size="icon"
              className="size-9 bg-background"
              onClick={() => void refresh()}
              disabled={busy}
              aria-label="Refresh dashboard"
            >
              <RefreshCw className={cn("size-4", busy && "animate-spin")} />
            </Button>
          </IconTooltip>
        </header>

        <main className="min-h-0 flex-1 overflow-y-auto">
          <div className="mx-auto w-full max-w-[1600px] px-4 py-6 md:px-7 md:py-7">
            <AnimatePresence mode="wait">
              {error && (
                <motion.p
                  key={error}
                  role="alert"
                  className="mb-5 border-l-2 border-destructive bg-destructive/5 px-3 py-2 text-sm text-destructive"
                  initial={reduceMotion ? false : { opacity: 0, y: -8 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={{ opacity: 0 }}
                >
                  {error}
                </motion.p>
              )}
            </AnimatePresence>

            <AnimatePresence mode="wait">
              <motion.div
                key={view}
                initial={reduceMotion ? false : { opacity: 0, y: 12 }}
                animate={{ opacity: 1, y: 0 }}
                exit={reduceMotion ? undefined : { opacity: 0, y: -8 }}
                transition={{ duration: 0.35, ease: EASE }}
              >
                {view === "overview" && (
                  <Overview
                    stats={stats}
                    applications={pendingApplications}
                    notifications={unreadNotifications}
                    events={events}
                    users={users}
                    busy={busy}
                  />
                )}
                {view === "applications" && (
                  <div className="space-y-6">
                    <AdminStatCards cards={applicationCards} loading={busy && !applicationTotal} />
                    <AdminUserTable
                      users={applications}
                      actions={renderApplicationActions}
                      emptyMessage="No applications match the current search."
                      searchPlaceholder="Search by name, email, or institution"
                      searchValue={applicationSearchInput}
                      onSearchChange={setApplicationSearchInput}
                      totalCount={applicationTotal}
                    />
                    <AdminPagination
                      page={applicationPage}
                      pageSize={applicationPageSize}
                      total={applicationTotal}
                      onPageChange={setApplicationPage}
                      onPageSizeChange={(size) => {
                        setApplicationPageSize(size);
                        setApplicationPage(1);
                      }}
                    />
                  </div>
                )}
                {view === "users" && (
                  <div className="space-y-6">
                    <AdminStatCards cards={userCards} loading={busy && !userTotal} />
                    <section className="admin-user-filters" aria-labelledby="user-filters-heading">
                      <div className="admin-filter-heading">
                        <div>
                          <p className="admin-section-kicker">Directory controls</p>
                          <h2 id="user-filters-heading">Find and manage accounts</h2>
                        </div>
                        {(userSearchInput || userRole !== "ALL" || userStatus !== "ALL") && (
                          <Button
                            type="button"
                            variant="ghost"
                            size="sm"
                            onClick={() => {
                              setUserSearchInput("");
                              setUserRole("ALL");
                              setUserStatus("ALL");
                            }}
                          >
                            <X /> Clear filters
                          </Button>
                        )}
                      </div>
                      <div className="admin-filter-grid">
                        <div className="relative admin-filter-search">
                          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
                          <Input
                            value={userSearchInput}
                            onChange={(event) => setUserSearchInput(event.target.value)}
                            placeholder="Search name, email, or institution"
                            aria-label="Search users"
                            className="h-10 bg-background pl-9"
                          />
                        </div>
                        <Select
                          value={userRole}
                          onValueChange={(value) => setUserRole(value as "ALL" | UserRole)}
                        >
                          <SelectTrigger className="h-10 bg-background" aria-label="Filter by role">
                            <SelectValue placeholder="All roles" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="ALL">All roles</SelectItem>
                            <SelectItem value="ADMIN">Administrators</SelectItem>
                            <SelectItem value="BIOLOGIST">Biologists</SelectItem>
                          </SelectContent>
                        </Select>
                        <Select
                          value={userStatus}
                          onValueChange={(value) => setUserStatus(value as "ALL" | UserStatus)}
                        >
                          <SelectTrigger
                            className="h-10 bg-background"
                            aria-label="Filter by status"
                          >
                            <SelectValue placeholder="All statuses" />
                          </SelectTrigger>
                          <SelectContent>
                            <SelectItem value="ALL">All statuses</SelectItem>
                            <SelectItem value="ACTIVE">Active</SelectItem>
                            <SelectItem value="DISABLED">Disabled</SelectItem>
                            <SelectItem value="INVITED">Invited</SelectItem>
                            <SelectItem value="PENDING_EMAIL">Pending email</SelectItem>
                            <SelectItem value="PENDING_APPROVAL">Pending approval</SelectItem>
                            <SelectItem value="REJECTED">Rejected</SelectItem>
                          </SelectContent>
                        </Select>
                      </div>
                    </section>

                    <AnimatePresence initial={false}>
                      {selectedUsers.size > 0 && (
                        <motion.div
                          className="admin-bulk-toolbar"
                          initial={reduceMotion ? false : { opacity: 0, y: -8 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: -8 }}
                        >
                          <div>
                            <Badge>{selectedUsers.size} selected</Badge>
                            <span>Selection is preserved while you browse pages.</span>
                          </div>
                          <div>
                            <Button
                              type="button"
                              variant="outline"
                              size="sm"
                              disabled={bulkBusy || !allSelectedDisabled}
                              title={
                                allSelectedDisabled
                                  ? "Reactivate selected accounts"
                                  : "Only disabled accounts can be reactivated together"
                              }
                              onClick={() => void bulkReactivate()}
                            >
                              {bulkBusy ? <Loader2 className="animate-spin" /> : <UserCheck />}
                              Reactivate
                            </Button>
                            <Button
                              type="button"
                              variant="destructive"
                              size="sm"
                              disabled={bulkBusy}
                              onClick={() => setBulkDeleteOpen(true)}
                            >
                              <Trash2 /> Delete selected
                            </Button>
                            <IconTooltip label="Clear selection" side="top">
                              <Button
                                type="button"
                                variant="ghost"
                                size="icon"
                                className="size-8"
                                aria-label="Clear selection"
                                onClick={() => setSelectedUsers(new Map())}
                              >
                                <X />
                              </Button>
                            </IconTooltip>
                          </div>
                        </motion.div>
                      )}
                    </AnimatePresence>

                    <AdminUserTable
                      users={users}
                      actions={renderUserActions}
                      emptyMessage="No users match the current filters."
                      showSearch={false}
                      totalCount={userTotal}
                      selectable
                      selectedIds={selectedIds}
                      onToggleUser={toggleSelectedUser}
                      onTogglePage={toggleSelectedPage}
                      isRowSelectable={(account) => account.id !== user?.id}
                    />
                    <AdminPagination
                      page={userPage}
                      pageSize={userPageSize}
                      total={userTotal}
                      onPageChange={setUserPage}
                      onPageSizeChange={(size) => {
                        setUserPageSize(size);
                        setUserPage(1);
                      }}
                    />
                  </div>
                )}
                {view === "invite" && (
                  <AdminInvitation
                    onSubmit={async (payload) => {
                      await apiRequest("/api/v1/admin/invitations", {
                        method: "POST",
                        body: JSON.stringify(payload),
                      });
                      await refresh();
                    }}
                  />
                )}
                {view === "notifications" && (
                  <NotificationsView
                    notifications={notifications}
                    stats={adminStats}
                    onMarkRead={(notificationId) =>
                      performAction(`/api/v1/admin/notifications/${notificationId}/read`)
                    }
                    onMarkAllRead={() => performAction("/api/v1/admin/notifications/read-all")}
                  />
                )}
                {view === "audit" && (
                  <div className="space-y-6">
                    <AdminAuditDashboard events={events} />
                    <AdminPagination
                      page={auditPage}
                      pageSize={auditPageSize}
                      total={auditTotal}
                      onPageChange={setAuditPage}
                      onPageSizeChange={(size) => {
                        setAuditPageSize(size);
                        setAuditPage(1);
                      }}
                    />
                  </div>
                )}
              </motion.div>
            </AnimatePresence>

            {busy && (users.length > 0 || applications.length > 0) && (
              <div className="mt-5 flex items-center gap-2 text-xs text-muted-foreground">
                <Loader2 className="size-3.5 animate-spin" /> Synchronizing dashboard data
              </div>
            )}
          </div>
        </main>
      </div>

      <AlertDialog
        open={Boolean(deleteCandidate)}
        onOpenChange={(open) => {
          if (!open && !deleting) setDeleteCandidate(null);
        }}
      >
        <AlertDialogContent className="admin-delete-dialog">
          <AlertDialogHeader className="admin-delete-dialog-header">
            <span className="admin-delete-dialog-icon" aria-hidden="true">
              <AlertTriangle />
            </span>
            <div>
              <p className="admin-section-kicker">Permanent action</p>
              <AlertDialogTitle>Delete this account?</AlertDialogTitle>
              <AlertDialogDescription className="admin-delete-dialog-description">
                This will permanently remove the account for{" "}
                <strong>{deleteCandidate?.full_name || deleteCandidate?.email}</strong>.
              </AlertDialogDescription>
            </div>
          </AlertDialogHeader>
          <div className="admin-delete-dialog-summary">
            <p>{deleteCandidate?.email}</p>
            <ul>
              <li>Active sessions and temporary uploads will be revoked.</li>
              <li>The account will no longer be able to sign in.</li>
              <li>A security audit record of this action will remain.</li>
            </ul>
          </div>
          <AlertDialogFooter className="admin-delete-dialog-footer">
            <AlertDialogCancel disabled={deleting}>Keep account</AlertDialogCancel>
            <Button variant="destructive" disabled={deleting} onClick={() => void deleteUser()}>
              {deleting ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Trash2 className="size-4" />
              )}
              {deleting ? "Deleting..." : "Delete account"}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>

      <AlertDialog
        open={bulkDeleteOpen}
        onOpenChange={(open) => {
          if (!bulkBusy) setBulkDeleteOpen(open);
        }}
      >
        <AlertDialogContent className="admin-delete-dialog">
          <AlertDialogHeader className="admin-delete-dialog-header">
            <span className="admin-delete-dialog-icon" aria-hidden="true">
              <AlertTriangle />
            </span>
            <div>
              <p className="admin-section-kicker">Bulk permanent action</p>
              <AlertDialogTitle>Delete {selectedUsers.size} accounts?</AlertDialogTitle>
              <AlertDialogDescription className="admin-delete-dialog-description">
                Every selected account will be permanently removed in one secured operation.
              </AlertDialogDescription>
            </div>
          </AlertDialogHeader>
          <div className="admin-delete-dialog-summary">
            <p>
              {[...selectedUsers.values()]
                .slice(0, 3)
                .map((account) => account.full_name || account.email)
                .join(", ")}
              {selectedUsers.size > 3 ? ` +${selectedUsers.size - 3} more` : ""}
            </p>
            <ul>
              <li>All active sessions and temporary uploads will be revoked.</li>
              <li>Deleted accounts will immediately lose access.</li>
              <li>One audit event will remain for each deleted account.</li>
            </ul>
          </div>
          <AlertDialogFooter className="admin-delete-dialog-footer">
            <AlertDialogCancel disabled={bulkBusy}>Keep accounts</AlertDialogCancel>
            <Button variant="destructive" disabled={bulkBusy} onClick={() => void bulkDelete()}>
              {bulkBusy ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Trash2 className="size-4" />
              )}
              {bulkBusy ? "Deleting..." : "Delete selected"}
            </Button>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </div>
  );
}

function Overview({
  stats,
  applications,
  notifications,
  events,
  users,
  busy,
}: {
  stats: {
    totalUsers: number;
    activeUsers: number;
    pendingApplications: number;
    unreadNotifications: number;
  };
  applications: User[];
  notifications: Notification[];
  events: AuditEvent[];
  users: User[];
  busy: boolean;
}) {
  const reduceMotion = useReducedMotion();
  const cards = [
    {
      label: "Total accounts",
      value: stats.totalUsers,
      note: "Admins and biologists",
      icon: Users,
      tone: "primary",
    },
    {
      label: "Active users",
      value: stats.activeUsers,
      note: `${percentage(stats.activeUsers, stats.totalUsers)}% account health`,
      icon: UserCheck,
      tone: "green",
    },
    {
      label: "Pending review",
      value: stats.pendingApplications,
      note: stats.pendingApplications ? "Action required" : "Queue is clear",
      icon: ClipboardCheck,
      tone: "amber",
    },
    {
      label: "Unread alerts",
      value: stats.unreadNotifications,
      note: stats.unreadNotifications ? "New admin activity" : "All caught up",
      icon: Bell,
      tone: "blue",
    },
  ];

  return (
    <div className="space-y-7">
      <motion.div
        className="admin-stat-grid"
        initial="hidden"
        animate="visible"
        variants={{
          hidden: {},
          visible: { transition: { staggerChildren: 0.08 } },
        }}
      >
        {cards.map((card) => (
          <motion.article
            key={card.label}
            className="admin-stat-card"
            variants={{
              hidden: reduceMotion ? {} : { opacity: 0, y: 14, scale: 0.98 },
              visible: { opacity: 1, y: 0, scale: 1 },
            }}
            transition={{ duration: 0.4, ease: EASE }}
            whileHover={reduceMotion ? undefined : { y: -3 }}
          >
            <span className={`admin-stat-icon is-${card.tone}`}>
              <card.icon />
            </span>
            <div>
              <p>{card.label}</p>
              <motion.strong
                key={card.value}
                initial={reduceMotion ? false : { opacity: 0, scale: 0.8 }}
                animate={{ opacity: 1, scale: 1 }}
              >
                {busy && !users.length ? "--" : card.value}
              </motion.strong>
              <small>{card.note}</small>
            </div>
          </motion.article>
        ))}
      </motion.div>

      <div className="admin-overview-grid">
        <section className="admin-dashboard-panel">
          <div className="admin-panel-heading">
            <div>
              <p className="admin-section-kicker">Access queue</p>
              <h2>Applications awaiting review</h2>
            </div>
            <Button asChild variant="ghost" size="sm">
              <Link to="/admin" search={{ view: "applications" }}>
                View all
              </Link>
            </Button>
          </div>
          <div className="admin-queue-list">
            {applications.length ? (
              applications.slice(0, 5).map((application, index) => (
                <motion.div
                  key={application.id}
                  initial={reduceMotion ? false : { opacity: 0, x: -12 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.06, duration: 0.3 }}
                >
                  <span className="admin-queue-index">{String(index + 1).padStart(2, "0")}</span>
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-medium">{application.full_name}</p>
                    <p className="truncate text-xs text-muted-foreground">
                      {application.institution || application.email}
                    </p>
                  </div>
                  <time>{formatRelativeTime(application.created_at)}</time>
                </motion.div>
              ))
            ) : (
              <EmptyPanel icon={<ClipboardCheck />} text="No applications need review." />
            )}
          </div>
        </section>

        <section className="admin-dashboard-panel">
          <div className="admin-panel-heading">
            <div>
              <p className="admin-section-kicker">Security pulse</p>
              <h2>Recent audit activity</h2>
            </div>
            <Button asChild variant="ghost" size="sm">
              <Link to="/admin" search={{ view: "audit" }}>
                View audit
              </Link>
            </Button>
          </div>
          <div className="admin-activity-list">
            {events.length ? (
              events.slice(0, 6).map((event, index) => (
                <motion.div
                  key={event.id}
                  initial={reduceMotion ? false : { opacity: 0, x: 12 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.05, duration: 0.3 }}
                >
                  <span
                    className={cn("admin-activity-dot", event.outcome !== "SUCCESS" && "is-failed")}
                  />
                  <div className="min-w-0 flex-1">
                    <p className="truncate font-mono text-xs">
                      {formatEventName(event.event_type)}
                    </p>
                    <p className="text-xs text-muted-foreground">
                      {event.actor_user_id
                        ? `Actor ${event.actor_user_id.slice(0, 8)}`
                        : "System event"}
                    </p>
                  </div>
                  <time>{formatRelativeTime(event.created_at)}</time>
                </motion.div>
              ))
            ) : (
              <EmptyPanel icon={<Activity />} text="Audit activity will appear here." />
            )}
          </div>
        </section>
      </div>

      <section className="admin-health-band">
        <div>
          <p className="admin-section-kicker">Account health</p>
          <h2>User status distribution</h2>
          <p>Current operational state across the loaded account directory.</p>
        </div>
        <div className="admin-health-metrics">
          {[
            ["Active", users.filter((account) => account.status === "ACTIVE").length, "green"],
            ["Invited", users.filter((account) => account.status === "INVITED").length, "blue"],
            [
              "Pending",
              users.filter((account) => account.status.startsWith("PENDING")).length,
              "amber",
            ],
            [
              "Restricted",
              users.filter((account) => ["REJECTED", "DISABLED"].includes(account.status)).length,
              "red",
            ],
          ].map(([label, value, tone]) => (
            <div key={String(label)}>
              <span>
                <i className={`is-${tone}`} /> {label}
              </span>
              <strong>{value}</strong>
              <div>
                <motion.span
                  className={`is-${tone}`}
                  initial={{ width: 0 }}
                  animate={{ width: `${percentage(Number(value), Math.max(users.length, 1))}%` }}
                  transition={{ duration: 0.7, ease: EASE }}
                />
              </div>
            </div>
          ))}
        </div>
      </section>

      {notifications.length > 0 && (
        <p className="text-xs text-muted-foreground">
          {notifications.length} unread notification{notifications.length === 1 ? "" : "s"}{" "}
          currently require attention.
        </p>
      )}
    </div>
  );
}

function NotificationsView({
  notifications,
  stats,
  onMarkRead,
  onMarkAllRead,
}: {
  notifications: Notification[];
  stats: AdminStats;
  onMarkRead: (notificationId: string) => Promise<void>;
  onMarkAllRead: () => Promise<void>;
}) {
  const [filter, setFilter] = useState<"ALL" | "UNREAD" | "READ">("ALL");
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [pageSize, setPageSize] = useState(DEFAULT_PAGE_SIZE);
  const filtered = useMemo(() => {
    const term = search.trim().toLowerCase();
    return notifications.filter((notification) => {
      if (filter === "UNREAD" && notification.read_at) return false;
      if (filter === "READ" && !notification.read_at) return false;
      if (!term) return true;
      return `${notification.title} ${notification.message} ${notification.kind ?? ""}`
        .toLowerCase()
        .includes(term);
    });
  }, [filter, notifications, search]);
  const visible = filtered.slice((page - 1) * pageSize, page * pageSize);
  const applicationAlerts = notifications.filter(
    (notification) => notification.kind === "BIOLOGIST_APPLICATION",
  ).length;
  const recentAlerts = notifications.filter(
    (notification) => Date.now() - new Date(notification.created_at).getTime() <= 86_400_000,
  ).length;
  const cards: AdminStatCard[] = [
    {
      label: "Unread",
      value: stats.unread_notifications,
      note: stats.unread_notifications ? "Needs your attention" : "Inbox is clear",
      icon: Bell,
      tone: "amber",
    },
    {
      label: "All notifications",
      value: stats.total_notifications,
      note: "Administrative activity",
      icon: Activity,
      tone: "primary",
    },
    {
      label: "Applications",
      value: applicationAlerts,
      note: "Access request alerts loaded",
      icon: ClipboardCheck,
      tone: "blue",
    },
    {
      label: "Last 24 hours",
      value: recentAlerts,
      note: "Recently received activity",
      icon: CheckCheck,
      tone: "green",
    },
  ];

  useEffect(() => setPage(1), [filter, pageSize, search]);
  useEffect(() => {
    setPage((current) => Math.min(current, Math.max(1, Math.ceil(filtered.length / pageSize))));
  }, [filtered.length, pageSize]);

  return (
    <div className="space-y-6">
      <AdminStatCards cards={cards} />
      <section className="admin-notification-center" aria-labelledby="notification-center-heading">
        <div className="admin-notification-toolbar">
          <div>
            <p className="admin-section-kicker">Activity inbox</p>
            <h2 id="notification-center-heading">Notification center</h2>
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={!stats.unread_notifications}
            onClick={() => void onMarkAllRead()}
          >
            <CheckCheck /> Mark all read
          </Button>
        </div>
        <div className="admin-notification-filters">
          <div className="relative w-full max-w-sm">
            <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={search}
              onChange={(event) => setSearch(event.target.value)}
              placeholder="Search notifications"
              aria-label="Search notifications"
              className="h-9 bg-background pl-9"
            />
          </div>
          <div className="admin-audit-segments" role="group" aria-label="Filter notifications">
            {(["ALL", "UNREAD", "READ"] as const).map((option) => (
              <button
                key={option}
                type="button"
                className={cn(filter === option && "is-active")}
                aria-pressed={filter === option}
                onClick={() => setFilter(option)}
              >
                {option === "ALL" ? "All" : option === "UNREAD" ? "Unread" : "Read"}
              </button>
            ))}
          </div>
        </div>
        <div className="admin-notification-list">
          {visible.length ? (
            visible.map((notification, index) => (
              <motion.article
                key={notification.id}
                className={cn(notification.read_at && "is-read")}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: index * 0.04, duration: 0.28 }}
              >
                <span className="admin-notification-icon">
                  {notification.kind === "BIOLOGIST_APPLICATION" ? <ClipboardCheck /> : <Bell />}
                </span>
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <h3>{notification.title}</h3>
                    {!notification.read_at && <Badge>New</Badge>}
                  </div>
                  <p>{notification.message}</p>
                  <time>{formatDateTime(notification.created_at)}</time>
                </div>
                {!notification.read_at && (
                  <Button
                    size="sm"
                    variant="outline"
                    onClick={() => void onMarkRead(notification.id)}
                  >
                    <CheckCheck className="size-4" /> Mark read
                  </Button>
                )}
              </motion.article>
            ))
          ) : (
            <EmptyState
              icon={<Bell />}
              title="No matching notifications"
              description="Try another search or inbox filter."
            />
          )}
        </div>
        <AdminPagination
          page={page}
          pageSize={pageSize}
          total={filtered.length}
          onPageChange={setPage}
          onPageSizeChange={(size) => {
            setPageSize(size);
            setPage(1);
          }}
        />
      </section>
    </div>
  );
}

function ActionButton({
  title,
  danger,
  disabled,
  onClick,
  children,
}: {
  title: string;
  danger?: boolean;
  disabled?: boolean;
  onClick: () => void;
  children: ReactElement;
}) {
  return (
    <IconTooltip label={title} side="top">
      <Button
        size="icon"
        variant="ghost"
        aria-label={title}
        onClick={onClick}
        disabled={disabled}
        className={cn("size-8", danger && "text-destructive hover:text-destructive")}
      >
        <span className="[&>svg]:size-4">{children}</span>
      </Button>
    </IconTooltip>
  );
}

function EmptyPanel({ icon, text }: { icon: ReactElement; text: string }) {
  return (
    <div className="admin-empty-panel">
      {icon}
      <p>{text}</p>
    </div>
  );
}

function EmptyState({
  icon,
  title,
  description,
}: {
  icon: ReactElement;
  title: string;
  description: string;
}) {
  return (
    <div className="flex min-h-80 flex-col items-center justify-center text-center">
      <span className="grid size-12 place-items-center rounded-full bg-muted text-primary [&>svg]:size-5">
        {icon}
      </span>
      <h2 className="mt-4 font-semibold">{title}</h2>
      <p className="mt-1 max-w-sm text-sm text-muted-foreground">{description}</p>
    </div>
  );
}

function useDebouncedValue<T>(value: T, delay: number): T {
  const [debounced, setDebounced] = useState(value);
  useEffect(() => {
    const timeout = window.setTimeout(() => setDebounced(value), delay);
    return () => window.clearTimeout(timeout);
  }, [delay, value]);
  return debounced;
}

function percentage(value: number, total: number) {
  return total ? Math.round((value / total) * 100) : 0;
}

function formatDateTime(value: string) {
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? "Unknown" : date.toLocaleString();
}

function formatRelativeTime(value: string) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "Unknown";
  const elapsed = Date.now() - date.getTime();
  if (elapsed < 60_000) return "Just now";
  if (elapsed < 3_600_000) return `${Math.floor(elapsed / 60_000)}m ago`;
  if (elapsed < 86_400_000) return `${Math.floor(elapsed / 3_600_000)}h ago`;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function formatEventName(value: string) {
  return value.replaceAll(".", " / ").replaceAll("_", " ");
}
