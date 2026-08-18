import type { CSSProperties } from "react";
import { useId, useMemo, useState } from "react";
import {
  closestCenter,
  DndContext,
  KeyboardSensor,
  MouseSensor,
  TouchSensor,
  useSensor,
  useSensors,
  type DragEndEvent,
} from "@dnd-kit/core";
import { restrictToHorizontalAxis } from "@dnd-kit/modifiers";
import {
  arrayMove,
  horizontalListSortingStrategy,
  SortableContext,
  useSortable,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";
import {
  flexRender,
  getCoreRowModel,
  getFilteredRowModel,
  getSortedRowModel,
  useReactTable,
  type Cell,
  type ColumnDef,
  type Header,
  type SortingState,
} from "@tanstack/react-table";
import {
  Activity,
  ArrowDown,
  ArrowUp,
  CheckCircle2,
  ChevronsUpDown,
  CircleAlert,
  Fingerprint,
  GripVertical,
  KeyRound,
  LogIn,
  Search,
  ShieldCheck,
  UserRound,
} from "lucide-react";
import { motion, useReducedMotion } from "framer-motion";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import {
  Table,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from "@/components/ui/table";
import { IconTooltip } from "@/components/umbrella/icon-tooltip";
import { cn } from "@/lib/utils";

const EASE = [0.22, 1, 0.36, 1] as const;

export interface AuditEvent {
  id: string;
  event_type: string;
  outcome: string;
  created_at: string;
  actor_user_id?: string | null;
  target_user_id?: string | null;
  ip_address?: string | null;
  request_id?: string | null;
}

type OutcomeFilter = "ALL" | "SUCCESS" | "EXCEPTION";

export function AdminAuditDashboard({ events }: { events: AuditEvent[] }) {
  const reduceMotion = useReducedMotion();
  const successful = events.filter((event) => event.outcome === "SUCCESS").length;
  const exceptions = events.length - successful;
  const successRate = events.length ? Math.round((successful / events.length) * 100) : 0;
  const uniqueActors = new Set(
    events.flatMap((event) => (event.actor_user_id ? [event.actor_user_id] : [])),
  ).size;
  const cards = [
    {
      label: "Recorded events",
      value: events.length,
      note: "Latest loaded security activity",
      icon: Activity,
      tone: "primary",
    },
    {
      label: "Success rate",
      value: `${successRate}%`,
      note: `${successful} successful operations`,
      icon: CheckCircle2,
      tone: "green",
    },
    {
      label: "Exceptions",
      value: exceptions,
      note: exceptions ? "Review failed or denied actions" : "No exceptions in this window",
      icon: CircleAlert,
      tone: "amber",
    },
    {
      label: "Unique actors",
      value: uniqueActors,
      note: "Authenticated identities represented",
      icon: Fingerprint,
      tone: "blue",
    },
  ];

  return (
    <div className="space-y-7">
      <motion.div
        className="admin-stat-grid"
        initial="hidden"
        animate="visible"
        variants={{ hidden: {}, visible: { transition: { staggerChildren: 0.08 } } }}
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
                {card.value}
              </motion.strong>
              <small>{card.note}</small>
            </div>
          </motion.article>
        ))}
      </motion.div>

      <AuditDataTable events={events} />
    </div>
  );
}

function AuditDataTable({ events }: { events: AuditEvent[] }) {
  const dndId = useId();
  const reduceMotion = useReducedMotion();
  const [globalFilter, setGlobalFilter] = useState("");
  const [outcomeFilter, setOutcomeFilter] = useState<OutcomeFilter>("ALL");
  const [sorting, setSorting] = useState<SortingState>([{ id: "created_at", desc: true }]);
  const filteredEvents = useMemo(
    () =>
      events.filter((event) => {
        if (outcomeFilter === "SUCCESS") return event.outcome === "SUCCESS";
        if (outcomeFilter === "EXCEPTION") return event.outcome !== "SUCCESS";
        return true;
      }),
    [events, outcomeFilter],
  );
  const columns = useMemo<ColumnDef<AuditEvent>[]>(
    () => [
      {
        id: "event_type",
        accessorKey: "event_type",
        header: "Event",
        size: 240,
        cell: ({ row }) => (
          <div className="flex min-w-60 items-center gap-3">
            <span className="admin-audit-event-icon">{eventIcon(row.original.event_type)}</span>
            <div className="min-w-0">
              <p className="truncate font-medium text-foreground">
                {formatEventName(row.original.event_type)}
              </p>
              <p className="truncate font-mono text-[0.68rem] text-muted-foreground">
                {row.original.event_type}
              </p>
            </div>
          </div>
        ),
      },
      {
        id: "outcome",
        accessorKey: "outcome",
        header: "Outcome",
        size: 115,
        cell: ({ row }) => <OutcomeBadge outcome={row.original.outcome} />,
      },
      {
        id: "actor_user_id",
        accessorFn: (event) => event.actor_user_id ?? "system",
        header: "Actor",
        size: 125,
        cell: ({ row }) => <IdentityCell value={row.original.actor_user_id} systemLabel="System" />,
      },
      {
        id: "target_user_id",
        accessorFn: (event) => event.target_user_id ?? "",
        header: "Target",
        size: 125,
        cell: ({ row }) => <IdentityCell value={row.original.target_user_id} systemLabel="None" />,
      },
      {
        id: "ip_address",
        accessorFn: (event) => event.ip_address ?? "",
        header: "IP address",
        size: 120,
        cell: ({ row }) => (
          <span className="font-mono text-xs text-muted-foreground">
            {row.original.ip_address ?? "Not captured"}
          </span>
        ),
      },
      {
        id: "request_id",
        accessorFn: (event) => event.request_id ?? "",
        header: "Request ID",
        size: 130,
        cell: ({ row }) => (
          <span
            className="block max-w-40 truncate font-mono text-xs text-muted-foreground"
            title={row.original.request_id ?? undefined}
          >
            {row.original.request_id?.slice(0, 12) ?? "Not available"}
          </span>
        ),
      },
      {
        id: "created_at",
        accessorKey: "created_at",
        header: "Date",
        size: 140,
        cell: ({ row }) => <DateTimeCell value={row.original.created_at} />,
      },
    ],
    [],
  );
  const [columnOrder, setColumnOrder] = useState<string[]>(() =>
    columns.map((column) => column.id as string),
  );
  const sensors = useSensors(
    useSensor(MouseSensor, { activationConstraint: { distance: 5 } }),
    useSensor(TouchSensor, { activationConstraint: { delay: 150, tolerance: 5 } }),
    useSensor(KeyboardSensor),
  );
  const table = useReactTable({
    data: filteredEvents,
    columns,
    columnResizeMode: "onChange",
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getSortedRowModel: getSortedRowModel(),
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    onColumnOrderChange: setColumnOrder,
    enableSortingRemoval: false,
    state: { sorting, globalFilter, columnOrder },
  });

  function handleDragEnd(event: DragEndEvent) {
    const { active, over } = event;
    if (!over || active.id === over.id) return;
    setColumnOrder((current) => {
      const oldIndex = current.indexOf(String(active.id));
      const newIndex = current.indexOf(String(over.id));
      return arrayMove(current, oldIndex, newIndex);
    });
  }

  return (
    <section className="admin-table-region" aria-labelledby="audit-log-heading">
      <div className="admin-audit-heading">
        <div>
          <p className="admin-section-kicker">Security ledger</p>
          <h2 id="audit-log-heading">Audit event log</h2>
          <p>Filter and inspect the latest immutable administrative and authentication events.</p>
        </div>
        <Badge variant="outline" className="admin-audit-retention">
          <ShieldCheck /> 365-day retention
        </Badge>
      </div>

      <div className="admin-table-toolbar admin-audit-toolbar">
        <div className="relative w-full max-w-sm">
          <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
          <Input
            value={globalFilter}
            onChange={(event) => setGlobalFilter(event.target.value)}
            placeholder="Search events, identities, IPs, or requests"
            aria-label="Search audit events"
            className="h-9 bg-background pl-9 shadow-none"
          />
        </div>
        <div className="admin-audit-segments" role="group" aria-label="Filter by outcome">
          {(["ALL", "SUCCESS", "EXCEPTION"] as const).map((filter) => (
            <button
              key={filter}
              type="button"
              className={outcomeFilter === filter ? "is-active" : undefined}
              aria-pressed={outcomeFilter === filter}
              onClick={() => setOutcomeFilter(filter)}
            >
              {filter === "ALL" ? "All" : filter === "SUCCESS" ? "Successful" : "Exceptions"}
            </button>
          ))}
        </div>
        <p className="text-xs text-muted-foreground" aria-live="polite">
          {table.getFilteredRowModel().rows.length} of {events.length}
        </p>
      </div>

      <div className="admin-table-shell admin-audit-table-shell">
        <DndContext
          id={dndId}
          collisionDetection={closestCenter}
          modifiers={[restrictToHorizontalAxis]}
          onDragEnd={handleDragEnd}
          sensors={sensors}
        >
          <Table>
            <TableHeader>
              {table.getHeaderGroups().map((headerGroup) => (
                <TableRow key={headerGroup.id} className="bg-muted/45 hover:bg-muted/45">
                  <SortableContext items={columnOrder} strategy={horizontalListSortingStrategy}>
                    {headerGroup.headers.map((header) => (
                      <AuditTableHeader key={header.id} header={header} />
                    ))}
                  </SortableContext>
                </TableRow>
              ))}
            </TableHeader>
            <TableBody>
              {table.getRowModel().rows.length ? (
                table.getRowModel().rows.map((row, index) => (
                  <TableRow
                    key={row.id}
                    className={cn("h-16 bg-card/45", !reduceMotion && "admin-audit-row")}
                    style={{ animationDelay: `${Math.min(index, 12) * 28}ms` }}
                  >
                    {row.getVisibleCells().map((cell) => (
                      <SortableContext
                        key={cell.id}
                        items={columnOrder}
                        strategy={horizontalListSortingStrategy}
                      >
                        <AuditTableCell cell={cell} />
                      </SortableContext>
                    ))}
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={columns.length} className="h-36 text-center">
                    <CircleAlert className="mx-auto size-5 text-primary" />
                    <p className="mt-3 font-medium text-foreground">No matching events</p>
                    <p className="mt-1 text-xs text-muted-foreground">
                      Adjust the search or outcome filter to expand the result set.
                    </p>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </DndContext>
      </div>
    </section>
  );
}

function AuditTableHeader({ header }: { header: Header<AuditEvent, unknown> }) {
  const { attributes, isDragging, listeners, setNodeRef, transform, transition } = useSortable({
    id: header.column.id,
  });
  const style: CSSProperties = {
    opacity: isDragging ? 0.75 : 1,
    position: "relative",
    transform: CSS.Translate.toString(transform),
    transition,
    whiteSpace: "nowrap",
    width: header.column.getSize(),
    zIndex: isDragging ? 2 : 0,
  };
  const sorted = header.column.getIsSorted();

  return (
    <TableHead
      ref={setNodeRef}
      style={style}
      className="relative border-r border-border/60 last:border-r-0"
      aria-sort={sorted === "asc" ? "ascending" : sorted === "desc" ? "descending" : "none"}
    >
      <div className="flex items-center gap-1">
        <IconTooltip label="Move column" side="top">
          <Button
            type="button"
            size="icon"
            variant="ghost"
            className="-ml-1 size-7 cursor-grab active:cursor-grabbing"
            aria-label={`Move ${header.column.id} column`}
            {...attributes}
            {...listeners}
          >
            <GripVertical className="size-3.5 text-muted-foreground" />
          </Button>
        </IconTooltip>
        <span className="min-w-0 flex-1 truncate text-[0.72rem] font-semibold uppercase">
          {header.isPlaceholder
            ? null
            : flexRender(header.column.columnDef.header, header.getContext())}
        </span>
        {header.column.getCanSort() && (
          <IconTooltip label="Sort column" side="top">
            <Button
              type="button"
              size="icon"
              variant="ghost"
              className="size-7"
              onClick={header.column.getToggleSortingHandler()}
              aria-label={`Sort by ${header.column.id}`}
            >
              {sorted === "asc" ? (
                <ArrowUp className="size-3.5" />
              ) : sorted === "desc" ? (
                <ArrowDown className="size-3.5" />
              ) : (
                <ChevronsUpDown className="size-3.5 text-muted-foreground" />
              )}
            </Button>
          </IconTooltip>
        )}
      </div>
    </TableHead>
  );
}

function AuditTableCell({ cell }: { cell: Cell<AuditEvent, unknown> }) {
  const { isDragging, setNodeRef, transform, transition } = useSortable({ id: cell.column.id });
  const style: CSSProperties = {
    opacity: isDragging ? 0.75 : 1,
    position: "relative",
    transform: CSS.Translate.toString(transform),
    transition,
    width: cell.column.getSize(),
    zIndex: isDragging ? 1 : 0,
  };
  return (
    <TableCell ref={setNodeRef} style={style} className="border-r border-border/40 last:border-r-0">
      {flexRender(cell.column.columnDef.cell, cell.getContext())}
    </TableCell>
  );
}

function OutcomeBadge({ outcome }: { outcome: string }) {
  const successful = outcome === "SUCCESS";
  return (
    <Badge
      variant="outline"
      className={cn("admin-audit-outcome", successful ? "is-success" : "is-exception")}
    >
      {successful ? <CheckCircle2 /> : <CircleAlert />}
      {formatLabel(outcome)}
    </Badge>
  );
}

function IdentityCell({ value, systemLabel }: { value?: string | null; systemLabel: string }) {
  return (
    <div className="flex items-center gap-2">
      {value ? (
        <Fingerprint className="size-3.5 text-muted-foreground" />
      ) : (
        <UserRound className="size-3.5 text-muted-foreground" />
      )}
      <span className={cn("text-xs", value && "font-mono")} title={value ?? undefined}>
        {value?.slice(0, 8) ?? systemLabel}
      </span>
    </div>
  );
}

function DateTimeCell({ value }: { value: string }) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return <span className="text-muted-foreground">Unknown</span>;
  return (
    <div title={date.toLocaleString()}>
      <p className="text-xs">{formatRelativeTime(date)}</p>
      <p className="mt-0.5 text-[0.68rem] text-muted-foreground">
        {date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}
      </p>
    </div>
  );
}

function eventIcon(eventType: string) {
  if (eventType.includes("login")) return <LogIn />;
  if (eventType.includes("mfa") || eventType.includes("password")) return <KeyRound />;
  if (eventType.startsWith("admin")) return <ShieldCheck />;
  return <Activity />;
}

function formatEventName(value: string) {
  return value.split(".").map(formatLabel).join(" / ");
}

function formatLabel(value: string) {
  return value
    .toLowerCase()
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}

function formatRelativeTime(date: Date) {
  const elapsed = Date.now() - date.getTime();
  if (elapsed < 60_000) return "Just now";
  if (elapsed < 3_600_000) return `${Math.floor(elapsed / 60_000)}m ago`;
  if (elapsed < 86_400_000) return `${Math.floor(elapsed / 3_600_000)}h ago`;
  return date.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}
