import type { CSSProperties, ReactNode } from "react";
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
import { ArrowDown, ArrowUp, ChevronsUpDown, GripVertical, Search } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
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
import { UserAvatar } from "@/components/umbrella/user-avatar";
import type { User, UserStatus } from "@/lib/umbrella-types";
import { cn } from "@/lib/utils";

export function AdminUserTable({
  users,
  actions,
  emptyMessage,
  searchPlaceholder = "Search users",
  searchValue,
  onSearchChange,
  showSearch = true,
  totalCount,
  selectable = false,
  selectedIds,
  onToggleUser,
  onTogglePage,
  isRowSelectable,
}: {
  users: User[];
  actions: (user: User) => ReactNode;
  emptyMessage: string;
  searchPlaceholder?: string;
  searchValue?: string;
  onSearchChange?: (value: string) => void;
  showSearch?: boolean;
  totalCount?: number;
  selectable?: boolean;
  selectedIds?: Set<string>;
  onToggleUser?: (user: User, selected: boolean) => void;
  onTogglePage?: (users: User[], selected: boolean) => void;
  isRowSelectable?: (user: User) => boolean;
}) {
  const dndId = useId();
  const columns = useMemo<ColumnDef<User>[]>(
    () => [
      ...(selectable
        ? [
            {
              id: "select",
              header: () => {
                const eligibleUsers = users.filter((user) => isRowSelectable?.(user) ?? true);
                const allSelected =
                  eligibleUsers.length > 0 &&
                  eligibleUsers.every((user) => selectedIds?.has(user.id));
                const someSelected = eligibleUsers.some((user) => selectedIds?.has(user.id));
                return (
                  <Checkbox
                    checked={allSelected ? true : someSelected ? "indeterminate" : false}
                    onCheckedChange={(checked) => onTogglePage?.(eligibleUsers, checked === true)}
                    disabled={!eligibleUsers.length}
                    aria-label="Select all users on this page"
                  />
                );
              },
              size: 48,
              enableSorting: false,
              cell: ({ row }) => (
                <Checkbox
                  checked={selectedIds?.has(row.original.id) ?? false}
                  onCheckedChange={(checked) => onToggleUser?.(row.original, checked === true)}
                  disabled={isRowSelectable ? !isRowSelectable(row.original) : false}
                  aria-label={`Select ${row.original.full_name || row.original.email}`}
                />
              ),
            } satisfies ColumnDef<User>,
          ]
        : []),
      {
        id: "user",
        accessorFn: (user) => `${user.full_name} ${user.email}`,
        header: "User",
        size: 260,
        cell: ({ row }) => (
          <div className="flex min-w-56 items-center gap-3">
            <UserAvatar name={row.original.full_name} className="size-9" />
            <div className="min-w-0">
              <p className="truncate font-medium text-foreground">{row.original.full_name}</p>
              <p className="truncate text-xs text-muted-foreground">{row.original.email}</p>
            </div>
          </div>
        ),
        sortUndefined: "last",
        sortDescFirst: false,
      },
      {
        id: "institution",
        accessorKey: "institution",
        header: "Institution",
        size: 190,
        cell: ({ row }) => (
          <div className="max-w-48">
            <p className="truncate">{row.original.institution || "Not provided"}</p>
            <p className="truncate text-xs text-muted-foreground">
              {row.original.professional_title || "Biology professional"}
            </p>
          </div>
        ),
      },
      {
        id: "role",
        accessorKey: "role",
        header: "Role",
        size: 120,
        cell: ({ row }) => (
          <Badge variant={row.original.role === "ADMIN" ? "default" : "secondary"}>
            {formatLabel(row.original.role)}
          </Badge>
        ),
      },
      {
        id: "status",
        accessorKey: "status",
        header: "Status",
        size: 150,
        cell: ({ row }) => <StatusBadge status={row.original.status} />,
      },
      {
        id: "created_at",
        accessorKey: "created_at",
        header: "Joined",
        size: 140,
        cell: ({ row }) => <DateCell value={row.original.created_at} />,
      },
      {
        id: "last_login_at",
        accessorKey: "last_login_at",
        header: "Last login",
        size: 140,
        cell: ({ row }) =>
          row.original.last_login_at ? (
            <DateCell value={row.original.last_login_at} />
          ) : (
            <span className="text-muted-foreground">Never</span>
          ),
      },
      {
        id: "actions",
        header: "Actions",
        size: 220,
        enableSorting: false,
        cell: ({ row }) => (
          <div className="flex min-w-48 justify-end gap-1">{actions(row.original)}</div>
        ),
      },
    ],
    [actions, isRowSelectable, onTogglePage, onToggleUser, selectable, selectedIds, users],
  );
  const [sorting, setSorting] = useState<SortingState>([]);
  const [globalFilter, setGlobalFilter] = useState("");
  const [columnOrder, setColumnOrder] = useState<string[]>(() =>
    columns.map((column) => column.id as string),
  );
  const sensors = useSensors(
    useSensor(MouseSensor, { activationConstraint: { distance: 5 } }),
    useSensor(TouchSensor, { activationConstraint: { delay: 150, tolerance: 5 } }),
    useSensor(KeyboardSensor),
  );
  const table = useReactTable({
    data: users,
    columns,
    getRowId: (user) => user.id,
    manualFiltering: searchValue !== undefined,
    columnResizeMode: "onChange",
    getCoreRowModel: getCoreRowModel(),
    getFilteredRowModel: getFilteredRowModel(),
    getSortedRowModel: getSortedRowModel(),
    onSortingChange: setSorting,
    onGlobalFilterChange: setGlobalFilter,
    onColumnOrderChange: setColumnOrder,
    enableSortingRemoval: false,
    state: { sorting, globalFilter: searchValue === undefined ? globalFilter : "", columnOrder },
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
    <div className="admin-table-region">
      <div className="admin-table-toolbar">
        {showSearch ? (
          <div className="relative w-full max-w-sm">
            <Search className="pointer-events-none absolute left-3 top-1/2 size-4 -translate-y-1/2 text-muted-foreground" />
            <Input
              value={searchValue ?? globalFilter}
              onChange={(event) =>
                onSearchChange
                  ? onSearchChange(event.target.value)
                  : setGlobalFilter(event.target.value)
              }
              placeholder={searchPlaceholder}
              aria-label={searchPlaceholder}
              className="h-9 bg-background pl-9 shadow-none"
            />
          </div>
        ) : (
          <span />
        )}
        <p className="text-xs text-muted-foreground">
          {table.getFilteredRowModel().rows.length} of {totalCount ?? users.length}
        </p>
      </div>

      <div className="admin-table-shell">
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
                      <DraggableTableHeader key={header.id} header={header} />
                    ))}
                  </SortableContext>
                </TableRow>
              ))}
            </TableHeader>
            <TableBody>
              {table.getRowModel().rows.length ? (
                table.getRowModel().rows.map((row) => (
                  <TableRow
                    key={row.id}
                    data-state={selectedIds?.has(row.original.id) ? "selected" : undefined}
                    className="h-16 bg-card/45"
                  >
                    {row.getVisibleCells().map((cell) => (
                      <SortableContext
                        key={cell.id}
                        items={columnOrder}
                        strategy={horizontalListSortingStrategy}
                      >
                        <DragAlongCell cell={cell} />
                      </SortableContext>
                    ))}
                  </TableRow>
                ))
              ) : (
                <TableRow>
                  <TableCell colSpan={columns.length} className="h-32 text-center">
                    <p className="font-medium text-foreground">No results</p>
                    <p className="mt-1 text-xs text-muted-foreground">{emptyMessage}</p>
                  </TableCell>
                </TableRow>
              )}
            </TableBody>
          </Table>
        </DndContext>
      </div>
    </div>
  );
}

function DraggableTableHeader({ header }: { header: Header<User, unknown> }) {
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

  if (header.column.id === "select") {
    return (
      <TableHead style={{ width: header.column.getSize() }} className="border-r border-border/60">
        {flexRender(header.column.columnDef.header, header.getContext())}
      </TableHead>
    );
  }

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

function DragAlongCell({ cell }: { cell: Cell<User, unknown> }) {
  const { isDragging, setNodeRef, transform, transition } = useSortable({ id: cell.column.id });
  const style: CSSProperties = {
    opacity: isDragging ? 0.75 : 1,
    position: "relative",
    transform: CSS.Translate.toString(transform),
    transition,
    width: cell.column.getSize(),
    zIndex: isDragging ? 1 : 0,
  };

  if (cell.column.id === "select") {
    return (
      <TableCell style={{ width: cell.column.getSize() }} className="border-r border-border/40">
        {flexRender(cell.column.columnDef.cell, cell.getContext())}
      </TableCell>
    );
  }

  return (
    <TableCell ref={setNodeRef} style={style} className="border-r border-border/40 last:border-r-0">
      {flexRender(cell.column.columnDef.cell, cell.getContext())}
    </TableCell>
  );
}

function StatusBadge({ status }: { status: UserStatus }) {
  return (
    <Badge
      variant="outline"
      className={cn(
        "admin-status-badge",
        status === "ACTIVE" && "is-active",
        (status === "PENDING_APPROVAL" || status === "PENDING_EMAIL") && "is-pending",
        status === "INVITED" && "is-invited",
        (status === "REJECTED" || status === "DISABLED") && "is-blocked",
      )}
    >
      <span aria-hidden="true" />
      {formatLabel(status)}
    </Badge>
  );
}

function DateCell({ value }: { value: string }) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return <span className="text-muted-foreground">Unknown</span>;
  return (
    <span className="text-xs text-muted-foreground" title={date.toLocaleString()}>
      {date.toLocaleDateString(undefined, { month: "short", day: "numeric", year: "numeric" })}
    </span>
  );
}

function formatLabel(value: string) {
  return value
    .toLowerCase()
    .replaceAll("_", " ")
    .replace(/\b\w/g, (character) => character.toUpperCase());
}
