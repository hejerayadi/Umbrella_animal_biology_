import { ChevronLeft, ChevronRight } from "lucide-react";

import { Button } from "@/components/ui/button";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";
import { IconTooltip } from "@/components/umbrella/icon-tooltip";
import { cn } from "@/lib/utils";

export function AdminPagination({
  page,
  pageSize,
  total,
  onPageChange,
  onPageSizeChange,
}: {
  page: number;
  pageSize: number;
  total: number;
  onPageChange: (page: number) => void;
  onPageSizeChange?: (pageSize: number) => void;
}) {
  const pageCount = Math.max(1, Math.ceil(total / pageSize));
  const currentPage = Math.min(page, pageCount);
  const first = total ? (currentPage - 1) * pageSize + 1 : 0;
  const last = Math.min(currentPage * pageSize, total);
  const pages = pageNumbers(currentPage, pageCount);

  return (
    <nav className="admin-pagination" aria-label="Table pagination">
      <div className="admin-pagination-summary">
        <span>
          Showing <strong>{first}</strong>-<strong>{last}</strong> of <strong>{total}</strong>
        </span>
        {onPageSizeChange && (
          <Select
            value={String(pageSize)}
            onValueChange={(value) => onPageSizeChange(Number(value))}
          >
            <SelectTrigger className="h-8 w-[76px] bg-background" aria-label="Rows per page">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {[5, 10, 20].map((size) => (
                <SelectItem key={size} value={String(size)}>
                  {size} rows
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        )}
      </div>
      <div className="admin-pagination-controls">
        <IconTooltip label="Previous page" side="top">
          <Button
            type="button"
            variant="outline"
            size="icon"
            className="size-8"
            disabled={currentPage <= 1}
            aria-label="Previous page"
            onClick={() => onPageChange(currentPage - 1)}
          >
            <ChevronLeft />
          </Button>
        </IconTooltip>
        {pages.map((item, index) =>
          item === "ellipsis" ? (
            <span
              key={`ellipsis-${index}`}
              className="admin-pagination-ellipsis"
              aria-hidden="true"
            >
              ...
            </span>
          ) : (
            <Button
              key={item}
              type="button"
              variant={item === currentPage ? "default" : "ghost"}
              size="icon"
              className={cn("size-8", item === currentPage && "shadow-sm")}
              aria-label={`Page ${item}`}
              aria-current={item === currentPage ? "page" : undefined}
              onClick={() => onPageChange(item)}
            >
              {item}
            </Button>
          ),
        )}
        <IconTooltip label="Next page" side="top">
          <Button
            type="button"
            variant="outline"
            size="icon"
            className="size-8"
            disabled={currentPage >= pageCount}
            aria-label="Next page"
            onClick={() => onPageChange(currentPage + 1)}
          >
            <ChevronRight />
          </Button>
        </IconTooltip>
      </div>
    </nav>
  );
}

function pageNumbers(current: number, count: number): Array<number | "ellipsis"> {
  if (count <= 5) return Array.from({ length: count }, (_, index) => index + 1);
  const values = new Set([1, count, current - 1, current, current + 1]);
  const ordered = [...values].filter((page) => page > 0 && page <= count).sort((a, b) => a - b);
  const result: Array<number | "ellipsis"> = [];
  ordered.forEach((page, index) => {
    if (index && page - ordered[index - 1] > 1) result.push("ellipsis");
    result.push(page);
  });
  return result;
}
