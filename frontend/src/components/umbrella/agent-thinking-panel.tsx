import { ChevronDown, Activity, CheckCircle2, Loader2 } from "lucide-react";
import { useState } from "react";

import type { AgentActivity } from "@/lib/umbrella-types";
import { cn } from "@/lib/utils";

export function AgentThinkingPanel({
  activities,
  isThinking,
}: {
  activities: AgentActivity[];
  isThinking: boolean;
}) {
  const [open, setOpen] = useState(false);
  if (activities.length === 0 && !isThinking) return null;

  const active = activities.find((a) => a.status === "running");
  const summary = isThinking
    ? (active?.description ?? "Planning workflow...")
    : `${activities.length} orchestration steps completed`;

  return (
    <div className="overflow-hidden rounded-xl border border-border bg-card/80">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        aria-expanded={open}
        className="flex w-full items-center gap-2.5 px-3.5 py-2.5 text-left text-xs transition-colors hover:bg-accent/40"
      >
        {isThinking ? (
          <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-primary" />
        ) : (
          <Activity className="h-3.5 w-3.5 shrink-0 text-primary" />
        )}
        <span className="font-medium">Agent Thinking</span>
        <span className="min-w-0 flex-1 truncate text-muted-foreground">{summary}</span>
        {active && (
          <span className="hidden rounded-full border border-primary/30 bg-accent px-2 py-0.5 text-[0.65rem] font-medium text-accent-foreground sm:inline">
            {active.agentName}
          </span>
        )}
        <ChevronDown
          className={cn(
            "h-3.5 w-3.5 shrink-0 text-muted-foreground transition-transform duration-300",
            open && "rotate-180",
          )}
        />
      </button>

      <div
        className={cn(
          "grid transition-all duration-300 ease-out",
          open ? "grid-rows-[1fr] opacity-100" : "grid-rows-[0fr] opacity-0",
        )}
      >
        <div className="overflow-hidden">
          <ol className="border-t border-border px-3.5 py-3">
            {activities.map((activity) => (
              <li key={activity.id} className="flex gap-3 py-1.5 text-xs">
                <span className="mt-0.5 shrink-0">
                  {activity.status === "running" ? (
                    <Loader2 className="h-3.5 w-3.5 animate-spin text-primary" />
                  ) : (
                    <CheckCircle2 className="h-3.5 w-3.5 text-muted-foreground" />
                  )}
                </span>
                <span className="min-w-0 flex-1">
                  <span className="font-medium">{activity.agentName}</span>
                  <span className="text-muted-foreground"> — {activity.description}</span>
                </span>
                <span className="shrink-0 font-[family-name:var(--font-mono-custom)] text-[0.65rem] text-muted-foreground">
                  {new Date(activity.timestamp).toLocaleTimeString([], {
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                  })}
                </span>
              </li>
            ))}
            {activities.length === 0 && (
              <li className="py-1.5 text-xs text-muted-foreground">
                Waiting for orchestration events…
              </li>
            )}
          </ol>
        </div>
      </div>
    </div>
  );
}