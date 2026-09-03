import { ChevronDown, Activity, Check, Loader2, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";

import type { AgentActivity } from "@/lib/umbrella-types";
import { cn } from "@/lib/utils";

/**
 * Reveals `text` a character at a time, restarting whenever it changes.
 *
 * The orchestrator's steps arrive as whole sentences - one HTTP event per
 * decision, not per word - so without this a line would pop into existence
 * fully formed. Typing it out is what makes the panel read as something
 * happening now rather than a log that was printed.
 */
function useTypedText(text: string, enabled: boolean) {
  const [shown, setShown] = useState(enabled ? "" : text);

  useEffect(() => {
    if (!enabled) {
      setShown(text);
      return;
    }
    let index = 0;
    setShown("");
    const interval = setInterval(() => {
      index = Math.min(text.length, index + 2);
      setShown(text.slice(0, index));
      if (index >= text.length) clearInterval(interval);
    }, 18);
    return () => clearInterval(interval);
  }, [text, enabled]);

  return shown;
}

function StepIcon({ status }: { status: AgentActivity["status"] }) {
  if (status === "running") return <Loader2 className="h-3 w-3 animate-spin text-primary" />;
  if (status === "failed") return <X className="h-3 w-3 text-destructive" />;
  return <Check className="h-3 w-3 text-muted-foreground/70" />;
}

/**
 * One line of commentary: what the orchestrator is doing, and - when the model
 * explained itself - why.
 *
 * Only the line still running is typed out. Replaying the animation on every
 * finished step each time the list re-renders would make the whole panel
 * flicker on every new event.
 */
function Step({ activity, typed }: { activity: AgentActivity; typed: boolean }) {
  const description = useTypedText(activity.description, typed);
  const running = activity.status === "running";

  return (
    <li className="flex gap-2.5 py-1">
      <span className="mt-1 shrink-0">
        <StepIcon status={activity.status} />
      </span>
      <span className="min-w-0 flex-1">
        <span
          className={cn(
            "text-xs leading-relaxed",
            running ? "text-foreground" : "text-muted-foreground",
          )}
        >
          {description}
          {running && typed && (
            <span className="ml-0.5 inline-block h-3 w-px translate-y-0.5 animate-pulse bg-primary" />
          )}
        </span>
        {activity.thought && (
          <span className="mt-0.5 block text-xs italic leading-relaxed text-muted-foreground/70">
            {activity.thought}
          </span>
        )}
      </span>
      <span className="shrink-0 font-[family-name:var(--font-mono-custom)] text-[0.65rem] text-muted-foreground/60">
        {new Date(activity.timestamp).toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
          second: "2-digit",
        })}
      </span>
    </li>
  );
}

export function AgentThinkingPanel({
  activities,
  isThinking,
}: {
  activities: AgentActivity[];
  isThinking: boolean;
}) {
  // Open by default while work is in progress: a live commentary nobody can
  // see is the same as no commentary. It stays whatever the user last chose
  // once they touch it, and collapses on its own when the run ends.
  const [openedByUser, setOpenedByUser] = useState<boolean | null>(null);
  const listRef = useRef<HTMLOListElement>(null);

  const open = openedByUser ?? isThinking;
  const active = activities.find((a) => a.status === "running");
  const lastId = activities.length ? activities[activities.length - 1].id : null;

  useEffect(() => {
    if (!isThinking) setOpenedByUser(null);
  }, [isThinking]);

  // Follow the newest line as steps arrive, the way a terminal does.
  useEffect(() => {
    if (open) listRef.current?.scrollTo({ top: listRef.current.scrollHeight, behavior: "smooth" });
  }, [activities.length, open]);

  if (activities.length === 0 && !isThinking) return null;

  const summary = active
    ? active.description
    : isThinking
      ? "Working on it"
      : `${activities.length} ${activities.length === 1 ? "step" : "steps"}`;

  return (
    <div className="overflow-hidden rounded-xl border border-border bg-card/80">
      <button
        type="button"
        onClick={() => setOpenedByUser(!open)}
        aria-expanded={open}
        className="flex w-full items-center gap-2.5 px-3.5 py-2.5 text-left text-xs transition-colors hover:bg-accent/40"
      >
        {isThinking ? (
          <Loader2 className="h-3.5 w-3.5 shrink-0 animate-spin text-primary" />
        ) : (
          <Activity className="h-3.5 w-3.5 shrink-0 text-muted-foreground" />
        )}
        <span className="min-w-0 flex-1 truncate text-muted-foreground">{summary}</span>
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
          <ol
            ref={listRef}
            className="max-h-56 overflow-y-auto border-t border-border px-3.5 py-2.5"
          >
            {activities.map((activity) => (
              <Step
                key={activity.id}
                activity={activity}
                // Only the newest line animates, and only while it is still
                // running - see the note on `Step`.
                typed={activity.id === lastId && activity.status === "running"}
              />
            ))}
            {activities.length === 0 && (
              <li className="py-1 text-xs text-muted-foreground">Reading your question…</li>
            )}
          </ol>
        </div>
      </div>
    </div>
  );
}
