import { useState } from "react";
import { ExternalLink, Maximize2, Minimize2, MapPin, RotateCcw, TriangleAlert } from "lucide-react";

import { Button } from "@/components/ui/button";
import type { BiodiversityMapSpec, BiodiversitySkill } from "@/lib/umbrella-types";
import { cn } from "@/lib/utils";

type LoadState = "loading" | "ready" | "failed";

const SKILL_LABEL: Record<BiodiversitySkill, string> = {
  distribution: "Species distribution",
  hotspots: "Biodiversity hotspots",
  habitat: "Habitat range",
  migration: "Migration forecast",
  unknown: "Biodiversity map",
};

/** One line saying what the reader is actually looking at on the map. */
const SKILL_CAPTION: Record<BiodiversitySkill, string> = {
  distribution: "Each point is a georeferenced GBIF occurrence record.",
  hotspots: "Cells shaded by diversity; outlines are the ranked clusters.",
  habitat: "Shaded blocks are the species' broad habitat regions.",
  migration: "Green marks observed positions, amber the predicted next one.",
  unknown: "Rendered by the Biodiversity Agent.",
};

/**
 * The interactive map under an assistant message, when the Biodiversity Agent
 * produced one.
 *
 * The map is a self-contained folium document served by the agent itself at
 * `GET /maps/{name}`, so it goes in an iframe rather than being reconstructed
 * with a JS mapping library: the agent already decided the projection, the
 * colour breaks and the legend, and redrawing that here would be a second
 * implementation to keep in step with the first.
 *
 * Three consequences of embedding someone else's document, all handled below:
 *
 *  1. **It is a network fetch of ~700 KB.** `loading="lazy"` keeps a scrolled-
 *     back conversation from pulling every map it ever rendered, and the frame
 *     is only mounted once the message finishes typing (see `chat-message`).
 *  2. **It can fail.** The agent evicts nothing, but it can be restarted, and
 *     a map rendered before that restart is gone. A dead frame is replaced
 *     with a retry rather than left blank.
 *  3. **It does not follow the app's theme.** Folium writes a light document,
 *     so the panel frames it on a neutral surface instead of pretending it
 *     will go dark with everything else.
 */
export function BiodiversityMap({ spec }: { spec: BiodiversityMapSpec }) {
  const [state, setState] = useState<LoadState>("loading");
  const [expanded, setExpanded] = useState(false);
  // Part of the iframe's key, so bumping it forces a fresh element and a
  // genuine re-fetch rather than React reusing the failed one.
  const [attempt, setAttempt] = useState(0);

  const title = spec.speciesName || spec.region || SKILL_LABEL[spec.skill];

  return (
    <section className="mt-4 overflow-hidden rounded-xl border border-border bg-card">
      <header className="flex flex-wrap items-start justify-between gap-2 border-b border-border px-4 py-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <MapPin className="size-4 shrink-0 text-primary" aria-hidden="true" />
            <h3 className="truncate text-sm font-semibold text-card-foreground">
              {/* A binomial is the species' name, so it is set in italics the
                  same way the Recognition panel sets its candidates. */}
              {spec.speciesName ? <span className="italic">{spec.speciesName}</span> : title}
            </h3>
          </div>
          <p className="mt-0.5 pl-6 text-xs text-muted-foreground">{SKILL_CAPTION[spec.skill]}</p>
        </div>

        <span className="shrink-0 rounded-full bg-accent px-2.5 py-1 text-[0.7rem] font-semibold text-accent-foreground">
          {SKILL_LABEL[spec.skill]}
        </span>
      </header>

      {spec.isIllustrative && (
        <p className="border-b border-border bg-accent/60 px-4 py-2 text-xs text-accent-foreground">
          <span className="font-semibold">Illustrative extent.</span> These regions are curated
          placeholders, not measured range polygons.
        </p>
      )}

      {spec.stats.length > 0 && (
        <dl className="flex flex-wrap gap-x-6 gap-y-2 border-b border-border px-4 py-2.5">
          {spec.stats.map((stat) => (
            <div key={stat.label} className="min-w-0">
              <dt className="text-[0.7rem] text-muted-foreground">{stat.label}</dt>
              <dd className="font-mono text-sm tabular-nums text-card-foreground">{stat.value}</dd>
            </div>
          ))}
        </dl>
      )}

      <div
        className={cn(
          "relative w-full bg-muted transition-[height] duration-200",
          expanded ? "h-[38rem]" : "h-[22rem]",
        )}
      >
        {state === "failed" ? (
          <MapUnavailable
            onRetry={() => {
              setState("loading");
              setAttempt((value) => value + 1);
            }}
          />
        ) : (
          <>
            {state === "loading" && <MapSkeleton />}
            <iframe
              key={`${spec.url}#${attempt}`}
              src={spec.url}
              title={`${SKILL_LABEL[spec.skill]} map for ${title}`}
              loading="lazy"
              referrerPolicy="no-referrer"
              /* The document is written by our own agent and served from its
                 own origin, not this one, so `allow-same-origin` scopes it to
                 the agent rather than handing it this page. Leaflet needs both
                 to run its tile layer and popups. */
              sandbox="allow-scripts allow-same-origin allow-popups"
              onLoad={() => setState("ready")}
              onError={() => setState("failed")}
              className={cn(
                "absolute inset-0 size-full border-0 bg-white transition-opacity duration-300",
                state === "ready" ? "opacity-100" : "opacity-0",
              )}
            />
          </>
        )}
      </div>

      <footer className="flex flex-wrap items-center justify-between gap-2 border-t border-border px-4 py-2.5">
        <p className="min-w-0 text-[0.7rem] text-muted-foreground">
          {spec.sourceAgents.length > 0 ? spec.sourceAgents.join(" · ") : "Biodiversity Agent"}
        </p>

        <div className="flex shrink-0 items-center gap-1">
          <Button
            variant="ghost"
            size="sm"
            className="h-7 gap-1.5 px-2 text-xs"
            onClick={() => setExpanded((value) => !value)}
            aria-expanded={expanded}
          >
            {expanded ? (
              <Minimize2 className="size-3.5" aria-hidden="true" />
            ) : (
              <Maximize2 className="size-3.5" aria-hidden="true" />
            )}
            {expanded ? "Shrink" : "Expand"}
          </Button>

          <Button asChild variant="ghost" size="sm" className="h-7 gap-1.5 px-2 text-xs">
            <a href={spec.url} target="_blank" rel="noreferrer">
              <ExternalLink className="size-3.5" aria-hidden="true" />
              Open
            </a>
          </Button>
        </div>
      </footer>
    </section>
  );
}

/** Placeholder while the folium document downloads. */
function MapSkeleton() {
  return (
    <div className="absolute inset-0 flex items-center justify-center bg-muted">
      <div className="flex flex-col items-center gap-2">
        <div className="size-6 animate-spin rounded-full border-2 border-border border-t-primary" />
        <p className="text-xs text-muted-foreground">Loading map…</p>
      </div>
    </div>
  );
}

/**
 * Shown when the frame could not load the map.
 *
 * Names the likely cause rather than saying "something went wrong": the map
 * lives on the agent's own port, so the realistic failure is that the agent is
 * no longer running or was restarted since the answer was written.
 */
function MapUnavailable({ onRetry }: { onRetry: () => void }) {
  return (
    <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-muted px-6 text-center">
      <TriangleAlert className="size-5 text-muted-foreground" aria-hidden="true" />
      <div>
        <p className="text-sm font-medium text-foreground">Map could not be loaded</p>
        <p className="mt-1 text-xs text-muted-foreground">
          The Biodiversity Agent serves this file itself &mdash; it may have been restarted since
          this answer was written.
        </p>
      </div>
      <Button variant="outline" size="sm" className="h-7 gap-1.5 px-2.5 text-xs" onClick={onRetry}>
        <RotateCcw className="size-3.5" aria-hidden="true" />
        Retry
      </Button>
    </div>
  );
}
