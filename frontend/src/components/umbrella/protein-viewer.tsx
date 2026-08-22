import { useEffect, useRef, useState } from "react";
import { Boxes, ExternalLink, RotateCcw, TriangleAlert } from "lucide-react";

import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { ProteinViewerSpec } from "@/lib/umbrella-types";
import { cn } from "@/lib/utils";

type LoadState = "loading" | "ready" | "failed";

/** Where a structure can be read about, by the database that published it. */
function sourceUrl(spec: ProteinViewerSpec): string | null {
  const { id, source } = spec.structure;
  if (source === "RCSB_PDB") return `https://www.rcsb.org/structure/${id}`;
  // AlphaFold ids are "AF-<accession>-F1"; its entry pages are keyed by the
  // accession in the middle.
  if (source === "ALPHAFOLD_DB") {
    const accession = id.split("-")[1];
    return accession ? `https://alphafold.ebi.ac.uk/entry/${accession}` : null;
  }
  return null;
}

function sourceLabel(source: string): string {
  if (source === "RCSB_PDB") return "RCSB PDB";
  if (source === "ALPHAFOLD_DB") return "AlphaFold DB";
  return source;
}

/**
 * The 3D structure panel under an assistant message.
 *
 * Mol* is loaded on demand - see `protein-viewer-molstar.ts` - so a
 * conversation that never asks about a protein never downloads it. `attempt`
 * is bumped to retry: it is part of the effect's dependencies, so changing it
 * tears the old plugin down and mounts a fresh one.
 */
export function ProteinViewer({ spec }: { spec: ProteinViewerSpec }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const [state, setState] = useState<LoadState>("loading");
  const [attempt, setAttempt] = useState(0);

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;

    // Guards the async gap. Under StrictMode this effect runs twice, and a
    // conversation can be switched away from mid-load; without this, the
    // second plugin would mount into a container the first still owns.
    let cancelled = false;
    let mounted: { dispose: () => void } | null = null;

    setState("loading");

    void (async () => {
      try {
        const { mountStructureViewer } = await import("./protein-viewer-molstar");
        if (cancelled) return;

        const viewer = await mountStructureViewer(container, spec.structure);
        if (cancelled) {
          viewer.dispose();
          return;
        }
        mounted = viewer;
        setState("ready");
      } catch (error) {
        console.error("Mol* failed to render the structure", error);
        if (!cancelled) setState("failed");
      }
    })();

    return () => {
      cancelled = true;
      mounted?.dispose();
    };
  }, [spec.structure, attempt]);

  const href = sourceUrl(spec);
  const { structure } = spec;
  const predicted = structure.structure_type === "PREDICTED";
  // Only the domains the agent could place in the structure's own numbering
  // are shown as positioned spans; see `ProteinDomain.applicable`.
  const placedDomains = spec.domains.filter((domain) => domain.applicable);

  return (
    <figure className="mt-4 overflow-hidden rounded-xl border border-border bg-card">
      <figcaption className="flex flex-wrap items-center gap-2 border-b border-border px-3 py-2">
        <Boxes className="size-4 shrink-0 text-muted-foreground" aria-hidden />
        <span className="text-sm font-medium">{structure.id}</span>
        <Badge variant={predicted ? "outline" : "secondary"}>
          {predicted ? "Predicted model" : "Experimental"}
        </Badge>
        <span className="text-xs text-muted-foreground">{sourceLabel(structure.source)}</span>
        {structure.chain_id && (
          <span className="text-xs text-muted-foreground">chain {structure.chain_id}</span>
        )}
        {href && (
          <a
            href={href}
            target="_blank"
            rel="noreferrer"
            className="ml-auto inline-flex items-center gap-1 text-xs text-muted-foreground underline-offset-4 hover:underline"
          >
            Source <ExternalLink className="size-3" aria-hidden />
          </a>
        )}
      </figcaption>

      <div className="relative">
        {/* Mol* measures its canvas from this element, so it needs a real
            height before the plugin mounts - a height that only exists once
            the structure loads would give it a zero-sized viewport. */}
        <div
          ref={containerRef}
          className={cn(
            "h-[22rem] w-full",
            // Mol* positions its viewport absolutely against the nearest
            // positioned ancestor.
            "relative",
            state !== "ready" && "invisible",
          )}
        />

        {state !== "ready" && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 bg-muted/30 px-4 text-center">
            {state === "loading" ? (
              <>
                <div className="size-5 animate-spin rounded-full border-2 border-muted-foreground/30 border-t-muted-foreground" />
                <p className="text-sm text-muted-foreground">
                  Loading {structure.id} from {sourceLabel(structure.source)}…
                </p>
              </>
            ) : (
              <>
                <TriangleAlert className="size-5 text-muted-foreground" aria-hidden />
                <p className="text-sm text-muted-foreground">
                  The 3D structure could not be loaded. {sourceLabel(structure.source)} may be
                  unreachable from this network.
                </p>
                <Button variant="outline" size="sm" onClick={() => setAttempt((n) => n + 1)}>
                  <RotateCcw className="size-3.5" aria-hidden /> Try again
                </Button>
              </>
            )}
          </div>
        )}
      </div>

      {(placedDomains.length > 0 || spec.selections.length > 0) && (
        <div className="flex flex-wrap gap-1.5 border-t border-border px-3 py-2">
          {spec.selections.map((selection, index) => (
            <Badge key={`sel-${index}`} variant="secondary">
              {selection.label}
            </Badge>
          ))}
          {placedDomains.map((domain) => (
            // Keyed by the span as well as the accession: one InterPro entry
            // legitimately matches a protein at several positions, so
            // MC1R alone returns IPR000761 three times over different ranges.
            <Badge
              key={`${domain.accession}-${domain.start}-${domain.end}`}
              variant="outline"
              title={domain.accession}
            >
              {domain.label} · {domain.start}–{domain.end}
            </Badge>
          ))}
        </div>
      )}
    </figure>
  );
}
