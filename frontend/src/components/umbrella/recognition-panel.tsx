import { useState } from "react";
import { ChevronRight } from "lucide-react";

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import type {
  RecognitionCandidate,
  RecognitionProvenance,
  RecognitionResult,
} from "@/lib/umbrella-types";
import { cn } from "@/lib/utils";

/**
 * The Recognition agent's ranked candidates, shown under the written answer.
 *
 * Two things this panel must never do, because the agent goes to real lengths
 * to avoid them and the UI is the last place they can be undone:
 *
 *  1. Present the score as a probability. It is a ranking value over
 *     BioCLIP-2's label set - the agent reports `score_is_probability: false`
 *     and a "% confident" reading would invent a statistic nobody computed.
 *  2. Present fixture data as measurement. When the classifier or the taxonomy
 *     sources ran in mock mode that is disclosed on the panel itself, not
 *     buried in a tooltip.
 */
export function RecognitionPanel({ result }: { result: RecognitionResult }) {
  const { candidates, provenance } = result;
  if (candidates.length === 0 && result.decision !== "not_identified") return null;

  const isMockClassifier = (provenance.recognitionMode ?? "").includes("mock");
  const isMockTaxonomy =
    provenance.gbifMode === "mock" || provenance.ncbiMode === "mock";

  return (
    <section className="mt-4 overflow-hidden rounded-xl border border-border bg-card">
      <header className="flex flex-wrap items-center justify-between gap-2 border-b border-border px-4 py-3">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-card-foreground">
            Species identification
          </h3>
          <p className="mt-0.5 text-xs text-muted-foreground">
            Ranked taxonomic labels with the classifier&rsquo;s own scores.
          </p>
        </div>
        <DecisionBadge decision={result.decision} />
      </header>

      {(isMockClassifier || isMockTaxonomy) && (
        <p className="border-b border-border bg-accent/60 px-4 py-2 text-xs text-accent-foreground">
          <span className="font-semibold">Demonstration data.</span>{" "}
          {isMockClassifier
            ? "These predictions come from a deterministic test fixture, not from BioCLIP-2 inference."
            : "Taxonomic identifiers come from a committed fixture, not from live GBIF or NCBI lookups."}
        </p>
      )}

      {candidates.length > 0 && (
        <ol className="divide-y divide-border">
          {candidates.map((candidate, index) => (
            <CandidateRow
              key={`${candidate.speciesId || candidate.scientificName}-${index}`}
              candidate={candidate}
              position={index + 1}
              isPrimary={index === 0 && result.decision === "identified"}
            />
          ))}
        </ol>
      )}

      {candidates.length === 0 && (
        <p className="px-4 py-6 text-sm text-muted-foreground">
          The classifier returned no taxonomic label strong enough to name a
          species, so none is claimed.
        </p>
      )}

      <p className="border-t border-border px-4 py-2.5 text-xs leading-relaxed text-muted-foreground">
        A score is a <span className="font-medium">ranking value</span> over the
        model&rsquo;s label set &mdash; not a probability, and not a percentage
        confidence.
      </p>

      <SourcesSection provenance={provenance} />
    </section>
  );
}

function CandidateRow({
  candidate,
  position,
  isPrimary,
}: {
  candidate: RecognitionCandidate;
  position: number;
  isPrimary: boolean;
}) {
  // Scores are ranking values on a 0-1 scale, so the bar maps them directly.
  // Clamped because nothing downstream guarantees the range.
  const width = Math.max(0, Math.min(1, candidate.classificationScore)) * 100;

  return (
    <li className={cn("px-4 py-3", isPrimary && "bg-accent/30")}>
      <div className="flex items-baseline justify-between gap-3">
        <div className="flex min-w-0 items-baseline gap-2">
          <span className="shrink-0 text-xs tabular-nums text-muted-foreground">
            #{position}
          </span>
          <span className="truncate text-sm font-semibold italic text-card-foreground">
            {candidate.scientificName}
          </span>
        </div>
        <span className="shrink-0 font-mono text-xs tabular-nums text-muted-foreground">
          {/* Four decimals: enough to separate neighbouring candidates without
              implying precision the ranking does not carry. */}
          {candidate.classificationScore.toFixed(4)}
        </span>
      </div>

      {candidate.commonName && (
        <p className="mt-0.5 pl-6 text-xs text-muted-foreground">
          {candidate.commonName}
        </p>
      )}

      <div
        className="mt-2 ml-6 h-1.5 overflow-hidden rounded-full bg-muted"
        role="img"
        aria-label={`Ranking score ${candidate.classificationScore.toFixed(4)} of 1`}
      >
        <div
          className={cn(
            "h-full rounded-full transition-all",
            isPrimary ? "bg-primary" : "bg-muted-foreground/40",
          )}
          style={{ width: `${width}%` }}
        />
      </div>

      <div className="mt-2 flex flex-wrap items-center gap-x-3 gap-y-1.5 pl-6">
        <TaxonomyBadge status={candidate.taxonomyStatus} />
        {candidate.speciesId && (
          <Meta label="species_id" value={candidate.speciesId} />
        )}
        <Meta
          label="GBIF"
          value={candidate.gbifId}
          href={
            candidate.gbifId
              ? `https://www.gbif.org/species/${candidate.gbifId}`
              : undefined
          }
        />
        <Meta
          label="NCBI"
          value={candidate.ncbiTaxId}
          href={
            candidate.ncbiTaxId
              ? `https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id=${candidate.ncbiTaxId}`
              : undefined
          }
        />
        {candidate.rank && <Meta label="rank" value={candidate.rank} />}
      </div>
    </li>
  );
}

/** One `label value` pair, linked to the source record where there is one. */
function Meta({
  label,
  value,
  href,
}: {
  label: string;
  value: string | number | null | undefined;
  href?: string;
}) {
  // An absent identifier is shown as a dash rather than hidden: "NCBI did not
  // match this taxon" is information, and a missing row would read as though
  // the lookup never happened.
  const shown = value === null || value === undefined || value === "" ? "—" : value;

  return (
    <span className="text-[0.7rem] text-muted-foreground">
      <span className="text-muted-foreground/70">{label}</span>{" "}
      {href ? (
        <a
          href={href}
          target="_blank"
          rel="noreferrer"
          className="font-medium text-foreground underline decoration-dotted underline-offset-2 hover:text-primary"
        >
          {shown}
        </a>
      ) : (
        <span className="font-medium text-foreground">{shown}</span>
      )}
    </span>
  );
}

function DecisionBadge({ decision }: { decision: string }) {
  const label =
    decision === "identified"
      ? "Identified"
      : decision === "uncertain"
        ? "Uncertain"
        : "Not identified";

  return (
    <span
      className={cn(
        "shrink-0 rounded-full px-2.5 py-1 text-[0.7rem] font-semibold",
        decision === "identified" && "bg-primary text-primary-foreground",
        decision === "uncertain" && "bg-accent text-accent-foreground",
        decision === "not_identified" && "bg-muted text-muted-foreground",
      )}
    >
      {label}
    </span>
  );
}

/**
 * Whether both taxonomy sources agreed on this taxon.
 *
 * `mock_verified` is kept visually distinct from `verified` on purpose - the
 * agent uses two different words for them precisely so a fixture match cannot
 * be read as a live one.
 */
function TaxonomyBadge({ status }: { status?: string | null }) {
  if (!status) return null;

  const label = status.replace(/_/g, " ");

  return (
    <span
      className={cn(
        "rounded px-1.5 py-0.5 text-[0.7rem] font-medium",
        status === "verified" && "bg-primary/15 text-primary",
        status === "partial" && "bg-accent text-accent-foreground",
        status !== "verified" && status !== "partial" && "bg-muted text-muted-foreground",
      )}
    >
      {label}
    </span>
  );
}

/** Where the ranking and the identifiers actually came from. */
function SourcesSection({ provenance }: { provenance: RecognitionProvenance }) {
  const [open, setOpen] = useState(false);

  const model = provenance.modelVersion ?? provenance.modelTarget;
  const rows: Array<{ label: string; value: string }> = [];

  if (model) {
    rows.push({
      label: "Classifier",
      value: provenance.recognitionMode?.includes("mock")
        ? `${model} (mock fixture — no inference was run)`
        : model,
    });
  }
  if (provenance.remoteSpaceId) {
    rows.push({
      label: "Inference host",
      value: provenance.remoteSpaceRevision
        ? `${provenance.remoteSpaceId} @ ${provenance.remoteSpaceRevision.slice(0, 10)}`
        : provenance.remoteSpaceId,
    });
  }
  if (provenance.gbifMode) {
    rows.push({
      label: "GBIF",
      value:
        provenance.gbifMode === "real"
          ? "Live GBIF Species API lookup"
          : "Committed fixture (no network call)",
    });
  }
  if (provenance.ncbiMode) {
    rows.push({
      label: "NCBI",
      value:
        provenance.ncbiMode === "real"
          ? "Live NCBI Entrez taxonomy lookup"
          : "Committed fixture (no network call)",
    });
  }
  if (provenance.scoreKind) {
    rows.push({ label: "Score kind", value: provenance.scoreKind.replace(/_/g, " ") });
  }
  if (provenance.reasoningLlmProvider) {
    rows.push({
      label: "Explanation",
      value: `${provenance.reasoningLlmProvider}${
        provenance.reasoningLlmCalls !== null && provenance.reasoningLlmCalls !== undefined
          ? ` · ${provenance.reasoningLlmCalls} call${provenance.reasoningLlmCalls === 1 ? "" : "s"}`
          : ""
      } — phrasing only, it decides nothing scientific`,
    });
  }

  if (rows.length === 0) return null;

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="flex w-full items-center gap-1.5 border-t border-border px-4 py-2.5 text-left text-xs font-medium text-muted-foreground transition-colors hover:text-foreground">
        <ChevronRight
          className={cn("size-3.5 transition-transform", open && "rotate-90")}
          aria-hidden="true"
        />
        Sources
        {provenance.taxonomyDegraded && (
          <span className="ml-1 rounded bg-accent px-1.5 py-0.5 text-[0.65rem] text-accent-foreground">
            partial
          </span>
        )}
      </CollapsibleTrigger>

      <CollapsibleContent>
        <dl className="space-y-1.5 border-t border-border px-4 py-3">
          {rows.map((row) => (
            <div key={row.label} className="flex gap-2 text-xs">
              <dt className="w-28 shrink-0 text-muted-foreground">{row.label}</dt>
              <dd className="min-w-0 flex-1 break-words text-foreground">{row.value}</dd>
            </div>
          ))}
          {provenance.taxonomyDegraded && (
            <p className="pt-1 text-xs text-muted-foreground">
              At least one taxonomy source was unavailable or returned no match.
              A missing identifier is left empty rather than filled in.
            </p>
          )}
        </dl>
      </CollapsibleContent>
    </Collapsible>
  );
}
