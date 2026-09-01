import { useState } from "react";
import { ChevronRight, Sparkles } from "lucide-react";

import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from "@/components/ui/collapsible";
import type {
  ReconstructedGap,
  ReconstructionSelection,
  ReconstructionSpec,
} from "@/lib/umbrella-types";
import { cn } from "@/lib/utils";

/** Coordinates as a genomicist reads them: grouped, never in scientific notation. */
function bp(value: number | null | undefined): string {
  return value === null || value === undefined ? "—" : value.toLocaleString();
}

/**
 * The agent's reason codes, in words.
 *
 * Every one of these is a *refusal*, not a crash - the enum says so itself -
 * and the wording has to carry that. `INSUFFICIENT_GAP_SPANNING_HOMOLOGS`
 * means nothing in the reference database aligned across both flanks, and the
 * correct response is to leave the gap alone; it reads as a finding here, not
 * as an apology.
 */
const REASONS: Record<string, string> = {
  NO_HOMOLOGS_FOUND: "Nothing similar found",
  INSUFFICIENT_GAP_SPANNING_HOMOLOGS: "Nothing spans the gap",
  CONTRADICTORY_ALIGNMENTS: "References disagreed",
  EVIDENCE_RETRIEVAL_FAILED: "Evidence could not be fetched",
  CONFIDENCE_BELOW_THRESHOLD: "Scored too low to accept",
  BIOLOGICAL_VALIDATION_FAILED: "Failed biological checks",
  GAP_TOO_LONG: "Gap too long to bridge",
  BUDGET_EXHAUSTED: "Ran out of budget",
  DEADLINE_EXCEEDED: "Ran out of time",
  NOT_ATTEMPTED: "Not attempted",
};

/**
 * The reason code in words.
 *
 * Keyed by the enum's own SCREAMING_CASE values, because that is what
 * `UnresolvedReason` serialises to - a lowercase key here matches nothing and
 * silently falls through to the raw code, which is how this first shipped.
 *
 * `EVIDENCE_RETRIEVAL_FAILED` is worded as a fetch failure rather than an
 * absence of homologues on purpose: the agent's own enum warns that the two
 * must not be conflated, since the homologues demonstrably exist and only
 * their sequences could not be retrieved.
 */
function reasonLabel(reason: string | null): string {
  if (!reason) return "Not reconstructed";
  return REASONS[reason.toUpperCase()] ?? reason.replace(/_/g, " ").toLowerCase();
}

const STATUS_LABELS: Record<string, string> = {
  completed: "Complete",
  partially_completed: "Partially complete",
  failed: "Failed",
};

/**
 * The Reconstruction Agent's result, shown under the written answer.
 *
 * The thing this panel exists to make legible is the fill *in its place*: seven
 * bases mean nothing on their own, and the written answer can only list them.
 * Here they sit between their real flanks, in the colour that says where they
 * came from.
 *
 * Two distinctions are load-bearing and are carried by colour rather than by
 * caption, because a reader who skims must still get them right:
 *
 *  1. Homology-derived bases (teal) versus model-written bases (violet). Both
 *     are plausible strings of the same length; only one was observed in a real
 *     organism. The agent marks this and the UI must not flatten it.
 *  2. Filled (teal) versus still missing (amber). A panel where every gap looks
 *     alike would let one low-confidence fill stand in for nine untouched gaps.
 */
export function ReconstructionPanel({ spec }: { spec: ReconstructionSpec }) {
  const resolved = spec.gaps.filter((gap) => gap.resolved);
  const pending = spec.gaps.filter((gap) => !gap.resolved);

  return (
    <section className="mt-4 overflow-hidden rounded-xl border border-border bg-card">
      <header className="flex flex-wrap items-start justify-between gap-3 border-b border-border px-4 py-3">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-foreground">Genome reconstruction</h3>
          <p className="mt-0.5 truncate text-xs text-muted-foreground">
            {spec.scientificName ? <em>{spec.scientificName}</em> : "Target sequence"}
            {spec.sequenceAccession && (
              <>
                {" · "}
                <a
                  href={`https://www.ncbi.nlm.nih.gov/nuccore/${spec.sequenceAccession}`}
                  target="_blank"
                  rel="noreferrer"
                  className="font-medium underline decoration-dotted underline-offset-2 hover:text-primary"
                >
                  {spec.sequenceAccession}
                </a>
              </>
            )}
          </p>
        </div>
        <StatusBadge status={spec.status} />
      </header>

      <Tally resolved={spec.resolvedGaps} total={spec.requestedGaps} />

      {resolved.length > 0 && (
        <div className="flex flex-col gap-3 px-4 py-4">
          {resolved.map((gap) => (
            <FilledGap key={gap.gapId} gap={gap} />
          ))}
        </div>
      )}

      {pending.length > 0 && (
        <PendingGaps gaps={pending} anyResolved={resolved.length > 0} />
      )}

      {spec.selection && <SelectionFootnote selection={spec.selection} />}
    </section>
  );
}

/** Resolved against requested, as a bar rather than a fraction to parse. */
function Tally({ resolved, total }: { resolved: number; total: number }) {
  const pct = total > 0 ? Math.round((resolved / total) * 100) : 0;

  return (
    <div className="flex items-center gap-3 border-b border-border px-4 py-3">
      <div className="flex items-baseline gap-1.5">
        <span className="font-mono text-2xl font-bold leading-none tabular-nums text-[var(--seq-fill)]">
          {resolved}
        </span>
        <span className="font-mono text-sm tabular-nums text-muted-foreground">
          / {total}
        </span>
      </div>
      <div className="min-w-0 flex-1">
        <div
          className="h-1.5 overflow-hidden rounded-full bg-muted"
          role="img"
          aria-label={`${resolved} of ${total} gaps reconstructed`}
        >
          <div
            className="h-full rounded-full bg-[var(--seq-fill)] transition-[width] duration-500"
            style={{ width: `${pct}%` }}
          />
        </div>
        <p className="mt-1 text-[0.7rem] text-muted-foreground">
          {resolved === 0
            ? "No gap could be filled from the available evidence."
            : `${resolved === 1 ? "One gap" : `${resolved} gaps`} reconstructed; ${
                total - resolved
              } left in place.`}
        </p>
      </div>
    </div>
  );
}

/**
 * One filled gap, with the fill shown between its real flanks.
 *
 * The flanks are trimmed to the last/first 24 bases. The agent sends 50 each
 * side, which is the right amount for an aligner and far too much to read - at
 * 24 the fill still sits in visible context and the track fits without
 * scrolling on a phone.
 */
function FilledGap({ gap }: { gap: ReconstructedGap }) {
  const [open, setOpen] = useState(false);
  const tone = gap.isModelGenerated ? "var(--seq-model)" : "var(--seq-fill)";
  const toneSoft = gap.isModelGenerated ? "var(--seq-model-soft)" : "var(--seq-fill-soft)";

  const left = (gap.leftFlank ?? "").slice(-24);
  const right = (gap.rightFlank ?? "").slice(0, 24);
  const hasFlanks = left.length > 0 || right.length > 0;

  return (
    <div
      className="overflow-hidden rounded-lg border"
      style={{ borderColor: tone, backgroundColor: toneSoft }}
    >
      <div className="flex flex-wrap items-center justify-between gap-2 px-3 pt-2.5">
        <span className="font-mono text-[0.7rem] font-medium text-foreground">
          {bp(gap.start)}–{bp(gap.end)}
          <span className="ml-2 text-muted-foreground">{bp(gap.lengthBp)} bp gap</span>
        </span>
        <span
          className="inline-flex items-center gap-1 rounded-full px-2 py-0.5 text-[0.65rem] font-semibold text-white"
          style={{ backgroundColor: tone }}
        >
          {gap.isModelGenerated && <Sparkles className="size-3" aria-hidden />}
          {gap.isModelGenerated ? "Model-written" : "From homology"}
        </span>
      </div>

      {/* The sequence track. Horizontally scrollable on its own so a long fill
          never makes the whole message scroll sideways. */}
      <div className="overflow-x-auto px-3 py-3">
        <div className="flex w-max items-stretch font-mono text-[0.78rem] leading-none">
          {hasFlanks && (
            <span className="self-center py-2 tracking-[0.08em] text-muted-foreground">
              {left || "…"}
            </span>
          )}
          <span
            className="mx-1 rounded px-1.5 py-2 font-bold tracking-[0.12em] text-white"
            style={{ backgroundColor: tone }}
          >
            {gap.fill}
          </span>
          {hasFlanks && (
            <span className="self-center py-2 tracking-[0.08em] text-muted-foreground">
              {right || "…"}
            </span>
          )}
        </div>
        <p className="mt-2 text-[0.7rem] text-muted-foreground">
          {hasFlanks
            ? "Flanking bases in grey are the assembly's own; the highlighted bases are proposed."
            : "Proposed bases. The surrounding sequence was not carried into this answer."}
        </p>
      </div>

      <div className="flex flex-wrap items-center gap-x-4 gap-y-1 border-t px-3 py-2" style={{ borderColor: tone }}>
        <Confidence value={gap.confidence} level={gap.confidenceLevel} />
        {gap.fillLengthBp !== null && (
          <Meta label="fill" value={`${gap.fillLengthBp} bp`} />
        )}
        {gap.supportingOrganisms.length > 0 && (
          <Meta label="from" value={gap.supportingOrganisms.join(", ")} italic />
        )}
      </div>

      {gap.scores.length > 0 && (
        <Collapsible open={open} onOpenChange={setOpen}>
          <CollapsibleTrigger
            className="flex w-full items-center gap-1 border-t px-3 py-2 text-left text-[0.7rem] font-medium text-muted-foreground hover:text-foreground"
            style={{ borderColor: tone }}
          >
            <ChevronRight
              className={cn("size-3 transition-transform", open && "rotate-90")}
              aria-hidden
            />
            {open ? "Hide" : "Show"} the evidence behind this score
          </CollapsibleTrigger>
          <CollapsibleContent>
            <dl className="grid grid-cols-2 gap-x-4 gap-y-2 px-3 pb-3 pt-1 sm:grid-cols-3">
              {gap.scores.map((score) => (
                <div key={score.label}>
                  <dt className="text-[0.65rem] text-muted-foreground">{score.label}</dt>
                  <dd className="mt-0.5 flex items-center gap-1.5">
                    {score.value === null ? (
                      // Not "0.00": a signal that was never consulted is not a
                      // signal that scored nothing, and a zero-width bar would
                      // read as the latter.
                      <span className="text-[0.65rem] italic text-muted-foreground">
                        not consulted
                      </span>
                    ) : (
                      <>
                        <div className="h-1 flex-1 overflow-hidden rounded-full bg-muted">
                          <div
                            className="h-full rounded-full"
                            style={{
                              width: `${Math.round(Math.min(1, Math.max(0, score.value)) * 100)}%`,
                              backgroundColor: tone,
                            }}
                          />
                        </div>
                        <span className="font-mono text-[0.65rem] tabular-nums text-foreground">
                          {score.value.toFixed(2)}
                        </span>
                      </>
                    )}
                  </dd>
                </div>
              ))}
            </dl>
            {gap.supportingHits.length > 0 && (
              <p className="px-3 pb-3 text-[0.65rem] text-muted-foreground">
                <span className="text-muted-foreground/70">accessions</span>{" "}
                {gap.supportingHits.map((hit, i) => (
                  <span key={hit}>
                    {i > 0 && ", "}
                    <a
                      href={`https://www.ncbi.nlm.nih.gov/nuccore/${hit}`}
                      target="_blank"
                      rel="noreferrer"
                      className="font-medium text-foreground underline decoration-dotted underline-offset-2 hover:text-primary"
                    >
                      {hit}
                    </a>
                  </span>
                ))}
              </p>
            )}
          </CollapsibleContent>
        </Collapsible>
      )}
    </div>
  );
}

/**
 * The gaps that were left alone, as a compact list.
 *
 * Collapsed when something was filled, because then they are the background to
 * the finding. Open when nothing was, because then they *are* the finding and
 * hiding them behind a toggle would understate the result.
 */
function PendingGaps({
  gaps,
  anyResolved,
}: {
  gaps: ReconstructedGap[];
  anyResolved: boolean;
}) {
  const [open, setOpen] = useState(!anyResolved);

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <CollapsibleTrigger className="flex w-full items-center gap-1.5 border-t border-border px-4 py-2.5 text-left text-xs font-medium text-muted-foreground hover:text-foreground">
        <ChevronRight
          className={cn("size-3.5 transition-transform", open && "rotate-90")}
          aria-hidden
        />
        {gaps.length} {gaps.length === 1 ? "gap" : "gaps"} left in place
      </CollapsibleTrigger>
      <CollapsibleContent>
        <ul className="flex flex-col">
          {gaps.map((gap) => (
            <li
              key={gap.gapId}
              className="flex flex-wrap items-center justify-between gap-x-3 gap-y-1 border-t border-border/60 px-4 py-2"
            >
              <span className="flex items-center gap-2">
                <span
                  className="h-3.5 w-1 shrink-0 rounded-full bg-[var(--seq-gap)]"
                  aria-hidden
                />
                <span className="font-mono text-[0.72rem] text-foreground">
                  {bp(gap.start)}–{bp(gap.end)}
                </span>
                <span className="font-mono text-[0.7rem] text-muted-foreground">
                  {bp(gap.lengthBp)} bp
                </span>
              </span>
              <span className="rounded-full bg-[var(--seq-gap-soft)] px-2 py-0.5 text-[0.65rem] font-medium text-[var(--seq-gap)]">
                {reasonLabel(gap.unresolvedReason)}
              </span>
            </li>
          ))}
        </ul>
      </CollapsibleContent>
    </Collapsible>
  );
}

/**
 * What the gaps above are a sample of.
 *
 * Only rendered when the Genome Agent actually said. Ten gaps with no context
 * reads as "this assembly has ten gaps"; it had thirty on one record of
 * several thousand, and that difference is the whole meaning of the number.
 */
function SelectionFootnote({ selection }: { selection: ReconstructionSelection }) {
  const parts: string[] = [];

  if (selection.gapsSelected !== null && selection.gapsFound !== null) {
    parts.push(
      `${selection.gapsSelected} of ${selection.gapsFound} gaps found on the scanned record`,
    );
  }
  if (selection.recordsInAssembly !== null) {
    parts.push(
      `which is 1 of ${selection.recordsInAssembly.toLocaleString()} records in the assembly`,
    );
  }
  if (selection.assemblyGapBasesBp !== null) {
    const share =
      selection.assemblyGapFraction !== null
        ? ` (${(selection.assemblyGapFraction * 100).toFixed(2)}%)`
        : "";
    parts.push(
      `the assembly reports ${selection.assemblyGapBasesBp.toLocaleString()} unresolved bases in total${share}`,
    );
  }
  if (parts.length === 0) return null;

  return (
    <p className="border-t border-border bg-muted/40 px-4 py-2.5 text-[0.7rem] leading-relaxed text-muted-foreground">
      <span className="font-medium text-foreground">Scope.</span>{" "}
      {parts.join("; ")}.
    </p>
  );
}

/**
 * The confidence, with its own word alongside the number.
 *
 * `0.35` and `LOW` are the same statement, and showing only the number invites
 * a reader to decide for themselves what counts as good. The agent already
 * decided; the panel repeats its verdict.
 */
function Confidence({ value, level }: { value: number | null; level: string | null }) {
  if (value === null && !level) return null;
  const low = (level ?? "").toLowerCase() === "low";

  return (
    <span className="text-[0.7rem]">
      <span className="text-muted-foreground/70">confidence</span>{" "}
      <span className="font-mono font-semibold tabular-nums text-foreground">
        {value === null ? "—" : value.toFixed(2)}
      </span>
      {level && (
        <span
          className={cn(
            "ml-1.5 rounded px-1 py-px text-[0.6rem] font-bold uppercase tracking-wide",
            low
              ? "bg-[var(--seq-gap-soft)] text-[var(--seq-gap)]"
              : "bg-muted text-muted-foreground",
          )}
        >
          {level}
        </span>
      )}
    </span>
  );
}

function Meta({
  label,
  value,
  italic,
}: {
  label: string;
  value: string;
  italic?: boolean;
}) {
  return (
    <span className="text-[0.7rem]">
      <span className="text-muted-foreground/70">{label}</span>{" "}
      <span className={cn("font-medium text-foreground", italic && "italic")}>{value}</span>
    </span>
  );
}

function StatusBadge({ status }: { status: string }) {
  const label = STATUS_LABELS[status] ?? status.replace(/_/g, " ");

  return (
    <span
      className={cn(
        "shrink-0 rounded-full px-2.5 py-1 text-[0.7rem] font-semibold",
        status === "completed" && "bg-[var(--seq-fill)] text-white",
        status === "partially_completed" && "bg-[var(--seq-gap-soft)] text-[var(--seq-gap)]",
        status === "failed" && "bg-destructive text-destructive-foreground",
        !(status in STATUS_LABELS) && "bg-muted text-muted-foreground",
      )}
    >
      {label}
    </span>
  );
}
