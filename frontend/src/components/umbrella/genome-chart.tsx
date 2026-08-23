import { useMemo, useState } from "react";
import { Dna, Download, Maximize2, Minimize2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import type { GenomeChartSpec } from "@/lib/umbrella-types";
import { cn } from "@/lib/utils";

/** Base pairs as the unit a genomicist would actually say out loud. */
function formatBasePairs(value: number): string {
  if (value >= 1e9) return `${(value / 1e9).toFixed(2)} Gb`;
  if (value >= 1e6) return `${(value / 1e6).toFixed(1)} Mb`;
  if (value >= 1e3) return `${(value / 1e3).toFixed(1)} kb`;
  return `${value} bp`;
}

/**
 * Whether an assembly's chromosome count can be read as the species' karyotype.
 *
 * Only a chromosome-level assembly has actually placed the sequence onto
 * chromosomes. Below that, the count is "how many chromosome-scale pieces this
 * assembly happens to contain" - the polar bear's best assembly is
 * scaffold-level and reports 2, while the animal has 74.
 */
function isChromosomeLevel(level: string | null | undefined): boolean {
  return (level ?? "").toLowerCase() === "chromosome";
}

/**
 * The genome chart under an assistant message, when the Genome Agent drew one.
 *
 * The agent renders its own SVG - a size comparison against reference species,
 * or a chromosome/gene map - and it arrives inline in the response rather than
 * behind a URL, because these are a few kilobytes rather than the ~700 KB of a
 * folium map.
 *
 * It is rendered through an `<img>` data URI rather than injected as markup.
 * The document is built by our own backend, but it interpolates gene names and
 * species labels that came from NCBI, and `<img>` cannot execute script no
 * matter what ends up inside it. That costs nothing here: the chart is static,
 * with no interactivity to lose.
 */
export function GenomeChart({ spec }: { spec: GenomeChartSpec }) {
  const [expanded, setExpanded] = useState(false);

  // Percent-encoded rather than base64: the SVG carries species names, which
  // are not always ASCII, and btoa() throws on anything outside Latin-1.
  const dataUri = useMemo(
    () => `data:image/svg+xml;charset=utf-8,${encodeURIComponent(spec.svg)}`,
    [spec.svg],
  );

  const isComparison = spec.comparisons.length > 0;
  const title = isComparison ? "Genome size comparison" : "Chromosome map";
  const caption = isComparison
    ? `This species measured against ${spec.comparisons.length} reference ${
        spec.comparisons.length === 1 ? "genome" : "genomes"
      }.`
    : "Annotated genes and their positions on the assembly.";

  const countIsKaryotype = isChromosomeLevel(spec.assemblyLevel);

  return (
    <section className="mt-4 overflow-hidden rounded-xl border border-border bg-card">
      <header className="flex flex-wrap items-start justify-between gap-2 border-b border-border px-4 py-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <Dna className="size-4 shrink-0 text-primary" aria-hidden="true" />
            <h3 className="truncate text-sm font-semibold text-card-foreground">
              {spec.speciesName ? <span className="italic">{spec.speciesName}</span> : title}
            </h3>
          </div>
          <p className="mt-0.5 pl-6 text-xs text-muted-foreground">{caption}</p>
        </div>

        <span className="shrink-0 rounded-full bg-accent px-2.5 py-1 text-[0.7rem] font-semibold text-accent-foreground">
          {title}
        </span>
      </header>

      <dl className="flex flex-wrap gap-x-6 gap-y-2 border-b border-border px-4 py-2.5">
        {spec.genomeSizeBp !== null && spec.genomeSizeBp !== undefined && (
          <Stat label="Genome size" value={formatBasePairs(spec.genomeSizeBp)} />
        )}
        {spec.chromosomeCount !== null && spec.chromosomeCount !== undefined && (
          <Stat
            label={countIsKaryotype ? "Chromosomes" : "Chromosomes in assembly"}
            value={String(spec.chromosomeCount)}
          />
        )}
        {spec.assemblyLevel && <Stat label="Assembly level" value={spec.assemblyLevel} />}
        {spec.assemblyId && <Stat label="Assembly" value={spec.assemblyId} mono />}
      </dl>

      {/* The count is only the karyotype on a chromosome-level assembly. Said
          here rather than left to the reader, because the number looks like a
          biological fact and on a scaffold-level assembly it is not one. */}
      {spec.chromosomeCount !== null && spec.chromosomeCount !== undefined && !countIsKaryotype && (
        <p className="border-b border-border bg-accent/60 px-4 py-2 text-xs text-accent-foreground">
          <span className="font-semibold">Assembly figure, not karyotype.</span> This is a{" "}
          {(spec.assemblyLevel ?? "sub-chromosome").toLowerCase()}-level assembly, so the count is
          the chromosome-scale pieces it contains &mdash; not the number of chromosomes the species
          carries.
        </p>
      )}

      {spec.note && (
        <p className="border-b border-border px-4 py-2 text-xs text-muted-foreground">
          {spec.note}
        </p>
      )}

      <div
        className={cn(
          "w-full overflow-auto bg-white p-4 transition-[max-height] duration-200",
          expanded ? "max-h-[42rem]" : "max-h-[24rem]",
        )}
      >
        <img
          src={dataUri}
          alt={
            isComparison
              ? `Bar chart comparing the genome size of ${spec.speciesName ?? "this species"} with reference species`
              : `Map of annotated genes on assembly ${spec.assemblyId ?? "for this species"}`
          }
          className="h-auto w-full max-w-full"
        />
      </div>

      <footer className="flex flex-wrap items-center justify-between gap-2 border-t border-border px-4 py-2.5">
        <p className="min-w-0 text-[0.7rem] text-muted-foreground">Genome Agent &middot; NCBI</p>

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

          {/* Opens the SVG on its own rather than downloading it: the chat is
              sandboxed enough that a download attribute is not reliable, and a
              new tab gives the same "look at it full size" outcome. */}
          <Button asChild variant="ghost" size="sm" className="h-7 gap-1.5 px-2 text-xs">
            <a href={dataUri} target="_blank" rel="noreferrer">
              <Download className="size-3.5" aria-hidden="true" />
              Full size
            </a>
          </Button>
        </div>
      </footer>
    </section>
  );
}

function Stat({ label, value, mono = false }: { label: string; value: string; mono?: boolean }) {
  return (
    <div className="min-w-0">
      <dt className="text-[0.7rem] text-muted-foreground">{label}</dt>
      <dd
        className={cn(
          "text-sm text-card-foreground",
          mono ? "font-mono text-xs" : "font-mono tabular-nums",
        )}
      >
        {value}
      </dd>
    </div>
  );
}
