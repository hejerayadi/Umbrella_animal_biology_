import { useMemo, useState } from "react";
import { Check, ChevronRight, Copy } from "lucide-react";

import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import {
  clades,
  layoutTree,
  niceScaleStep,
  parseNewick,
  walkLayout,
  type LaidOutNode,
  type TreeLayout,
} from "@/lib/newick";
import type { EvolutionSpec, SimilarityScore } from "@/lib/umbrella-types";
import { cn } from "@/lib/utils";

/** Geometry of the phylogram, in SVG user units. */
const ROW_HEIGHT = 34;
const LABEL_WIDTH = 168;
const PLOT_WIDTH = 300;
const PAD_LEFT = 10;
const PAD_TOP = 14;
const SCALE_BAR_HEIGHT = 34;

/**
 * Support, as a class of evidence rather than a number.
 *
 * UFBoot is conventionally read at 95: at or above, a split is treated as
 * resolved; below ~70 it is barely evidence at all. Colouring by those bands
 * means a reader who skims the picture still gets the distinction that a
 * reader who reads every number would.
 */
function supportTone(support: number): { line: string; text: string; label: string } {
  // The --seq-* tokens, not --chart-*: the chart ramp is monochromatic (every
  // stop sits near hue 30), so a "strong versus weak" distinction drawn from
  // it is invisible. These carry the meaning they already carry in the
  // reconstruction panel - teal for confirmed, amber for provisional.
  if (support >= 95) return { line: "var(--seq-fill)", text: "text-foreground", label: "strong" };
  if (support >= 70) return { line: "var(--seq-gap)", text: "text-foreground", label: "moderate" };
  return { line: "var(--muted-foreground)", text: "text-muted-foreground", label: "weak" };
}

/** "homo sapiens" -> "Homo sapiens". The agent lower-cases some name fields. */
function titleCase(name: string): string {
  return name.charAt(0).toUpperCase() + name.slice(1);
}

/**
 * The Evolution Agent's result, shown under the written answer.
 *
 * The thing this panel exists to make legible is the *topology*. A Newick
 * string is a complete description of the tree and an unreadable one - the
 * written answer can only transliterate it, and an LLM asked to describe it in
 * prose reliably gets the basal trifurcation wrong, reporting an unrooted
 * tree's three top-level branches as three lineages that diverged together.
 *
 * Two things are therefore load-bearing here:
 *
 *  1. **The tree is drawn, not described.** Branch lengths are real distances
 *     on the x axis, with a scale bar, so "human and chimp are close" is
 *     something the reader sees rather than something they are told.
 *  2. **Rootedness is stated.** IQ-TREE returns an unrooted tree; nothing in
 *     the Newick says so except the shape of the top-level node. The panel
 *     says it in words, because every visual convention for drawing a tree
 *     implies a root that this analysis did not estimate.
 */
export function EvolutionPanel({ spec }: { spec: EvolutionSpec }) {
  const parsed = useMemo(() => {
    if (!spec.newick) return null;
    try {
      return layoutTree(parseNewick(spec.newick));
    } catch {
      // A tree we cannot parse is not drawn. The Newick itself is still
      // offered below, so nothing is lost but the picture.
      return null;
    }
  }, [spec.newick]);

  const supported = useMemo(() => (parsed ? clades(parsed.root.node) : []), [parsed]);

  const hasTree = Boolean(spec.newick);
  const hasNetwork = spec.similarityScores.length > 0;

  return (
    <section className="mt-4 overflow-hidden rounded-xl border border-border bg-card">
      <header className="flex flex-wrap items-start justify-between gap-3 border-b border-border px-4 py-3">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold text-foreground">
            {hasTree && hasNetwork
              ? "Evolutionary analysis"
              : hasTree
                ? "Phylogenetic tree"
                : "Molecular comparison"}
          </h3>
          <p className="mt-0.5 truncate text-xs text-muted-foreground">
            {spec.speciesList.length} species
            {spec.model && (
              <>
                {" · model "}
                <span className="font-mono text-foreground">{spec.model}</span>
              </>
            )}
          </p>
        </div>
        <SupportBadge confidence={spec.overallConfidence} hasTree={hasTree} />
      </header>

      {parsed && (
        <>
          <Phylogram layout={parsed} />
          <Rootedness rooted={parsed.rooted} />
        </>
      )}

      {spec.newick && !parsed && (
        <p className="border-b border-border px-4 py-3 text-xs text-muted-foreground">
          The tree could not be drawn — the Newick below did not parse.
        </p>
      )}

      {supported.length > 0 && <Clades clades={supported} />}

      {hasNetwork && (
        <SimilarityMatrix
          species={spec.speciesList}
          scores={spec.similarityScores}
          groups={spec.speciesGroups}
        />
      )}

      {spec.warnings.length > 0 && <Warnings warnings={spec.warnings} />}

      {spec.newick && <RawNewick newick={spec.newick} />}
    </section>
  );
}

/** Mean branch support, or an explicit "not measured". */
function SupportBadge({ confidence, hasTree }: { confidence: number | null; hasTree: boolean }) {
  // Null is not zero. The agent reports null when UFBoot did not run (fewer
  // than four taxa) or when there was no group separation to measure, and a
  // "0%" badge would report an unmeasured tree as a maximally bad one.
  if (confidence === null) {
    return (
      <span className="shrink-0 rounded-full border border-border px-2.5 py-1 text-xs text-muted-foreground">
        {hasTree ? "Support not measured" : "Confidence not measured"}
      </span>
    );
  }

  const pct = Math.round(confidence * 100);
  const tone = supportTone(pct);

  return (
    <span
      className="shrink-0 rounded-full px-2.5 py-1 text-xs font-medium"
      style={{ backgroundColor: `color-mix(in oklab, ${tone.line} 16%, transparent)` }}
    >
      <span className="tabular-nums text-foreground">{pct}%</span>
      <span className="text-muted-foreground"> mean support</span>
    </span>
  );
}

/**
 * A rectangular phylogram.
 *
 * Drawn as inline SVG rather than with a chart library: a phylogram is a few
 * dozen line segments, and the layout it needs (x from accumulated branch
 * length, y from tip order) is not something a generic charting API expresses.
 */
function Phylogram({ layout }: { layout: TreeLayout }) {
  const rows = layout.tips.length;
  const height = rows * ROW_HEIGHT + PAD_TOP * 2 + SCALE_BAR_HEIGHT;
  const width = PAD_LEFT + PLOT_WIDTH + LABEL_WIDTH;

  const xOf = (value: number) => PAD_LEFT + (value / layout.maxDepth) * PLOT_WIDTH;
  const yOf = (value: number) => PAD_TOP + value * ROW_HEIGHT + ROW_HEIGHT / 2;

  const nodes = walkLayout(layout.root);
  const step = niceScaleStep(layout.maxDepth);
  const scaleY = PAD_TOP + rows * ROW_HEIGHT + 16;

  return (
    <div className="overflow-x-auto px-4 py-3">
      <svg
        viewBox={`0 0 ${width} ${height}`}
        width={width}
        height={height}
        role="img"
        aria-label={`Phylogenetic tree of ${layout.tips.map((tip) => tip.node.name).join(", ")}`}
        className="max-w-full"
      >
        {nodes.map((node, index) => (
          <Branches key={index} node={node} xOf={xOf} yOf={yOf} />
        ))}

        {layout.tips.map((tip, index) => (
          <text
            key={index}
            x={xOf(tip.x) + 8}
            y={yOf(tip.y)}
            dominantBaseline="middle"
            fontSize={12}
            fontStyle="italic"
            className="fill-foreground"
          >
            {tip.node.name}
          </text>
        ))}

        {nodes
          .filter((node) => node.children.length > 0 && node.node.support !== null)
          .map((node, index) => (
            <SupportLabel
              key={index}
              x={xOf(node.x)}
              y={yOf(node.y)}
              support={node.node.support!}
            />
          ))}

        {!layout.cladogram && (
          <ScaleBar
            x={PAD_LEFT}
            y={scaleY}
            step={step}
            width={(step / layout.maxDepth) * PLOT_WIDTH}
          />
        )}
      </svg>
    </div>
  );
}

/**
 * The elbow joining a node to its children: one vertical span across the
 * clade, one horizontal segment per child.
 */
function Branches({
  node,
  xOf,
  yOf,
}: {
  node: LaidOutNode;
  xOf: (value: number) => number;
  yOf: (value: number) => number;
}) {
  if (node.children.length === 0) return null;

  const first = node.children[0];
  const last = node.children[node.children.length - 1];
  const stroke = node.node.support !== null ? supportTone(node.node.support).line : "var(--border)";

  return (
    <g>
      <line
        x1={xOf(node.x)}
        y1={yOf(first.y)}
        x2={xOf(node.x)}
        y2={yOf(last.y)}
        stroke={stroke}
        strokeWidth={1.5}
        strokeLinecap="round"
      />
      {node.children.map((child, index) => (
        <line
          key={index}
          x1={xOf(node.x)}
          y1={yOf(child.y)}
          x2={xOf(child.x)}
          y2={yOf(child.y)}
          stroke="var(--border)"
          strokeWidth={1.5}
          strokeLinecap="round"
        />
      ))}
    </g>
  );
}

/** UFBoot percentage, on the node it supports. */
function SupportLabel({ x, y, support }: { x: number; y: number; support: number }) {
  const tone = supportTone(support);
  const text = String(Math.round(support));
  const boxWidth = text.length * 7 + 8;

  return (
    <g>
      <rect
        x={x - boxWidth - 3}
        y={y - 8}
        width={boxWidth}
        height={16}
        rx={4}
        fill="var(--card)"
        stroke={tone.line}
        strokeWidth={1}
      />
      <text
        x={x - boxWidth / 2 - 3}
        y={y}
        textAnchor="middle"
        dominantBaseline="middle"
        fontSize={10}
        className={cn("font-medium", tone.text)}
        fill="currentColor"
      >
        {text}
      </text>
    </g>
  );
}

/** Substitutions per site, so branch lengths can be read as distances. */
function ScaleBar({ x, y, step, width }: { x: number; y: number; step: number; width: number }) {
  return (
    <g>
      <line x1={x} y1={y} x2={x + width} y2={y} stroke="var(--muted-foreground)" strokeWidth={1} />
      <line x1={x} y1={y - 3} x2={x} y2={y + 3} stroke="var(--muted-foreground)" strokeWidth={1} />
      <line
        x1={x + width}
        y1={y - 3}
        x2={x + width}
        y2={y + 3}
        stroke="var(--muted-foreground)"
        strokeWidth={1}
      />
      <text
        x={x + width + 8}
        y={y}
        dominantBaseline="middle"
        fontSize={10}
        className="fill-muted-foreground"
      >
        {step} substitutions/site
      </text>
    </g>
  );
}

/**
 * Whether the tree has a root — stated, never implied.
 *
 * Drawing a tree left-to-right implies a root at the left edge. When the
 * analysis did not estimate one, that implication is a claim the data does not
 * support, and it is exactly the claim a reader takes away unchallenged.
 */
function Rootedness({ rooted }: { rooted: boolean }) {
  if (rooted) return null;
  return (
    <p className="border-b border-border bg-muted/40 px-4 py-2.5 text-xs text-muted-foreground">
      <span className="font-medium text-foreground">Unrooted.</span> No outgroup was specified, so
      the tree shows how the species group, not which lineage branched first. The leftmost split is
      a drawing convention, not a common ancestor.
    </p>
  );
}

/** The supported splits, as claims a reader can check against the picture. */
function Clades({ clades: list }: { clades: ReturnType<typeof clades> }) {
  return (
    <div className="border-b border-border px-4 py-3">
      <h4 className="mb-2 text-xs font-medium text-muted-foreground">Supported groupings</h4>
      <ul className="flex flex-col gap-1.5">
        {list.map((clade, index) => {
          const tone = supportTone(clade.support);
          return (
            <li key={index} className="flex items-baseline gap-2.5 text-xs">
              <span
                className="shrink-0 rounded px-1.5 py-0.5 font-mono tabular-nums"
                style={{
                  backgroundColor: `color-mix(in oklab, ${tone.line} 16%, transparent)`,
                }}
              >
                {Math.round(clade.support)}
              </span>
              <span className="text-muted-foreground">{tone.label}</span>
              <span className="min-w-0 flex-1 italic text-foreground">
                {clade.members.join(" + ")}
              </span>
            </li>
          );
        })}
      </ul>
    </div>
  );
}

/**
 * Pairwise similarity, as a matrix.
 *
 * Scores are centred cosine similarities in [-1, 1], not percentage identity —
 * a species pair is scored *relative to the others in the request*, so
 * negative values are ordinary and a percentage bar would be a wrong reading.
 * The diverging colour scale carries that: zero is neutral, not empty.
 */
function SimilarityMatrix({
  species,
  scores,
  groups,
}: {
  species: string[];
  scores: SimilarityScore[];
  groups: EvolutionSpec["speciesGroups"];
}) {
  // The agent lower-cases names inside `similarity_scores` while
  // `species_list` is title-cased, so every lookup here is case-insensitive.
  const key = (a: string, b: string) => [a.toLowerCase(), b.toLowerCase()].sort().join("|");
  const byPair = new Map(scores.map((edge) => [key(edge.speciesA, edge.speciesB), edge.score]));

  const names =
    species.length > 0
      ? species
      : Array.from(new Set(scores.flatMap((edge) => [edge.speciesA, edge.speciesB])));

  const groupOf = new Map<string, number>();
  groups.forEach((group) =>
    group.species.forEach((name) => groupOf.set(name.toLowerCase(), group.groupId)),
  );

  return (
    <div className="border-b border-border px-4 py-3">
      <h4 className="mb-2 text-xs font-medium text-muted-foreground">
        Sequence similarity
        <span className="ml-1.5 font-normal">
          (centred cosine, −1 to 1 — not percentage identity)
        </span>
      </h4>

      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-xs">
          <thead>
            <tr>
              <th className="w-px" />
              {names.map((name) => (
                <th
                  key={name}
                  className="px-1.5 pb-1.5 text-center align-bottom font-medium text-muted-foreground"
                >
                  {abbreviate(name)}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {names.map((row) => (
              <tr key={row}>
                <th className="whitespace-nowrap py-1 pr-2.5 text-right font-normal italic text-foreground">
                  {row}
                </th>
                {names.map((column) => {
                  if (row === column) {
                    return (
                      <td key={column} className="p-0.5">
                        <div className="rounded bg-muted/60 py-1 text-center text-muted-foreground">
                          —
                        </div>
                      </td>
                    );
                  }
                  const value = byPair.get(key(row, column));
                  return (
                    <td key={column} className="p-0.5">
                      <ScoreCell value={value} />
                    </td>
                  );
                })}
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {groups.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1.5">
          {groups.map((group) => (
            <span
              key={group.groupId}
              className="rounded-full border border-border px-2 py-0.5 text-xs italic text-foreground"
            >
              {group.species.map(titleCase).join(" + ")}
            </span>
          ))}
        </div>
      )}
      {groups.length > 0 && (
        <p className="mt-1.5 text-xs text-muted-foreground">
          {groups.length === 1
            ? "All species fell into a single similarity group."
            : `${groups.length} similarity groups.`}
        </p>
      )}
      {groupOf.size === 0 && null}
    </div>
  );
}

/**
 * One cell of the matrix, on a diverging scale.
 *
 * Positive scores read warm, negative cool, and the midpoint is a visible
 * neutral rather than blank — an empty-looking cell would read as missing data
 * when it in fact means "no signal either way".
 */
function ScoreCell({ value }: { value: number | undefined }) {
  if (value === undefined) {
    return <div className="py-1 text-center text-muted-foreground">·</div>;
  }

  const magnitude = Math.min(Math.abs(value), 1);
  // Teal for "more similar than average", violet for "less". Two hues rather
  // than two ends of one ramp, because the midpoint is a real boundary: it is
  // where a pair stops being closer than the batch.
  const hue = value >= 0 ? "var(--seq-fill)" : "var(--seq-model)";

  return (
    <div
      className="rounded py-1 text-center font-mono tabular-nums text-foreground"
      style={{
        backgroundColor: `color-mix(in oklab, ${hue} ${Math.round(12 + magnitude * 48)}%, transparent)`,
      }}
      title={`${value}`}
    >
      {value.toFixed(2)}
    </div>
  );
}

/** "Homo sapiens" → "H. sapiens", so column headers stay narrow. */
function abbreviate(name: string): string {
  const parts = name.trim().split(/\s+/);
  if (parts.length < 2) return name;
  return `${parts[0][0].toUpperCase()}. ${parts.slice(1).join(" ")}`;
}

/** The agent's own caveats, in words rather than as machine flags. */
const WARNING_LABELS: Record<string, string> = {
  ufboot_not_run:
    "Branch support was not computed — UFBoot needs at least 4 species. The topology stands, but nothing here says how well it is supported.",
  interpretation_unavailable:
    "The written interpretation could not be produced or verified, so a deterministic summary was used instead.",
};

function warningLabel(warning: string): string {
  if (WARNING_LABELS[warning]) return WARNING_LABELS[warning];
  if (warning.startsWith("offline_sequence_fallback")) {
    return "UniProt was unreachable, so short offline reference fragments were used. Treat this tree as a smoke test, not a finding.";
  }
  if (warning.startsWith("molecular_comparison_failed")) {
    return `The similarity comparison failed, so only the tree is shown. ${warning.split(": ").slice(1).join(": ")}`;
  }
  if (warning.startsWith("phylogenetic_tree_failed")) {
    return `Tree reconstruction failed, so only the similarity network is shown. ${warning.split(": ").slice(1).join(": ")}`;
  }
  return warning.replace(/_/g, " ");
}

function Warnings({ warnings }: { warnings: string[] }) {
  return (
    <ul className="flex flex-col gap-1.5 border-b border-border bg-[var(--chart-4)]/8 px-4 py-3">
      {warnings.map((warning) => (
        <li key={warning} className="text-xs text-muted-foreground">
          {warningLabel(warning)}
        </li>
      ))}
    </ul>
  );
}

/** The Newick itself, folded away but copyable — it is the portable artefact. */
function RawNewick({ newick }: { newick: string }) {
  const [open, setOpen] = useState(false);
  const [copied, setCopied] = useState(false);

  const copy = () => {
    void navigator.clipboard.writeText(newick).then(
      () => {
        setCopied(true);
        window.setTimeout(() => setCopied(false), 1600);
      },
      () => setCopied(false),
    );
  };

  return (
    <Collapsible open={open} onOpenChange={setOpen}>
      <div className="flex items-center justify-between px-4 py-2.5">
        <CollapsibleTrigger className="flex items-center gap-1.5 text-xs text-muted-foreground transition-colors hover:text-foreground">
          <ChevronRight className={cn("size-3.5 transition-transform", open && "rotate-90")} />
          Newick
        </CollapsibleTrigger>
        <button
          type="button"
          onClick={copy}
          className="flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
        >
          {copied ? <Check className="size-3.5" /> : <Copy className="size-3.5" />}
          {copied ? "Copied" : "Copy"}
        </button>
      </div>
      <CollapsibleContent>
        <pre className="overflow-x-auto border-t border-border bg-muted/40 px-4 py-3 font-mono text-xs leading-relaxed text-foreground">
          {newick}
        </pre>
      </CollapsibleContent>
    </Collapsible>
  );
}
