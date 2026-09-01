import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { Check, ChevronRight, Copy, RotateCcw } from "lucide-react";

import { Collapsible, CollapsibleContent, CollapsibleTrigger } from "@/components/ui/collapsible";
import { Slider } from "@/components/ui/slider";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { ToggleGroup, ToggleGroupItem } from "@/components/ui/toggle-group";
import { layoutForce, type GraphEdge, type Point } from "@/lib/force-graph";
import {
  clades,
  layoutRadial,
  layoutTree,
  niceScaleStep,
  parseNewick,
  tipNames,
  walkLayout,
  walkRadial,
  type LaidOutNode,
  type NewickNode,
  type RadialLayout,
  type TreeLayout,
} from "@/lib/newick";
import type { EvolutionSpec, SimilarityScore } from "@/lib/umbrella-types";
import { cn } from "@/lib/utils";

/** Geometry of the rectangular phylogram, in SVG user units. */
const ROW_HEIGHT = 30;
const PAD_LEFT = 12;
const PAD_TOP = 14;
const SCALE_BAR_HEIGHT = 34;
const MIN_PLOT_WIDTH = 220;

/** Rough advance width of the 12px italic label face, for reserving gutter. */
const LABEL_CHAR_WIDTH = 6.4;
const LABEL_MIN = 120;
const LABEL_MAX = 210;

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

/** Distinct hues for similarity groups, reused by the network and the chips. */
const GROUP_COLORS = [
  "var(--seq-fill)",
  "var(--seq-model)",
  "var(--seq-gap)",
  "var(--chart-1)",
  "var(--chart-2)",
];

/** "homo sapiens" -> "Homo sapiens". The agent lower-cases some name fields. */
function titleCase(name: string): string {
  return name.charAt(0).toUpperCase() + name.slice(1);
}

/** "Homo sapiens" → "H. sapiens", so labels stay narrow. */
function abbreviate(name: string): string {
  const parts = name.trim().split(/\s+/);
  if (parts.length < 2) return name;
  return `${parts[0][0].toUpperCase()}. ${parts.slice(1).join(" ")}`;
}

/** The agent mixes casing between fields, so every cross-field lookup folds it. */
const fold = (name: string) => name.trim().toLowerCase();

/**
 * The element's rendered width, tracked.
 *
 * The tree used to be drawn at a fixed 468px whatever it was given, which is
 * cramped in a wide chat column and overflows a narrow one. Measuring lets the
 * plot take the room it actually has, and lets the label gutter be sized from
 * the names in this particular tree rather than from a guess.
 */
function useMeasuredWidth<T extends HTMLElement>(fallback: number) {
  const ref = useRef<T | null>(null);
  const [width, setWidth] = useState(fallback);

  useEffect(() => {
    const element = ref.current;
    if (!element) return;

    // A progressive enhancement over the fallback width, not a requirement:
    // ResizeObserver is absent in some SSR and test environments.
    if (typeof ResizeObserver === "undefined") return;

    const observer = new ResizeObserver((entries) => {
      const measured = entries[0]?.contentRect.width;
      if (measured && measured > 0) setWidth(measured);
    });
    observer.observe(element);
    return () => observer.disconnect();
  }, []);

  return [ref, width] as const;
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
 * Three things are load-bearing here:
 *
 *  1. **The tree is drawn, not described.** Branch lengths are real distances
 *     on the x axis, with a scale bar, so "human and chimp are close" is
 *     something the reader sees rather than something they are told.
 *  2. **Rootedness is stated.** IQ-TREE returns an unrooted tree; nothing in
 *     the Newick says so except the shape of the top-level node. The panel
 *     says it in words, because every visual convention for drawing a tree
 *     implies a root that this analysis did not estimate.
 *  3. **The network is drawn as a network.** The agent returns a weighted
 *     graph, and it used to be shown only as a table of numbers - the one
 *     form that hides the structure the graph exists to show.
 */
export function EvolutionPanel({ spec }: { spec: EvolutionSpec }) {
  const parsed = useMemo(() => {
    if (!spec.newick) return null;
    try {
      return parseNewick(spec.newick);
    } catch {
      // A tree we cannot parse is not drawn. The Newick itself is still
      // offered below, so nothing is lost but the picture.
      return null;
    }
  }, [spec.newick]);

  const supported = useMemo(() => (parsed ? clades(parsed) : []), [parsed]);

  const hasTree = Boolean(spec.newick);
  const networkEdges = spec.similarityNetwork?.edges ?? spec.similarityScores;
  const hasNetwork = networkEdges.length > 0;

  const tabs = [hasTree && "tree", hasNetwork && "network", hasNetwork && "matrix"].filter(
    Boolean,
  ) as string[];

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

      {tabs.length > 0 && (
        <Tabs defaultValue={tabs[0]}>
          {tabs.length > 1 && (
            <div className="border-b border-border px-4 pt-3">
              <TabsList className="h-8">
                {hasTree && (
                  <TabsTrigger value="tree" className="text-xs">
                    Tree
                  </TabsTrigger>
                )}
                {hasNetwork && (
                  <TabsTrigger value="network" className="text-xs">
                    Network
                  </TabsTrigger>
                )}
                {hasNetwork && (
                  <TabsTrigger value="matrix" className="text-xs">
                    Matrix
                  </TabsTrigger>
                )}
              </TabsList>
            </div>
          )}

          {hasTree && (
            <TabsContent value="tree" className="mt-0">
              <TreeView root={parsed} newick={spec.newick ?? ""} clades={supported} />
            </TabsContent>
          )}

          {hasNetwork && (
            <TabsContent value="network" className="mt-0">
              <NetworkView
                species={spec.similarityNetwork?.species ?? spec.speciesList}
                edges={networkEdges}
                groups={spec.speciesGroups}
              />
            </TabsContent>
          )}

          {hasNetwork && (
            <TabsContent value="matrix" className="mt-0">
              <SimilarityMatrix
                species={spec.speciesList}
                scores={spec.similarityScores}
                groups={spec.speciesGroups}
              />
            </TabsContent>
          )}
        </Tabs>
      )}

      {spec.warnings.length > 0 && <Warnings warnings={spec.warnings} />}
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

// ---------------------------------------------------------------------------
// Tree
// ---------------------------------------------------------------------------

type TreeShape = "rectangular" | "radial";

/**
 * The tree, in whichever projection suits the number of tips.
 *
 * Hovering a clade - in the picture or in the list below it - highlights the
 * same set of tips in both. The supported groupings are claims about the
 * picture, and pairing them by hover is what lets a reader check one against
 * the other instead of taking the list on faith.
 */
function TreeView({
  root,
  newick,
  clades: list,
}: {
  root: NewickNode | null;
  newick: string;
  clades: ReturnType<typeof clades>;
}) {
  const [shape, setShape] = useState<TreeShape>("rectangular");
  const [hovered, setHovered] = useState<Set<string> | null>(null);
  const [ref, width] = useMeasuredWidth<HTMLDivElement>(560);

  const rectangular = useMemo(() => (root ? layoutTree(root) : null), [root]);
  const radial = useMemo(() => (root ? layoutRadial(root) : null), [root]);

  // Tip sets per node, computed once. Hover asks "is this branch inside the
  // highlighted clade?" for every node on every pointer move; recomputing the
  // subtree each time turns that into quadratic work.
  const tipsOf = useMemo(() => {
    const map = new Map<NewickNode, string[]>();
    if (!root) return map;
    const walk = (node: NewickNode) => {
      map.set(node, tipNames(node));
      node.children.forEach(walk);
    };
    walk(root);
    return map;
  }, [root]);

  const isHighlighted = useCallback(
    (node: NewickNode) => {
      if (!hovered) return false;
      const tips = tipsOf.get(node) ?? [];
      return tips.length > 0 && tips.every((tip) => hovered.has(tip));
    },
    [hovered, tipsOf],
  );

  const highlight = useCallback(
    (node: NewickNode | null) => setHovered(node ? new Set(tipsOf.get(node) ?? []) : null),
    [tipsOf],
  );

  if (!root || !rectangular || !radial) {
    return (
      <>
        <p className="border-b border-border px-4 py-3 text-xs text-muted-foreground">
          The tree could not be drawn — the Newick below did not parse.
        </p>
        <RawNewick newick={newick} />
      </>
    );
  }

  return (
    <>
      <div className="flex items-center justify-between gap-3 px-4 pb-1 pt-3">
        <ToggleGroup
          type="single"
          size="sm"
          value={shape}
          onValueChange={(value) => value && setShape(value as TreeShape)}
          className="gap-1"
        >
          <ToggleGroupItem value="rectangular" className="h-6 px-2 text-xs">
            Rectangular
          </ToggleGroupItem>
          <ToggleGroupItem value="radial" className="h-6 px-2 text-xs">
            Radial
          </ToggleGroupItem>
        </ToggleGroup>
        <span className="text-xs text-muted-foreground">
          {rectangular.tips.length} tips
          {rectangular.cladogram && " · no branch lengths"}
        </span>
      </div>

      <div ref={ref} className="overflow-x-auto px-4 pb-3">
        {shape === "rectangular" ? (
          <Phylogram
            layout={rectangular}
            width={width}
            isHighlighted={isHighlighted}
            onHover={highlight}
          />
        ) : (
          <RadialTree
            layout={radial}
            width={width}
            isHighlighted={isHighlighted}
            onHover={highlight}
          />
        )}
      </div>

      <Rootedness rooted={rectangular.rooted} />

      {list.length > 0 && (
        <Clades
          clades={list}
          hovered={hovered}
          onHover={(members) => setHovered(members ? new Set(members) : null)}
        />
      )}

      <RawNewick newick={newick} />
    </>
  );
}

/**
 * A rectangular phylogram.
 *
 * Drawn as inline SVG rather than with a chart library: a phylogram is a few
 * dozen line segments, and the layout it needs (x from accumulated branch
 * length, y from tip order) is not something a generic charting API expresses.
 */
function Phylogram({
  layout,
  width: available,
  isHighlighted,
  onHover,
}: {
  layout: TreeLayout;
  width: number;
  isHighlighted: (node: NewickNode) => boolean;
  onHover: (node: NewickNode | null) => void;
}) {
  const rows = layout.tips.length;
  const height = rows * ROW_HEIGHT + PAD_TOP * 2 + SCALE_BAR_HEIGHT;

  const longest = layout.tips.reduce((max, tip) => Math.max(max, tip.node.name.length), 0);
  const labelWidth = Math.min(Math.max(longest * LABEL_CHAR_WIDTH + 16, LABEL_MIN), LABEL_MAX);
  const plotWidth = Math.max(available - PAD_LEFT - labelWidth - 8, MIN_PLOT_WIDTH);
  const width = PAD_LEFT + plotWidth + labelWidth;

  const xOf = (value: number) => PAD_LEFT + (value / layout.maxDepth) * plotWidth;
  const yOf = (value: number) => PAD_TOP + value * ROW_HEIGHT + ROW_HEIGHT / 2;

  const nodes = walkLayout(layout.root);
  const step = niceScaleStep(layout.maxDepth);
  const scaleY = PAD_TOP + rows * ROW_HEIGHT + 16;

  return (
    <svg
      viewBox={`0 0 ${width} ${height}`}
      width={width}
      height={height}
      role="img"
      aria-label={`Phylogenetic tree of ${layout.tips.map((tip) => tip.node.name).join(", ")}`}
      className="max-w-full"
      onPointerLeave={() => onHover(null)}
    >
      {nodes.map((node, index) => (
        <Branches
          key={index}
          node={node}
          xOf={xOf}
          yOf={yOf}
          highlighted={isHighlighted(node.node)}
          onHover={onHover}
        />
      ))}

      {layout.tips.map((tip, index) => (
        <text
          key={index}
          x={xOf(tip.x) + 8}
          y={yOf(tip.y)}
          dominantBaseline="middle"
          fontSize={12}
          fontStyle="italic"
          className={cn("fill-foreground", isHighlighted(tip.node) && "font-semibold")}
        >
          {tip.node.name}
        </text>
      ))}

      {nodes
        .filter((node) => node.children.length > 0 && node.node.support !== null)
        .map((node, index) => (
          <SupportLabel key={index} x={xOf(node.x)} y={yOf(node.y)} support={node.node.support!} />
        ))}

      {!layout.cladogram && (
        <ScaleBar
          x={PAD_LEFT}
          y={scaleY}
          step={step}
          width={(step / layout.maxDepth) * plotWidth}
        />
      )}
    </svg>
  );
}

/**
 * The elbow joining a node to its children: one vertical span across the
 * clade, one horizontal segment per child.
 *
 * The wide transparent stroke under the span is the hover target. A 1.5px line
 * is close to unhittable with a pointer, and a clade you cannot hover is a
 * clade you cannot check against the list.
 */
function Branches({
  node,
  xOf,
  yOf,
  highlighted,
  onHover,
}: {
  node: LaidOutNode;
  xOf: (value: number) => number;
  yOf: (value: number) => number;
  highlighted: boolean;
  onHover: (node: NewickNode | null) => void;
}) {
  if (node.children.length === 0) return null;

  const first = node.children[0];
  const last = node.children[node.children.length - 1];
  const support = node.node.support;
  const stroke = highlighted
    ? "var(--primary)"
    : support !== null
      ? supportTone(support).line
      : "var(--border)";
  const childStroke = highlighted ? "var(--primary)" : "var(--border)";

  return (
    <g onPointerEnter={() => onHover(node.node)} style={{ cursor: "pointer" }}>
      <line
        x1={xOf(node.x)}
        y1={yOf(first.y)}
        x2={xOf(node.x)}
        y2={yOf(last.y)}
        stroke="transparent"
        strokeWidth={12}
      />
      <line
        x1={xOf(node.x)}
        y1={yOf(first.y)}
        x2={xOf(node.x)}
        y2={yOf(last.y)}
        stroke={stroke}
        strokeWidth={highlighted ? 2.5 : 1.5}
        strokeLinecap="round"
      />
      {node.children.map((child, index) => (
        <line
          key={index}
          x1={xOf(node.x)}
          y1={yOf(child.y)}
          x2={xOf(child.x)}
          y2={yOf(child.y)}
          stroke={childStroke}
          strokeWidth={highlighted ? 2.5 : 1.5}
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
    <g style={{ pointerEvents: "none" }}>
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
    <g style={{ pointerEvents: "none" }}>
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
 * The same tree, drawn round.
 *
 * Past about a dozen tips the rectangular form becomes a tall strip that no
 * longer sits beside the answer it belongs to. The radial form spends
 * circumference instead of height, so the topology stays in one view.
 */
function RadialTree({
  layout,
  width: available,
  isHighlighted,
  onHover,
}: {
  layout: RadialLayout;
  width: number;
  isHighlighted: (node: NewickNode) => boolean;
  onHover: (node: NewickNode | null) => void;
}) {
  const longest = layout.tips.reduce((max, tip) => Math.max(max, tip.node.name.length), 0);
  const labelGutter = Math.min(longest * LABEL_CHAR_WIDTH + 14, LABEL_MAX);

  const size = Math.max(Math.min(available, 620), 320);
  const centre = size / 2;
  const plotRadius = Math.max(centre - labelGutter, 60);
  const rOf = (value: number) => (value / layout.maxRadius) * plotRadius;

  // Angle 0 at 12 o'clock, increasing clockwise - the convention `layoutRadial`
  // documents, converted in one place so the two forms cannot drift apart.
  const pointAt = (angle: number, radius: number) => ({
    x: centre + Math.sin(angle) * radius,
    y: centre - Math.cos(angle) * radius,
  });

  const nodes = walkRadial(layout.root);

  return (
    <svg
      viewBox={`0 0 ${size} ${size}`}
      width={size}
      height={size}
      role="img"
      aria-label={`Radial phylogenetic tree of ${layout.tips
        .map((tip) => tip.node.name)
        .join(", ")}`}
      className="mx-auto max-w-full"
      onPointerLeave={() => onHover(null)}
    >
      {nodes
        .filter((node) => node.children.length > 0)
        .map((node, index) => {
          const highlighted = isHighlighted(node.node);
          const support = node.node.support;
          const stroke = highlighted
            ? "var(--primary)"
            : support !== null
              ? supportTone(support).line
              : "var(--border)";

          const radius = rOf(node.radius);
          const first = node.children[0].angle;
          const last = node.children[node.children.length - 1].angle;
          const start = pointAt(first, radius);
          const end = pointAt(last, radius);
          const largeArc = Math.abs(last - first) > Math.PI ? 1 : 0;
          const arc = `M ${start.x} ${start.y} A ${radius} ${radius} 0 ${largeArc} 1 ${end.x} ${end.y}`;

          return (
            <g key={index} onPointerEnter={() => onHover(node.node)} style={{ cursor: "pointer" }}>
              {/* Wide invisible arc: the hover target for this clade. */}
              <path d={arc} fill="none" stroke="transparent" strokeWidth={12} />
              <path
                d={arc}
                fill="none"
                stroke={stroke}
                strokeWidth={highlighted ? 2.5 : 1.5}
                strokeLinecap="round"
              />
              {node.children.map((child, childIndex) => {
                const from = pointAt(child.angle, radius);
                const to = pointAt(child.angle, rOf(child.radius));
                return (
                  <line
                    key={childIndex}
                    x1={from.x}
                    y1={from.y}
                    x2={to.x}
                    y2={to.y}
                    stroke={highlighted ? "var(--primary)" : "var(--border)"}
                    strokeWidth={highlighted ? 2.5 : 1.5}
                    strokeLinecap="round"
                  />
                );
              })}
            </g>
          );
        })}

      {layout.tips.map((tip, index) => {
        const point = pointAt(tip.angle, rOf(tip.radius));
        const degrees = (tip.angle * 180) / Math.PI;
        // Flip the labels on the left half so none of them read upside down.
        const flipped = degrees > 180;
        const rotation = flipped ? degrees + 90 : degrees - 90;

        return (
          <text
            key={index}
            x={point.x}
            y={point.y}
            dx={flipped ? -7 : 7}
            transform={`rotate(${rotation}, ${point.x}, ${point.y})`}
            textAnchor={flipped ? "end" : "start"}
            dominantBaseline="middle"
            fontSize={11}
            fontStyle="italic"
            className={cn("fill-foreground", isHighlighted(tip.node) && "font-semibold")}
          >
            {tip.node.name}
          </text>
        );
      })}

      {nodes
        .filter((node) => node.children.length > 0 && node.node.support !== null)
        .map((node, index) => {
          const point = pointAt(node.angle, rOf(node.radius));
          return (
            <circle
              key={index}
              cx={point.x}
              cy={point.y}
              r={3}
              fill={supportTone(node.node.support!).line}
              style={{ pointerEvents: "none" }}
            >
              <title>{`${Math.round(node.node.support!)}% UFBoot support`}</title>
            </circle>
          );
        })}
    </svg>
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
    <p className="border-y border-border bg-muted/40 px-4 py-2.5 text-xs text-muted-foreground">
      <span className="font-medium text-foreground">Unrooted.</span> No outgroup was specified, so
      the tree shows how the species group, not which lineage branched first. The leftmost split is
      a drawing convention, not a common ancestor.
    </p>
  );
}

/** The supported splits, as claims a reader can check against the picture. */
function Clades({
  clades: list,
  hovered,
  onHover,
}: {
  clades: ReturnType<typeof clades>;
  hovered: Set<string> | null;
  onHover: (members: string[] | null) => void;
}) {
  const isActive = (members: string[]) =>
    hovered !== null &&
    members.length === hovered.size &&
    members.every((member) => hovered.has(member));

  return (
    <div className="border-b border-border px-4 py-3">
      <h4 className="mb-2 text-xs font-medium text-muted-foreground">Supported groupings</h4>
      <ul className="flex flex-col gap-0.5" onPointerLeave={() => onHover(null)}>
        {list.map((clade, index) => {
          const tone = supportTone(clade.support);
          return (
            <li
              key={index}
              onPointerEnter={() => onHover(clade.members)}
              className={cn(
                "-mx-1.5 flex cursor-pointer items-baseline gap-2.5 rounded px-1.5 py-1 text-xs transition-colors",
                isActive(clade.members) && "bg-muted",
              )}
            >
              <span
                className="shrink-0 rounded px-1.5 py-0.5 font-mono tabular-nums"
                style={{ backgroundColor: `color-mix(in oklab, ${tone.line} 16%, transparent)` }}
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

// ---------------------------------------------------------------------------
// Network
// ---------------------------------------------------------------------------

const NETWORK_HEIGHT = 340;

/**
 * The similarity graph, drawn as a graph.
 *
 * The agent returns a *complete* weighted graph - every species scored against
 * every other - so drawing all of it at once is a hairball that says nothing.
 * The threshold is therefore part of the reading rather than a nicety: it
 * answers "which species are close enough to count as related?", and the
 * answer changes as you move it.
 *
 * The layout is solved once from every edge and does not change with the
 * threshold. Re-solving on each slider step would move every node, which reads
 * as the data changing when only the filter did.
 */
function NetworkView({
  species,
  edges,
  groups,
}: {
  species: string[];
  edges: SimilarityScore[];
  groups: EvolutionSpec["speciesGroups"];
}) {
  const [ref, available] = useMeasuredWidth<HTMLDivElement>(560);
  const [dragged, setDragged] = useState<Map<string, Point>>(new Map());
  const [hovered, setHovered] = useState<string | null>(null);
  const [dragging, setDragging] = useState<string | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const width = Math.max(available, 280);

  // Node list: the graph's own membership when the agent sent one, otherwise
  // whatever the edges mention. An isolated species has to survive either way -
  // "nothing here is close to it" is a finding, not an absence.
  const nodes = useMemo(() => {
    const names = new Set(species.map(fold));
    for (const edge of edges) {
      names.add(fold(edge.speciesA));
      names.add(fold(edge.speciesB));
    }
    return Array.from(names);
  }, [species, edges]);

  const scores = useMemo(() => edges.map((edge) => edge.score), [edges]);
  const min = scores.length ? Math.min(...scores) : 0;
  const max = scores.length ? Math.max(...scores) : 1;
  const span = max - min || 1;

  // The median as the opening threshold: half the links, which is legible, and
  // a value that exists in the data rather than an arbitrary constant.
  const median = useMemo(() => {
    if (scores.length === 0) return 0;
    const sorted = [...scores].sort((a, b) => a - b);
    return sorted[Math.floor(sorted.length / 2)];
  }, [scores]);

  const [threshold, setThreshold] = useState(median);
  useEffect(() => setThreshold(median), [median]);

  const layout = useMemo(() => {
    const forceEdges: GraphEdge[] = edges.map((edge) => ({
      source: fold(edge.speciesA),
      target: fold(edge.speciesB),
      // Normalised to [0,1] for the spring. The raw score is a centred cosine
      // and is routinely negative; a negative spring constant would push
      // similar species apart.
      weight: (edge.score - min) / span,
    }));
    return layoutForce(nodes, forceEdges, { width, height: NETWORK_HEIGHT, padding: 56 });
  }, [nodes, edges, min, span, width]);

  const positionOf = useCallback(
    (id: string): Point =>
      dragged.get(id) ?? layout.get(id) ?? { x: width / 2, y: NETWORK_HEIGHT / 2 },
    [dragged, layout, width],
  );

  const visible = useMemo(
    () => edges.filter((edge) => edge.score >= threshold),
    [edges, threshold],
  );

  const groupOf = useMemo(() => {
    const map = new Map<string, number>();
    groups.forEach((group, index) => group.species.forEach((name) => map.set(fold(name), index)));
    return map;
  }, [groups]);

  const neighbours = useMemo(() => {
    if (!hovered) return null;
    const set = new Set<string>([hovered]);
    for (const edge of visible) {
      if (fold(edge.speciesA) === hovered) set.add(fold(edge.speciesB));
      if (fold(edge.speciesB) === hovered) set.add(fold(edge.speciesA));
    }
    return set;
  }, [hovered, visible]);

  const onPointerMove = (event: React.PointerEvent<SVGSVGElement>) => {
    if (!dragging || !svgRef.current) return;
    const box = svgRef.current.getBoundingClientRect();
    // The viewBox is in the same units as the rendered size, so the only
    // conversion needed is the element's offset and any CSS down-scaling.
    setDragged((current) =>
      new Map(current).set(dragging, {
        x: ((event.clientX - box.left) / box.width) * width,
        y: ((event.clientY - box.top) / box.height) * NETWORK_HEIGHT,
      }),
    );
  };

  return (
    <div ref={ref}>
      <div className="flex flex-wrap items-center gap-x-4 gap-y-2 px-4 pb-1 pt-3">
        <div className="flex min-w-[180px] flex-1 items-center gap-2.5">
          <span className="whitespace-nowrap text-xs text-muted-foreground">Link threshold</span>
          <Slider
            value={[threshold]}
            min={min}
            max={max}
            step={span / 100}
            onValueChange={([value]) => setThreshold(value)}
            className="max-w-[190px] flex-1"
            aria-label="Minimum similarity score for a link to be drawn"
          />
          <span className="w-11 shrink-0 font-mono text-xs tabular-nums text-foreground">
            {threshold.toFixed(2)}
          </span>
        </div>
        <span className="text-xs text-muted-foreground">
          {visible.length} of {edges.length} links
        </span>
        {dragged.size > 0 && (
          <button
            type="button"
            onClick={() => setDragged(new Map())}
            className="flex items-center gap-1.5 rounded-md px-2 py-1 text-xs text-muted-foreground transition-colors hover:bg-muted hover:text-foreground"
          >
            <RotateCcw className="size-3" />
            Reset layout
          </button>
        )}
      </div>

      <div className="px-4 pb-2">
        <svg
          ref={svgRef}
          viewBox={`0 0 ${width} ${NETWORK_HEIGHT}`}
          width={width}
          height={NETWORK_HEIGHT}
          role="img"
          aria-label={`Similarity network of ${nodes.map(titleCase).join(", ")}`}
          className="max-w-full touch-none select-none"
          onPointerMove={onPointerMove}
          onPointerUp={() => setDragging(null)}
          onPointerLeave={() => {
            setDragging(null);
            setHovered(null);
          }}
        >
          {visible.map((edge, index) => {
            const a = fold(edge.speciesA);
            const b = fold(edge.speciesB);
            const from = positionOf(a);
            const to = positionOf(b);
            const magnitude = (edge.score - min) / span;
            const dimmed = neighbours ? !(neighbours.has(a) && neighbours.has(b)) : false;

            return (
              <line
                key={index}
                x1={from.x}
                y1={from.y}
                x2={to.x}
                y2={to.y}
                // Teal above the batch average, violet below - the same two
                // hues the matrix uses, so a reader who has looked at one does
                // not have to relearn the other.
                stroke={edge.score >= 0 ? "var(--seq-fill)" : "var(--seq-model)"}
                strokeWidth={1 + magnitude * 2.6}
                strokeOpacity={dimmed ? 0.08 : 0.25 + magnitude * 0.5}
                strokeLinecap="round"
              >
                <title>{`${titleCase(edge.speciesA)} ↔ ${titleCase(edge.speciesB)}: ${edge.score.toFixed(3)}`}</title>
              </line>
            );
          })}

          {nodes.map((id) => {
            const point = positionOf(id);
            const group = groupOf.get(id);
            const colour =
              group === undefined
                ? "var(--muted-foreground)"
                : GROUP_COLORS[group % GROUP_COLORS.length];
            const dimmed = neighbours ? !neighbours.has(id) : false;
            const active = hovered === id;

            return (
              <g
                key={id}
                opacity={dimmed ? 0.25 : 1}
                onPointerEnter={() => setHovered(id)}
                onPointerDown={(event) => {
                  event.preventDefault();
                  setDragging(id);
                }}
                style={{ cursor: dragging === id ? "grabbing" : "grab" }}
              >
                <circle
                  cx={point.x}
                  cy={point.y}
                  r={active ? 9 : 7}
                  fill={colour}
                  stroke="var(--card)"
                  strokeWidth={2}
                />
                <text
                  x={point.x}
                  y={point.y - 13}
                  textAnchor="middle"
                  fontSize={11}
                  fontStyle="italic"
                  className={cn("fill-foreground", active && "font-semibold")}
                >
                  {abbreviate(titleCase(id))}
                </text>
              </g>
            );
          })}
        </svg>
      </div>

      <p className="px-4 pb-3 text-xs text-muted-foreground">
        Distance is similarity: species pulled together scored highly against each other. Scores are
        centred cosine (−1 to 1), <span className="text-foreground">not percentage identity</span> —
        a pair is scored relative to the others in this request, so negatives are ordinary. Drag a
        node to pull the layout apart.
      </p>

      {groups.length > 0 && (
        <div className="border-b border-border px-4 pb-3">
          <div className="flex flex-wrap gap-1.5">
            {groups.map((group, index) => (
              <span
                key={group.groupId}
                className="rounded-full px-2 py-0.5 text-xs italic text-foreground"
                style={{
                  backgroundColor: `color-mix(in oklab, ${GROUP_COLORS[index % GROUP_COLORS.length]} 18%, transparent)`,
                }}
              >
                {group.species.map(titleCase).join(" + ")}
              </span>
            ))}
          </div>
          <p className="mt-1.5 text-xs text-muted-foreground">
            {groups.length === 1
              ? "All species fell into a single similarity group."
              : `${groups.length} similarity groups.`}
          </p>
        </div>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Matrix
// ---------------------------------------------------------------------------

/**
 * Pairwise similarity, as a matrix.
 *
 * Kept alongside the network because the two answer different questions: the
 * network shows the structure, the matrix gives the exact number for a named
 * pair. Scores are centred cosine similarities in [-1, 1], not percentage
 * identity — a species pair is scored *relative to the others in the request*,
 * so negative values are ordinary and a percentage bar would be a wrong
 * reading. The diverging colour scale carries that: zero is neutral, not empty.
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
  const key = (a: string, b: string) => [fold(a), fold(b)].sort().join("|");
  const byPair = new Map(scores.map((edge) => [key(edge.speciesA, edge.speciesB), edge.score]));

  const names =
    species.length > 0
      ? species
      : Array.from(new Set(scores.flatMap((edge) => [edge.speciesA, edge.speciesB])));

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
          {groups.map((group, index) => (
            <span
              key={group.groupId}
              className="rounded-full px-2 py-0.5 text-xs italic text-foreground"
              style={{
                backgroundColor: `color-mix(in oklab, ${GROUP_COLORS[index % GROUP_COLORS.length]} 18%, transparent)`,
              }}
            >
              {group.species.map(titleCase).join(" + ")}
            </span>
          ))}
        </div>
      )}
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

// ---------------------------------------------------------------------------
// Warnings and the raw artefact
// ---------------------------------------------------------------------------

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
    <ul className="flex flex-col gap-1.5 border-t border-border bg-[var(--chart-4)]/8 px-4 py-3">
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
