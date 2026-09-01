/**
 * Newick parsing and phylogram layout.
 *
 * The Evolution Agent returns its tree as a Newick string and nothing else -
 * no coordinates, no clade list. Everything the panel draws is derived here so
 * the drawing code stays declarative.
 *
 * Two details of the agent's output shape this parser:
 *
 *  1. **Leaf labels are quoted**, because scientific names contain a space:
 *     `'Homo sapiens'`. A parser that splits on whitespace, or that treats the
 *     quote as part of the name, mangles every tip.
 *  2. **Internal labels are UFBoot support**, written between the closing
 *     paren and the colon: `)100:0.171`. They are numbers, not names, and are
 *     read as support rather than shown as a node label.
 */

/** One node of a parsed tree. Leaves have no children. */
export interface NewickNode {
  /** Tip label with quoting resolved, or "" for an internal node. */
  name: string;
  /** Branch length leading to this node. `null` when the Newick omits it. */
  length: number | null;
  /** UFBoot percentage on the branch leading here, when the tree carries one. */
  support: number | null;
  children: NewickNode[];
}

export class NewickError extends Error {}

/**
 * Parse a Newick string into a tree.
 *
 * Throws `NewickError` rather than returning a partial tree: half a phylogeny
 * drawn as if it were whole is worse than no phylogeny, since nothing in the
 * picture would say which clades went missing.
 */
export function parseNewick(input: string): NewickNode {
  const text = input.trim();
  if (!text) throw new NewickError("Empty tree string.");

  let i = 0;

  function error(what: string): never {
    throw new NewickError(`${what} at position ${i}.`);
  }

  function skipSpace(): void {
    while (i < text.length && /\s/.test(text[i])) i += 1;
  }

  /** A quoted label, with Newick's doubled-quote escape resolved. */
  function readQuoted(): string {
    i += 1; // opening quote
    let out = "";
    while (i < text.length) {
      if (text[i] === "'") {
        if (text[i + 1] === "'") {
          out += "'";
          i += 2;
          continue;
        }
        i += 1;
        return out;
      }
      out += text[i];
      i += 1;
    }
    error("Unterminated quoted label");
  }

  function readLabel(): string {
    skipSpace();
    if (text[i] === "'") return readQuoted();
    let out = "";
    while (i < text.length && !":,();".includes(text[i])) {
      out += text[i];
      i += 1;
    }
    // Unquoted Newick uses _ for a space.
    return out.trim().replace(/_/g, " ");
  }

  function readLength(): number | null {
    skipSpace();
    if (text[i] !== ":") return null;
    i += 1;
    skipSpace();
    const start = i;
    while (i < text.length && /[-+0-9.eE]/.test(text[i])) i += 1;
    const value = Number.parseFloat(text.slice(start, i));
    return Number.isFinite(value) ? value : null;
  }

  function readNode(): NewickNode {
    skipSpace();
    const children: NewickNode[] = [];

    if (text[i] === "(") {
      i += 1;
      for (;;) {
        children.push(readNode());
        skipSpace();
        if (text[i] === ",") {
          i += 1;
          continue;
        }
        if (text[i] === ")") {
          i += 1;
          break;
        }
        error("Expected ',' or ')'");
      }
    }

    const label = readLabel();
    const length = readLength();

    // On an internal node the label position carries branch support, not a
    // name. Anything non-numeric there is a genuine node label and is kept.
    const isInternal = children.length > 0;
    const supportValue = isInternal && label !== "" ? Number.parseFloat(label) : NaN;
    const support = Number.isFinite(supportValue) ? supportValue : null;

    return {
      name: isInternal && support !== null ? "" : label,
      length,
      support,
      children,
    };
  }

  const root = readNode();
  skipSpace();
  if (text[i] === ";") i += 1;
  skipSpace();
  if (i < text.length) error("Unexpected trailing content");
  if (root.children.length === 0 && !root.name) throw new NewickError("Tree has no taxa.");
  return root;
}

/** Every tip label, left to right as drawn. */
export function tipNames(node: NewickNode): string[] {
  if (node.children.length === 0) return node.name ? [node.name] : [];
  return node.children.flatMap(tipNames);
}

/**
 * Whether the tree is rooted.
 *
 * IQ-TREE returns an **unrooted** tree, which Newick expresses as a basal
 * trifurcation - three branches at the top level instead of two. Read as a
 * rooted tree it looks like three lineages diverging simultaneously, which is
 * a claim the analysis never made. The panel says so explicitly, and this is
 * how it knows.
 */
export function isRooted(root: NewickNode): boolean {
  return root.children.length > 0 && root.children.length <= 2;
}

/** A clade: an internal node's tip set with the support for that split. */
export interface Clade {
  support: number;
  members: string[];
}

/**
 * Supported clades, largest first.
 *
 * Only internal nodes carrying a support value are listed - an unsupported
 * split is not a finding. The root is excluded: "all five taxa" is true of
 * every tree and says nothing.
 */
export function clades(root: NewickNode): Clade[] {
  const out: Clade[] = [];

  function walk(node: NewickNode, isRoot: boolean): void {
    if (node.children.length === 0) return;
    if (!isRoot && node.support !== null) {
      out.push({ support: node.support, members: tipNames(node) });
    }
    for (const child of node.children) walk(child, false);
  }

  walk(root, true);
  return out.sort((a, b) => b.members.length - a.members.length);
}

// ---------------------------------------------------------------------------
// Layout
// ---------------------------------------------------------------------------

export interface LaidOutNode {
  node: NewickNode;
  /** Horizontal position, in substitutions/site accumulated from the root. */
  x: number;
  /** Vertical position, in tip units (0, 1, 2, ... for tips). */
  y: number;
  children: LaidOutNode[];
}

export interface TreeLayout {
  root: LaidOutNode;
  tips: LaidOutNode[];
  /** Greatest root-to-tip distance, for scaling x into pixels. */
  maxDepth: number;
  rooted: boolean;
  /** True when the tree carries no branch lengths at all (a cladogram). */
  cladogram: boolean;
}

/**
 * Position every node: x by accumulated branch length, y by tip order.
 *
 * Internal nodes sit at the midpoint of their children's span, which is what
 * makes a rectangular phylogram readable - the vertical connector spans
 * exactly the clade it joins.
 *
 * A tree with no branch lengths lays out by depth instead, so a topology-only
 * result still draws rather than collapsing every tip onto one column.
 */
export function layoutTree(root: NewickNode): TreeLayout {
  const anyLength = hasAnyLength(root);
  const tips: LaidOutNode[] = [];
  let nextY = 0;

  function place(node: NewickNode, parentX: number): LaidOutNode {
    const step = anyLength ? (node.length ?? 0) : 1;
    const x = parentX + step;

    if (node.children.length === 0) {
      const laid: LaidOutNode = { node, x, y: nextY, children: [] };
      nextY += 1;
      tips.push(laid);
      return laid;
    }

    const children = node.children.map((child) => place(child, x));
    const first = children[0].y;
    const last = children[children.length - 1].y;
    return { node, x, y: (first + last) / 2, children };
  }

  const laidRoot = place(root, 0);
  const maxDepth = tips.reduce((max, tip) => Math.max(max, tip.x), 0);

  return {
    root: laidRoot,
    tips,
    maxDepth: maxDepth > 0 ? maxDepth : 1,
    rooted: isRooted(root),
    cladogram: !anyLength,
  };
}

function hasAnyLength(node: NewickNode): boolean {
  if (node.length !== null && node.length > 0) return true;
  return node.children.some(hasAnyLength);
}

/** Every node of a laid-out tree, root first. */
export function walkLayout(node: LaidOutNode): LaidOutNode[] {
  return [node, ...node.children.flatMap(walkLayout)];
}

/**
 * A round-ish number under `max`, for the scale bar.
 *
 * A bar labelled "0.1" is read at a glance; one labelled "0.1274" is read as a
 * measurement of something.
 */
export function niceScaleStep(max: number): number {
  if (!(max > 0)) return 1;
  const target = max / 4;
  const magnitude = 10 ** Math.floor(Math.log10(target));
  for (const factor of [1, 2, 5]) {
    if (factor * magnitude >= target) return factor * magnitude;
  }
  return 10 * magnitude;
}

// ---------------------------------------------------------------------------
// Radial layout
// ---------------------------------------------------------------------------

/** One node of a radial layout, in polar coordinates. */
export interface RadialNode {
  node: NewickNode;
  /** Angle in radians, 0 at 12 o'clock, increasing clockwise. */
  angle: number;
  /** Distance from the centre, in substitutions/site (depth for a cladogram). */
  radius: number;
  children: RadialNode[];
}

export interface RadialLayout {
  root: RadialNode;
  tips: RadialNode[];
  /** Greatest root-to-tip distance, for scaling radius into pixels. */
  maxRadius: number;
  rooted: boolean;
  cladogram: boolean;
}

/**
 * The same tree drawn round instead of along.
 *
 * A rectangular phylogram spends its vertical extent linearly in the number of
 * tips, so past a dozen species it is a tall strip that no longer fits beside
 * the answer it belongs to. A radial layout spends the *circumference*
 * instead, which grows with the radius - the tips stay legible and the
 * topology stays visible in one screen.
 *
 * Radius is accumulated branch length, exactly as `layoutTree` computes x, so
 * the two views are the same measurements in different coordinates and a
 * reader can move between them without recalibrating.
 */
export function layoutRadial(root: NewickNode): RadialLayout {
  const anyLength = hasAnyLength(root);
  const tipCount = Math.max(tipNames(root).length, 1);
  const tips: RadialNode[] = [];
  let nextIndex = 0;

  function place(node: NewickNode, parentRadius: number): RadialNode {
    const step = anyLength ? (node.length ?? 0) : 1;
    const radius = parentRadius + step;

    if (node.children.length === 0) {
      const laid: RadialNode = {
        node,
        // Evenly spaced around the full circle. Index rather than a swept
        // fraction, so the gap between the last tip and the first is the same
        // as every other gap instead of collapsing to zero.
        angle: (nextIndex / tipCount) * Math.PI * 2,
        radius,
        children: [],
      };
      nextIndex += 1;
      tips.push(laid);
      return laid;
    }

    const children = node.children.map((child) => place(child, radius));
    // The midpoint of the clade's angular span, matching how `layoutTree`
    // centres an internal node between its children.
    const first = children[0].angle;
    const last = children[children.length - 1].angle;
    return { node, angle: (first + last) / 2, radius, children };
  }

  const laidRoot = place(root, 0);
  const maxRadius = tips.reduce((max, tip) => Math.max(max, tip.radius), 0);

  return {
    root: laidRoot,
    tips,
    maxRadius: maxRadius > 0 ? maxRadius : 1,
    rooted: isRooted(root),
    cladogram: !anyLength,
  };
}

/** Every node of a radial layout, root first. */
export function walkRadial(node: RadialNode): RadialNode[] {
  return [node, ...node.children.flatMap(walkRadial)];
}
