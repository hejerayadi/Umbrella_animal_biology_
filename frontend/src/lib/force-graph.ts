/**
 * A small deterministic force-directed layout, for the similarity network.
 *
 * Written here rather than pulled in from d3-force for two reasons. The graph
 * is tiny - the Evolution Agent compares a handful of species, and the network
 * it returns is complete, so `n` is under twenty and `edges` is `n(n-1)/2`.
 * And the layout has to be *deterministic*: the panel re-renders whenever the
 * threshold moves, and a simulation seeded from `Math.random` would reshuffle
 * the whole picture on every tick of the slider, which reads as the data
 * changing when only the filter did.
 *
 * The algorithm is Fruchterman-Reingold with weighted springs: repulsion
 * between every pair, attraction along edges scaled by similarity, and a
 * linearly cooling temperature. It runs to convergence synchronously - a few
 * hundred iterations over twenty nodes is well under a frame.
 */

export interface GraphEdge {
  source: string;
  target: string;
  /** Edge strength in [0, 1]. Stronger edges pull their endpoints closer. */
  weight: number;
}

export interface Point {
  x: number;
  y: number;
}

export interface ForceOptions {
  /** Layout box. Positions come back inside it, with `padding` to spare. */
  width: number;
  height: number;
  padding?: number;
  iterations?: number;
}

/**
 * Position every node. Returns a map from node id to a point in the box.
 *
 * Nodes with no surviving edges are still placed: an isolated species is a
 * finding ("nothing in this batch is close to it"), and dropping it from the
 * picture would hide exactly that.
 */
export function layoutForce(
  nodes: string[],
  edges: GraphEdge[],
  options: ForceOptions,
): Map<string, Point> {
  const { width, height, padding = 48, iterations = 400 } = options;
  const positions = new Map<string, Point>();

  if (nodes.length === 0) return positions;
  if (nodes.length === 1) {
    positions.set(nodes[0], { x: width / 2, y: height / 2 });
    return positions;
  }

  const innerWidth = Math.max(width - padding * 2, 1);
  const innerHeight = Math.max(height - padding * 2, 1);
  const area = innerWidth * innerHeight;
  // Fruchterman-Reingold's ideal edge length: the side of the square each node
  // would get if the nodes were spread evenly over the area.
  const k = Math.sqrt(area / nodes.length);

  // Seeded on a circle by index, so the same graph always lays out the same
  // way and moving the threshold slider only removes edges - it never
  // re-scrambles the nodes.
  const startRadius = Math.min(innerWidth, innerHeight) / 2.4;
  nodes.forEach((id, index) => {
    const angle = (index / nodes.length) * Math.PI * 2;
    positions.set(id, {
      x: width / 2 + Math.cos(angle) * startRadius,
      y: height / 2 + Math.sin(angle) * startRadius,
    });
  });

  // Edges referencing a node that is not in `nodes` would silently pull the
  // layout toward a point that is never drawn.
  const known = new Set(nodes);
  const live = edges.filter((edge) => known.has(edge.source) && known.has(edge.target));

  const displacement = new Map<string, Point>();
  let temperature = Math.min(innerWidth, innerHeight) / 10;
  const cooling = temperature / (iterations + 1);

  for (let step = 0; step < iterations; step += 1) {
    for (const id of nodes) displacement.set(id, { x: 0, y: 0 });

    // Repulsion: every pair pushes apart, harder the closer they are.
    for (let i = 0; i < nodes.length; i += 1) {
      for (let j = i + 1; j < nodes.length; j += 1) {
        const a = positions.get(nodes[i])!;
        const b = positions.get(nodes[j])!;
        let dx = a.x - b.x;
        let dy = a.y - b.y;
        let distance = Math.hypot(dx, dy);

        if (distance < 0.01) {
          // Two nodes exactly on top of each other have no direction to
          // separate along. Nudge them apart deterministically by index
          // rather than randomly, to keep the layout reproducible.
          dx = (i - j) * 0.01;
          dy = 0.01;
          distance = Math.hypot(dx, dy);
        }

        const force = (k * k) / distance;
        const ux = (dx / distance) * force;
        const uy = (dy / distance) * force;

        const da = displacement.get(nodes[i])!;
        const db = displacement.get(nodes[j])!;
        da.x += ux;
        da.y += uy;
        db.x -= ux;
        db.y -= uy;
      }
    }

    // Attraction along edges, scaled by weight: a strongly similar pair sits
    // closer than a weakly similar one, which is the whole point of drawing
    // this as a graph rather than a list.
    for (const edge of live) {
      const a = positions.get(edge.source)!;
      const b = positions.get(edge.target)!;
      const dx = a.x - b.x;
      const dy = a.y - b.y;
      const distance = Math.max(Math.hypot(dx, dy), 0.01);

      const force = ((distance * distance) / k) * Math.max(edge.weight, 0.05);
      const ux = (dx / distance) * force;
      const uy = (dy / distance) * force;

      const da = displacement.get(edge.source)!;
      const db = displacement.get(edge.target)!;
      da.x -= ux;
      da.y -= uy;
      db.x += ux;
      db.y += uy;
    }

    // Weak pull to the centre, so a disconnected node drifts to the edge of
    // the picture instead of off it.
    for (const id of nodes) {
      const point = positions.get(id)!;
      const d = displacement.get(id)!;
      d.x += (width / 2 - point.x) * 0.012;
      d.y += (height / 2 - point.y) * 0.012;
    }

    // Move, capped by the temperature, then cool.
    for (const id of nodes) {
      const point = positions.get(id)!;
      const d = displacement.get(id)!;
      const distance = Math.max(Math.hypot(d.x, d.y), 0.01);
      const limited = Math.min(distance, temperature);
      point.x += (d.x / distance) * limited;
      point.y += (d.y / distance) * limited;
    }

    temperature -= cooling;
  }

  // Fit the converged layout to the box. Scaling afterwards rather than
  // clamping during the run keeps the relative distances - which are the
  // information - intact.
  return fit(positions, width, height, padding);
}

function fit(
  positions: Map<string, Point>,
  width: number,
  height: number,
  padding: number,
): Map<string, Point> {
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;

  for (const point of positions.values()) {
    minX = Math.min(minX, point.x);
    maxX = Math.max(maxX, point.x);
    minY = Math.min(minY, point.y);
    maxY = Math.max(maxY, point.y);
  }

  const spanX = maxX - minX;
  const spanY = maxY - minY;
  const targetWidth = Math.max(width - padding * 2, 1);
  const targetHeight = Math.max(height - padding * 2, 1);

  // One scale for both axes, so the layout is not stretched into a shape the
  // distances do not mean.
  const scale = Math.min(
    spanX > 0.01 ? targetWidth / spanX : Infinity,
    spanY > 0.01 ? targetHeight / spanY : Infinity,
  );
  const factor = Number.isFinite(scale) ? scale : 1;

  const offsetX = (width - spanX * factor) / 2 - minX * factor;
  const offsetY = (height - spanY * factor) / 2 - minY * factor;

  const fitted = new Map<string, Point>();
  for (const [id, point] of positions) {
    fitted.set(id, { x: point.x * factor + offsetX, y: point.y * factor + offsetY });
  }
  return fitted;
}
