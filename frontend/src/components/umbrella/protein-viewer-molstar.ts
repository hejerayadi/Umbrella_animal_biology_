/**
 * The Mol* half of the protein viewer, kept in its own module on purpose.
 *
 * Mol* and its stylesheet are several megabytes. `protein-viewer.tsx` is
 * imported by every chat message, so importing Mol* there would put a 3D
 * engine in the bundle of a conversation that never mentions a protein. This
 * module is only ever reached through a dynamic `import()`, which makes Vite
 * split it into a chunk that is fetched the first time a structure is shown.
 *
 * The Mol* API is imperative and owns its own DOM, so this file exposes a
 * mount/dispose pair rather than a component: React contributes the container
 * element and nothing else inside it.
 */
// Deliberately `apps/viewer/app`, not the `apps/viewer` barrel next to it.
// The barrel is the entry point of Mol*'s own standalone web app, so it also
// imports index.html, a favicon and `skin/light.scss` - which fails the build
// with "Preprocessor dependency sass-embedded not found" unless a Sass
// toolchain is added for markup we do not use. `app.js` is where the Viewer
// class itself lives, with none of that shell.
import { Viewer } from "molstar/lib/apps/viewer/app";

// The stylesheet the barrel would have brought in, prebuilt as plain CSS.
import "molstar/build/viewer/molstar.css";

import type { ProteinStructureRef } from "@/lib/umbrella-types";

/** Mol*'s name for each format the Protein agent reports. */
const TRAJECTORY_FORMATS = { MMCIF: "mmcif", PDB: "pdb" } as const;

export interface MountedViewer {
  dispose: () => void;
}

/**
 * Render one structure into `container`.
 *
 * Resolves once the structure is on screen, so the caller can keep showing a
 * loading state until then. Rejects when the file cannot be fetched or parsed,
 * which is a real possibility: the URLs point at RCSB and AlphaFold, not at us.
 */
export async function mountStructureViewer(
  container: HTMLElement,
  structure: ProteinStructureRef,
): Promise<MountedViewer> {
  const viewer = await Viewer.create(container, {
    // An embedded panel in a chat transcript, not the full Mol* workbench:
    // the tree, sequence and log panels would each be taller than the canvas.
    layoutIsExpanded: false,
    layoutShowControls: false,
    layoutShowSequence: false,
    layoutShowLog: false,
    layoutShowLeftPanel: false,
    collapseLeftPanel: true,
    collapseRightPanel: true,
    // Keep only the viewport buttons that make a 3D view usable at this size:
    // reset the camera after the user tumbles it, and open it full-screen when
    // a domain is too small to read in a chat column.
    viewportShowExpand: true,
    viewportShowReset: true,
    viewportShowSelectionMode: false,
    viewportShowAnimation: false,
  });

  const format = TRAJECTORY_FORMATS[structure.format] ?? "mmcif";

  try {
    await viewer.loadStructureFromUrl(structure.url, format, false);
  } catch (error) {
    // A half-initialised plugin would leak its WebGL context and its resize
    // observers for the lifetime of the page.
    viewer.dispose();
    throw error;
  }

  return { dispose: () => viewer.dispose() };
}
