export type UserRole =
  | "Student"
  | "Researcher"
  | "Professor"
  | "Conservation Scientist"
  | "Bioinformatician"
  | "Developer"
  | "Other";

export const USER_ROLES: UserRole[] = [
  "Student",
  "Researcher",
  "Professor",
  "Conservation Scientist",
  "Bioinformatician",
  "Developer",
  "Other",
];

export const AI_CAPABILITIES = [
  "Genome Reconstruction",
  "Biodiversity Analysis",
  "Evolutionary Studies",
  "Trait Discovery",
  "Protein Visualization",
  "Scientific Literature",
  "Species Identification",
] as const;

export type AiCapability = (typeof AI_CAPABILITIES)[number];

export interface User {
  id: string;
  name: string;
  email: string;
  role: UserRole;
  purpose: string;
  mainInterest: string;
  goals: string;
  researchInterests: AiCapability[];
  createdAt: string;
}

export interface Conversation {
  id: string;
  title: string;
  createdAt: string;
  updatedAt: string;
}

/**
 * The 3D structure the Protein Visualization Agent selected.
 *
 * Mirrors `molstar_config.structure` from the agent's viewer capability
 * (`app/capabilities/visualization/service.py`). `url` points straight at RCSB
 * or AlphaFold - the backend never proxies the coordinates, so the browser
 * fetches them from the source that published them.
 */
export interface ProteinStructureRef {
  id: string;
  url: string;
  format: "MMCIF" | "PDB";
  /** "RCSB_PDB" for a solved structure, "ALPHAFOLD_DB" for a predicted model. */
  source: string;
  structure_type: "EXPERIMENTAL" | "PREDICTED";
  chain_id?: string | null;
}

/** One residue the agent could map onto the structure and wants marked. */
export interface ProteinSelection {
  label: string;
  chain?: string | null;
  residue_number?: string | null;
  color?: string | null;
}

/**
 * A functional region from InterPro or UniProt.
 *
 * `applicable` is the part that matters scientifically: domain spans are in
 * UniProt numbering, which only lines up with the structure's own numbering
 * for predicted models. On a solved structure the agent reports the span
 * untouched and marks it inapplicable rather than translating it without a
 * residue-level mapping.
 */
export interface ProteinDomain {
  label: string;
  accession: string;
  kind: string;
  start: number;
  end: number;
  coordinate_space: "structure" | "uniprot";
  applicable: boolean;
}

/** The whole Mol* scene, as the agent returns it under `context.protein_viewer`. */
export interface ProteinViewerSpec {
  viewer: "molstar";
  structure: ProteinStructureRef;
  representation?: { type?: string; color_theme?: string };
  selections: ProteinSelection[];
  domains: ProteinDomain[];
}

export type MessageSender = "user" | "assistant";

export interface Message {
  id: string;
  conversationId: string;
  sender: MessageSender;
  content: string;
  timestamp: string;
  /**
   * URL of an image the user attached, served by the backend. A URL rather
   * than the image data because messages are persisted to localStorage, and
   * one base64 photo would exhaust its ~5 MB quota. May 404 once the backend
   * evicts the image; the message renders without it in that case.
   */
  imageUrl?: string;
  /** The original filename, for the image's alt text. */
  imageName?: string;
  /**
   * The Mol* scene, when the Protein Visualization Agent contributed one to
   * this answer. Small enough for localStorage - it is identifiers, spans and
   * a URL, and the coordinates themselves are fetched from RCSB or AlphaFold
   * when the viewer mounts.
   */
  proteinViewer?: ProteinViewerSpec;
}

export type AgentStatus = "pending" | "running" | "complete" | "failed";

export interface AgentActivity {
  id: string;
  conversationId: string;
  agentName: string;
  status: AgentStatus;
  description: string;
  timestamp: string;
}
