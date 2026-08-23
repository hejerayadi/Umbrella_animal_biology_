export type UserRole = "ADMIN" | "BIOLOGIST";
export type UserStatus =
  | "PENDING_EMAIL"
  | "PENDING_APPROVAL"
  | "INVITED"
  | "ACTIVE"
  | "REJECTED"
  | "DISABLED";

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
  email: string;
  role: UserRole;
  status: UserStatus;
  full_name: string;
  institution: string;
  professional_title: string;
  country: string;
  orcid?: string | null;
  motivation: string;
  specialties: string[];
  email_verified_at?: string | null;
  last_login_at?: string | null;
  created_at: string;
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

/**
 * One ranked taxonomic label from the Multimodal Recognition Agent.
 *
 * `classificationScore` is a RANKING score over BioCLIP-2's label set, not a
 * calibrated probability - the agent says so in its own provenance and the UI
 * has to keep saying it. Displaying it as "% confident" would invent a
 * statistical claim the model never made.
 */
export interface RecognitionCandidate {
  speciesId: string;
  scientificName: string;
  commonName?: string | null;
  rank?: string | null;
  classificationScore: number;
  gbifId?: number | null;
  ncbiTaxId?: number | null;
  /**
   * "verified" = both GBIF and NCBI matched, "partial" = one did,
   * "unverified"/"mock_verified" = no live double match. Annotation only:
   * taxonomy never reorders or rescores a candidate.
   */
  taxonomyStatus?: string | null;
}

/**
 * Where the answer came from - the "source of knowledge" behind the ranking.
 *
 * Deliberately carries the mock/real flags. The agent refuses to let fixture
 * predictions be presented under a production banner, and this panel is the
 * last place that promise could be broken.
 */
export interface RecognitionProvenance {
  modelTarget?: string | null;
  modelVersion?: string | null;
  provider?: string | null;
  /** e.g. "remote_bioclip2_open_domain_species" or "mock_classification". */
  recognitionMode?: string | null;
  mockProviderVersion?: string | null;
  /** The Hugging Face Space that actually ran inference, and its revision. */
  remoteSpaceId?: string | null;
  remoteSpaceRevision?: string | null;
  /** "real" = live lookup, "mock" = committed fixture. */
  gbifMode?: string | null;
  ncbiMode?: string | null;
  taxonomyExecuted?: boolean;
  taxonomyDegraded?: boolean;
  scoreIsProbability?: boolean;
  scoreKind?: string | null;
  reasoningLlmProvider?: string | null;
  reasoningLlmCalls?: number | null;
}

/** The Multimodal Recognition Agent's contribution to one answer. */
export interface RecognitionResult {
  /** "identified" | "uncertain" | "not_identified". */
  decision: string;
  species?: string | null;
  speciesId?: string | null;
  gbifId?: number | null;
  ncbiTaxId?: number | null;
  candidates: RecognitionCandidate[];
  provenance: RecognitionProvenance;
  clarificationQuestion?: string | null;
}

/** Which of the Biodiversity Agent's four skills produced a map. */
export type BiodiversitySkill =
  | "distribution"
  | "hotspots"
  | "habitat"
  | "migration"
  | "unknown";

/** One headline figure shown beside a map, already formatted for display. */
export interface BiodiversityMapStat {
  label: string;
  value: string;
}

export interface BiodiversityMapSpec {
  /**
   * An http(s) URL served by the Biodiversity Agent (`GET /maps/{name}`).
   * Never a `file://` path - a page served over http cannot load one, which
   * is exactly why maps used not to appear at all.
   */
  url: string;
  skill: BiodiversitySkill;
  speciesName?: string | null;
  region?: string | null;
  stats: BiodiversityMapStat[];
  /** The worker agents credited with the answer, for the provenance line. */
  sourceAgents: string[];
  /**
   * True when the answering worker is still a placeholder returning curated
   * fixtures rather than measured data. Shown on the panel, following the
   * same rule the Recognition panel follows: fixture data is never presented
   * as measurement. Remove the entry in `PLACEHOLDER_AGENTS`
   * (orchestrator-client.ts) when the real worker ships.
   */
  isIllustrative: boolean;
}

/** One species' bar in a genome-size comparison chart. */
export interface GenomeComparison {
  scientificName?: string | null;
  commonName?: string | null;
  genomeSizeBp?: number | null;
  assemblyId?: string | null;
}

export interface GenomeChartSpec {
  /** Raw SVG markup, rendered as an image so it cannot execute anything. */
  svg: string;
  speciesName?: string | null;
  assemblyId?: string | null;
  genomeSizeBp?: number | null;
  chromosomeCount?: number | null;
  /**
   * "Chromosome", "Scaffold", "Contig"... Carried because it qualifies
   * `chromosomeCount`: a scaffold-level assembly reports the chromosomes it
   * managed to assemble, not the species' karyotype, and the polar bear's
   * "2" means the former. Shown next to the count so it cannot be read as
   * the latter.
   */
  assemblyLevel?: string | null;
  comparisons: GenomeComparison[];
  /** The agent's own caveat about this chart, when it set one. */
  note?: string | null;
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
  /**
   * URL of an illustration the Image Generation Agent produced for this answer.
   * A URL for the same reason as `imageUrl` above: the agent hands back a
   * ~440 KB base64 data URI, and these messages are persisted to localStorage.
   * May 404 once the backend's bounded store evicts it, in which case the
   * answer renders without the picture.
   */
  generatedImageUrl?: string;
  /**
   * The ranked species candidates, when the Multimodal Recognition Agent
   * contributed to this answer. Kept on the message like `proteinViewer`
   * above so it survives a reload - it is a handful of names, scores and
   * integers, nothing like the size of an image.
   */
  recognition?: RecognitionResult;
  /**
   * The map the Biodiversity Agent rendered for this answer, when it ran.
   * Only the URL and a few summary figures are kept - the map itself is a
   * ~700 KB self-contained folium document served by the agent, fetched by
   * the iframe when the panel mounts, and never persisted with the message.
   */
  biodiversityMap?: BiodiversityMapSpec;
  /**
   * The chart the Genome Agent rendered for this answer, when it ran.
   * The SVG travels inline because it is small (~1.4 KB for a size
   * comparison, ~9.5 KB for a chromosome map) - unlike the biodiversity map,
   * which is a 700 KB document fetched from the agent by URL.
   */
  genomeChart?: GenomeChartSpec;
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
