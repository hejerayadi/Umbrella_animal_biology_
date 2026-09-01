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

/**
 * A piece of academic text the Literature Agent wrote, and/or the venues it
 * suggested for it.
 *
 * Shown in its own panel rather than inside the answer text. The draft is not
 * a finding *about* the user's question the way a genome size is - it is the
 * deliverable itself, meant to be read as a unit, copied, and pasted into a
 * manuscript. Run together with the surrounding prose there is no boundary
 * showing where the generated text starts and stops.
 */
export interface WritingDraftSpec {
  /** "Abstract", "Introduction", "Related work"... Names what was written. */
  section: string;
  /** The generated text itself. Absent when only venues were requested. */
  draft?: string | null;
  /** Markdown list of suggested journals, when publication support ran. */
  recommendedJournals?: string | null;
  /** Citation lines the draft was allowed to use. Empty means it cites nothing. */
  referencesUsed: string[];
  /**
   * The references were placeholders, not real publications. Surfaced in the
   * panel because a draft that looks citable but is not is the one failure
   * this platform most needs to make visible.
   */
  referencesArePlaceholder: boolean;
  /** Whether the grammar/coherence pass actually ran on the draft. */
  styleCorrected: boolean;
  /** The agent's own caveat, when it set one. */
  notice?: string | null;
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
  /**
   * The text the Literature Agent wrote for this answer, when it ran. A few
   * kilobytes of prose, so it travels with the message like `genomeChart`
   * rather than being fetched.
   */
  writingDraft?: WritingDraftSpec;
  /**
   * The Reconstruction Agent's result. Travels with the message like
   * `genomeChart`: a handful of gaps with short fills, measured in kilobytes.
   */
  reconstruction?: ReconstructionSpec;
  /**
   * The Evolution Agent's tree and/or similarity network. Travels with the
   * message like `genomeChart`: a Newick string and a few dozen numbers.
   */
  evolution?: EvolutionSpec;
}

/** One pairwise similarity score from the Molecular Comparison sub-agent. */
export interface SimilarityScore {
  speciesA: string;
  speciesB: string;
  /**
   * Centred cosine similarity in [-1, 1] - NOT a percentage identity.
   * Embeddings are mean-centred across the species in the request, so a score
   * is a position relative to the others and is routinely negative. Rendering
   * it as a percentage bar is wrong.
   */
  score: number;
}

/**
 * The similarity graph the Molecular Comparison sub-agent actually built.
 *
 * The agent returns this as `similarity_network` (NetworkX node-link form) in
 * addition to the flat `similarity_scores` list. It is the authoritative
 * membership list: a species with no comparable sequence appears as a node
 * with no edges, which the flat score list cannot express at all.
 */
export interface SimilarityNetwork {
  /** Every species in the graph, including any with no edges. */
  species: string[];
  edges: SimilarityScore[];
}

/** A cluster of species the comparison put together. */
export interface SpeciesGroup {
  groupId: number;
  species: string[];
  meanScore: number | null;
}

/** The Evolution Agent's result, shaped for display. */
export interface EvolutionSpec {
  /** Canonical scientific names that were analysed. */
  speciesList: string[];
  /** Newick from IQ-TREE, leaf labels quoted. Absent on the molecular branch. */
  newick: string | null;
  /** Substitution model ModelFinder chose, e.g. "MTREV+G4". */
  model: string | null;
  /** Mean UFBoot support in [0, 1], or null when UFBoot did not run. */
  overallConfidence: number | null;
  similarityScores: SimilarityScore[];
  /**
   * The agent's own graph, when it sent one. Null falls back to a graph
   * derived from `similarityScores`, which is the same edges without the
   * isolated nodes.
   */
  similarityNetwork: SimilarityNetwork | null;
  speciesGroups: SpeciesGroup[];
  /** The Explainer's prose (LLM #2), when it produced a trustworthy one. */
  interpretation: string | null;
  /** `ufboot_not_run`, `offline_sequence_fallback`, `*_failed`, ... */
  warnings: string[];
}

/** One score behind a candidate's confidence, as the agent reported it. */
export interface ReconstructionScore {
  label: string;
  /** Null when the signal was not consulted at all - see `evo2`. */
  value: number | null;
}

/** One unresolved region, and what became of it. */
export interface ReconstructedGap {
  gapId: string;
  /** 1-based inclusive, as the Genome Agent reports coordinates. */
  start: number | null;
  end: number | null;
  lengthBp: number | null;
  /** "resolved", "skipped", "unresolved"... straight from the agent. */
  status: string;
  resolved: boolean;
  /**
   * Why it was not filled - "deadline_exceeded",
   * "insufficient_gap_spanning_homologs". Present only when unresolved, and
   * shown as the reason rather than as a failure: refusing to invent bases is
   * the correct outcome when nothing spans the gap.
   */
  unresolvedReason: string | null;
  explanation: string | null;

  /** The accepted fill. Null unless `resolved`. */
  fill: string | null;
  fillLengthBp: number | null;
  confidence: number | null;
  confidenceLevel: string | null;
  /**
   * Whether a genome model wrote these bases rather than a sequenced relative.
   * The single most consequential fact about a fill, and the panel gives it
   * its own colour rather than a footnote.
   */
  isModelGenerated: boolean;
  supportingOrganisms: string[];
  supportingHits: string[];
  scores: ReconstructionScore[];

  /**
   * Resolved sequence either side of the gap, from the Genome Agent's handoff.
   * Absent when the reconstruction response is read without that context - the
   * agent's own per-gap output carries coordinates but not flanks.
   */
  leftFlank: string | null;
  rightFlank: string | null;
}

/**
 * How the gaps that were worked on were chosen, when the Genome Agent said.
 *
 * Without this a list of ten gaps reads as "this assembly has ten gaps". It
 * had thirty on one record of several thousand.
 */
export interface ReconstructionSelection {
  gapsFound: number | null;
  gapsOverFloor: number | null;
  gapsSelected: number | null;
  recordsInAssembly: number | null;
  recordsOverCeiling: number | null;
  minGapBp: number | null;
  assemblyGapBasesBp: number | null;
  assemblyGapFraction: number | null;
}

/** The Reconstruction Agent's result, shaped for display. */
export interface ReconstructionSpec {
  /** "completed", "partially_completed", "failed". */
  status: string;
  summary: string | null;
  scientificName: string | null;
  assemblyId: string | null;
  sequenceAccession: string | null;
  requestedGaps: number;
  resolvedGaps: number;
  unresolvedGaps: number;
  gaps: ReconstructedGap[];
  selection: ReconstructionSelection | null;
  warnings: string[];
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
