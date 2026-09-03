import type {
  EvolutionSpec,
  SimilarityNetwork,
  SimilarityScore,
  SpeciesGroup,
  AgentActivity,
  AgentStatus,
  BiodiversityMapSpec,
  BiodiversityMapStat,
  BiodiversitySkill,
  GenomeChartSpec,
  GenomeComparison,
  ProteinDomain,
  ProteinSelection,
  ProteinViewerSpec,
  RecognitionCandidate,
  RecognitionProvenance,
  RecognitionResult,
  ReconstructedGap,
  ReconstructionScore,
  ReconstructionSelection,
  ReconstructionSpec,
  WritingDraftSpec,
} from "./umbrella-types";
import { ApiClientError, apiRequest, apiStream, apiUrl } from "./api-client";

/**
 * Base URL of the Python orchestrator API (backend/api.py).
 * Override with VITE_ORCHESTRATOR_API_URL in a .env file if it's not running
 * on the default local port.
 */
export interface ChatResponse {
  answer: string;
  execution_history: string[];
  context: Record<string, unknown>;
  /**
   * Set when an agent generated an illustration. A path into the orchestrator
   * API ("/api/upload/<id>"), never the image data: FLUX replies with a ~440 KB
   * base64 data URI, and messages are persisted to localStorage.
   */
  image_url?: string | null;
}

/** What POST /api/upload returns once it has accepted an image. */
export interface UploadedImage {
  image_id: string;
  /** The context key the backend expects this id under - never hardcode it. */
  context_key: string;
  media_type: string;
  filename: string;
  size_bytes: number;
}

/** Image formats the backend accepts. Used to filter the OS file picker. */
export const ACCEPTED_IMAGE_TYPES = ["image/jpeg", "image/png", "image/webp"];

/**
 * Where an uploaded image can be displayed from.
 *
 * Used instead of the local blob: URL once a message is sent: blob URLs are
 * revoked when the composer clears and would not survive a page reload, while
 * this one is fetched back from the server.
 */
export function imageUrlFor(imageId: string): string {
  return apiUrl(`/api/v1/uploads/${imageId}`);
}

/**
 * Uploads one image and returns the handle to attach to the next message.
 *
 * The image itself never travels in the chat request: the orchestrator would
 * broadcast the whole context to every agent and render it into an LLM prompt.
 * Only this short id goes with the message, and the backend swaps it back for
 * the bytes for the one agent that can read them.
 */
export async function uploadImage(file: File): Promise<UploadedImage> {
  const body = new FormData();
  body.append("file", file);

  return apiRequest<UploadedImage>("/api/v1/uploads", {
    method: "POST",
    body, // no Content-Type header: the browser sets the multipart boundary
  });
}

/**
 * Sends one chat message to the orchestrator and returns its final result.
 *
 * `image` is the handle from `uploadImage`, when the user attached one. It is
 * placed under the key the backend named in its own response, so the two sides
 * cannot drift apart.
 */
export async function askOrchestrator(
  query: string,
  image?: UploadedImage | null,
): Promise<ChatResponse> {
  const context: Record<string, unknown> = image ? { [image.context_key]: image.image_id } : {};

  return apiRequest<ChatResponse>("/api/v1/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, context }),
  });
}

/**
 * One thing the orchestrator did, sent the moment it happened.
 *
 * `step` arrives twice for the same `id` - once "running", once with the
 * outcome - so it is rendered as one line that updates rather than two.
 * Mirrors `backend/orchestrator/events.py`; keep the two in step.
 */
export type OrchestratorEvent =
  | { type: "step"; id: string; agent: string; text: string; state: "running" | "done" | "failed" }
  | { type: "thought"; id: string; text: string }
  | { type: "answer"; delta: string }
  | { type: "done"; payload: ChatResponse }
  | { type: "error"; message: string };

/**
 * Sends one chat message and reports the orchestrator's progress as it runs.
 *
 * `onEvent` fires for every step, thought and answer chunk, in order; the
 * promise resolves with the same complete result `askOrchestrator` returns, so
 * a caller that only wants the final answer can ignore the callback entirely.
 *
 * Falls back to the blocking `/chat` route when the backend has no streaming
 * one - the deployed API and the frontend are updated separately, and a 404
 * here should cost the live commentary, not the answer.
 */
export async function streamOrchestrator(
  query: string,
  image: UploadedImage | null | undefined,
  onEvent: (event: OrchestratorEvent) => void,
): Promise<ChatResponse> {
  const context: Record<string, unknown> = image ? { [image.context_key]: image.image_id } : {};

  let response: Response;
  try {
    response = await apiStream("/api/v1/chat/stream", {
      method: "POST",
      body: JSON.stringify({ query, context }),
    });
  } catch (error) {
    if (error instanceof ApiClientError && error.status === 404) {
      return askOrchestrator(query, image);
    }
    throw error;
  }

  if (!response.body) {
    // No streaming body to read (a proxy that buffers, an old browser). The
    // question still deserves an answer.
    return askOrchestrator(query, image);
  }

  let result: ChatResponse | null = null;
  let failure: string | null = null;

  for await (const event of readServerSentEvents(response.body)) {
    if (event.type === "done") result = event.payload;
    else if (event.type === "error") failure = event.message;
    onEvent(event);
  }

  if (failure) throw new Error(failure);
  if (!result) throw new Error("The orchestrator closed the connection without answering.");
  return result;
}

/**
 * Splits an SSE body into the events it carries.
 *
 * Frames are separated by a blank line and a single frame can straddle two
 * network chunks, so the tail of a chunk is carried over rather than parsed.
 * Lines starting with ":" are comments - the backend sends those as a
 * keep-alive while a slow agent works, and they are not events.
 */
async function* readServerSentEvents(
  body: ReadableStream<Uint8Array>,
): AsyncGenerator<OrchestratorEvent> {
  const reader = body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      let split = buffer.indexOf("\n\n");
      while (split !== -1) {
        const frame = buffer.slice(0, split);
        buffer = buffer.slice(split + 2);
        const event = parseFrame(frame);
        if (event) yield event;
        split = buffer.indexOf("\n\n");
      }
    }
  } finally {
    reader.releaseLock();
  }
}

function parseFrame(frame: string): OrchestratorEvent | null {
  const data = frame
    .split("\n")
    .filter((line) => line.startsWith("data:"))
    .map((line) => line.slice(5).trim())
    .join("\n");
  if (!data) return null;
  try {
    return JSON.parse(data) as OrchestratorEvent;
  } catch {
    // A truncated or malformed frame is not worth killing the run over: the
    // next one is probably fine, and `done` carries the whole result anyway.
    return null;
  }
}

/**
 * Absolute URL for an image the backend generated, or undefined if there wasn't one.
 *
 * The backend returns a root-relative path so it does not have to know what host
 * it is reached on; the API lives on a different origin from the dev server, so
 * it has to be resolved against the orchestrator's base URL rather than the page.
 */
export function generatedImageFrom(response: ChatResponse): string | undefined {
  const path = response.image_url;
  if (typeof path !== "string" || !path) return undefined;
  if (path.startsWith("http://") || path.startsWith("https://")) return path;
  return apiUrl(path);
}

/** The context key the Protein Visualization Agent publishes its Mol* scene under. */
const PROTEIN_VIEWER_KEY = "protein_viewer";

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

/**
 * Reads the Mol* scene out of a chat response's shared context.
 *
 * Returns undefined whenever the agent did not run, could not find a
 * structure, or answered with a shape this build does not recognise. The
 * checks are not ceremony: `context` is a free-form bag that every agent
 * writes into, so the key can be absent, null, or - the day an agent changes
 * its output - something else entirely. A missing viewer costs the user the
 * 3D panel; a malformed one handed to Mol* would take down the whole message.
 */
export function proteinViewerFrom(
  context: Record<string, unknown>,
): ProteinViewerSpec | undefined {
  const raw = context[PROTEIN_VIEWER_KEY];
  if (!isRecord(raw) || raw.viewer !== "molstar") return undefined;

  const structure = raw.structure;
  if (!isRecord(structure)) return undefined;
  if (typeof structure.url !== "string" || typeof structure.id !== "string") return undefined;

  // Anything other than the two formats Mol* is told to expect is dropped
  // rather than guessed at - loading mmCIF as PDB fails deep inside the parser.
  const format = structure.format === "PDB" ? "PDB" : structure.format === "MMCIF" ? "MMCIF" : null;
  if (!format) return undefined;

  return {
    viewer: "molstar",
    structure: {
      id: structure.id,
      url: structure.url,
      format,
      source: typeof structure.source === "string" ? structure.source : "unknown",
      structure_type: structure.structure_type === "PREDICTED" ? "PREDICTED" : "EXPERIMENTAL",
      chain_id: typeof structure.chain_id === "string" ? structure.chain_id : null,
    },
    representation: isRecord(raw.representation)
      ? {
          type: typeof raw.representation.type === "string" ? raw.representation.type : undefined,
          color_theme:
            typeof raw.representation.color_theme === "string"
              ? raw.representation.color_theme
              : undefined,
        }
      : undefined,
    selections: Array.isArray(raw.selections)
      ? (raw.selections.filter(isRecord) as unknown as ProteinSelection[])
      : [],
    domains: Array.isArray(raw.domains)
      ? (raw.domains.filter(
          (domain): domain is Record<string, unknown> =>
            isRecord(domain) && typeof domain.label === "string",
        ) as unknown as ProteinDomain[])
      : [],
  };
}

/** The context keys the Multimodal Recognition Agent publishes under. */
const RECOGNITION_KEY = "recognition";
const RECOGNITION_CANDIDATES_KEY = "recognition_candidates";
const RECOGNITION_PROVENANCE_KEY = "recognition_provenance";

function asNumber(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) ? value : null;
}

function asString(value: unknown): string | null {
  return typeof value === "string" && value ? value : null;
}

/**
 * Reads the Recognition agent's ranked candidates out of the shared context.
 *
 * Same defensiveness as `proteinViewerFrom`: `context` is a free-form bag that
 * every agent writes into, so every key can be absent or a different shape
 * than this build expects. A dropped panel costs the user the score table; a
 * malformed one thrown at the renderer would take down the whole message.
 *
 * Returns undefined when the agent did not run. It deliberately does NOT
 * return undefined for `not_identified` - "no species is strong enough" is a
 * real scientific answer and the user should still see what was considered.
 */
export function recognitionFrom(
  context: Record<string, unknown>,
): RecognitionResult | undefined {
  const recognition = context[RECOGNITION_KEY];
  if (!isRecord(recognition)) return undefined;

  const decision = asString(recognition.decision);
  if (!decision) return undefined;

  const rawCandidates = Array.isArray(context[RECOGNITION_CANDIDATES_KEY])
    ? (context[RECOGNITION_CANDIDATES_KEY] as unknown[])
    : [];

  const candidates: RecognitionCandidate[] = rawCandidates
    .filter(isRecord)
    .map((raw) => ({
      speciesId: asString(raw.species_id) ?? "",
      scientificName: asString(raw.scientific_name) ?? "",
      commonName: asString(raw.common_name),
      rank: asString(raw.rank),
      classificationScore: asNumber(raw.classification_score) ?? 0,
      gbifId: asNumber(raw.gbif_id),
      ncbiTaxId: asNumber(raw.ncbi_taxid),
      taxonomyStatus: asString(raw.taxonomy_status),
    }))
    // A row with no name is not renderable and not informative.
    .filter((candidate) => candidate.scientificName !== "");

  const raw = isRecord(context[RECOGNITION_PROVENANCE_KEY])
    ? (context[RECOGNITION_PROVENANCE_KEY] as Record<string, unknown>)
    : {};

  // Only the fields this panel displays are carried over. `taxonomy_report`
  // in particular is a per-candidate nested dict that nothing here renders,
  // and these messages are persisted to localStorage.
  const provenance: RecognitionProvenance = {
    modelTarget: asString(raw.model_target),
    modelVersion: asString(raw.model_version),
    provider: asString(raw.recognition_provider),
    recognitionMode: asString(raw.recognition_mode),
    mockProviderVersion: asString(raw.mock_provider_version),
    remoteSpaceId: asString(raw.remote_space_id),
    remoteSpaceRevision: asString(raw.remote_space_revision),
    gbifMode: asString(raw.gbif_mode),
    ncbiMode: asString(raw.ncbi_mode),
    taxonomyExecuted: raw.taxonomy_executed === true,
    taxonomyDegraded: raw.taxonomy_degraded === true,
    // Absent means "no promise was made", which must not read as "yes".
    scoreIsProbability: raw.score_is_probability === true,
    scoreKind: asString(raw.score_kind),
    reasoningLlmProvider: asString(raw.reasoning_llm_provider),
    reasoningLlmCalls: asNumber(raw.reasoning_llm_calls),
  };

  return {
    decision,
    species: asString(context.species),
    speciesId: asString(context.species_id),
    gbifId: asNumber(context.gbif_id),
    ncbiTaxId: asNumber(context.ncbi_taxid),
    candidates,
    provenance,
    clarificationQuestion: asString(recognition.clarification_question),
  };
}

/**
 * Turns the orchestrator's raw execution log (e.g. "Planner -> Genome",
 * "Genome -> completed") into the timeline entries the Agent Thinking panel
 * displays.
 */
export function parseExecutionHistory(history: string[], conversationId: string): AgentActivity[] {
  const baseTime = Date.now();

  return history.map((entry, index) => {
    const [left, right] = entry.split(" -> ");
    // "Planner -> Genome" and "Resolver -> Genome" name the agent that was
    // PICKED on the right. Every other entry ("Genome -> completed",
    // "Responder -> answer ready") names the reporting agent on the left.
    const isRoutingStep = left === "Planner" || left === "Resolver";

    return {
      id: `act_${conversationId}_${index}_${baseTime}`,
      conversationId,
      agentName: isRoutingStep ? right : left,
      status: statusFor(right),
      description: describe(left, right),
      // Small increasing offset so each step sorts/displays in order even
      // though they all arrive from the backend at once.
      timestamp: new Date(baseTime + index).toISOString(),
    };
  });
}

/**
 * The backend only replies once the whole workflow has finished, so every
 * step in the history is already over by the time it reaches us. Only a
 * genuine failure is anything other than "complete" - marking steps as
 * "running" here would leave a spinner on screen forever.
 */
function statusFor(right: string): AgentStatus {
  return right === "failed" ? "failed" : "complete";
}

function describe(left: string, right: string): string {
  if (left === "Planner") {
    return right === "Direct answer"
      ? "No research agent needed for this message"
      : "Selected as the starting agent";
  }
  if (left === "Resolver") return "Selected to help a waiting agent";
  if (left === "Responder") {
    return right === "answered directly" ? "Replied directly" : "Wrote the final answer";
  }

  switch (right) {
    case "completed":
      return "Completed";
    case "needs_agent":
      return "Needed another agent's help";
    case "continue":
      return "Continued";
    case "failed":
      return "Failed";
    default:
      return right;
  }
}

/**
 * The context keys the Biodiversity Agent publishes under, per its card.json.
 * The Global Orchestrator merges an agent's `output` straight into the shared
 * context, so these arrive at the top level.
 */
const BIODIVERSITY_MAP_KEY = "map_url";
const BIODIVERSITY_REPORT_KEY = "biodiversity_report";

/**
 * Workers that still answer from curated fixtures rather than measured data.
 *
 * Habitat is the Sprint 4 placeholder: its "regions" are hand-drawn rectangles
 * with hard-coded conservation statuses, not IUCN polygons. The panel says so,
 * for the same reason the Recognition panel discloses a mock classifier -
 * fixture data must never read as measurement. Delete the entry when the real
 * worker ships.
 */
const PLACEHOLDER_AGENTS = ["Habitat Visualization Agent"];

function asPositiveInt(value: unknown): number | null {
  return typeof value === "number" && Number.isFinite(value) && value >= 0
    ? Math.round(value)
    : null;
}

/** Groups digits so a six-figure occurrence count stays readable at a glance. */
function formatCount(value: number): string {
  return value.toLocaleString();
}

/**
 * Which skill answered, inferred from the shape of what came back.
 *
 * Deliberately reads the payload rather than the agent's name: `source_agents`
 * is a display string that can be reworded, while these keys are the worker's
 * actual output contract.
 */
function skillFrom(
  findings: Record<string, unknown>,
  report: Record<string, unknown>,
): BiodiversitySkill {
  if (Array.isArray(report.hotspots) || "ranked_hotspots" in findings) return "hotspots";
  if ("predicted_next" in findings || "migration_route" in report) return "migration";
  if ("conservation_status" in findings) return "habitat";
  if ("coordinates" in findings || "observation_count" in report) return "distribution";
  return "unknown";
}

/** The one or two figures worth showing beside each kind of map. */
function statsFrom(
  skill: BiodiversitySkill,
  findings: Record<string, unknown>,
  report: Record<string, unknown>,
): BiodiversityMapStat[] {
  const stats: BiodiversityMapStat[] = [];

  if (skill === "hotspots") {
    const clusters = Array.isArray(report.hotspots) ? report.hotspots.length : null;
    if (clusters !== null) {
      stats.push({ label: "Hotspots", value: String(clusters) });
    }
    const species = asPositiveInt(findings.species_analysed);
    if (species !== null) stats.push({ label: "Species", value: formatCount(species) });
    const records = asPositiveInt(findings.records_analysed);
    if (records !== null) stats.push({ label: "Records", value: formatCount(records) });
    return stats;
  }

  if (skill === "habitat") {
    const status = asString(findings.conservation_status);
    if (status) stats.push({ label: "IUCN status", value: status });
    const regions = Array.isArray(findings.habitat_regions)
      ? findings.habitat_regions.length
      : null;
    if (regions !== null) stats.push({ label: "Regions", value: String(regions) });
    return stats;
  }

  if (skill === "migration") {
    const next = findings.predicted_next;
    if (isRecord(next)) {
      const lat = asNumber(next.lat);
      const lon = asNumber(next.lon);
      if (lat !== null && lon !== null) {
        stats.push({
          label: "Predicted next",
          value: `${lat.toFixed(2)}, ${lon.toFixed(2)}`,
        });
      }
    }
    const observed = Array.isArray(findings.observed_routes)
      ? findings.observed_routes.length
      : null;
    if (observed !== null) {
      stats.push({ label: "Observations", value: formatCount(observed) });
    }
    return stats;
  }

  const occurrences = asPositiveInt(report.observation_count);
  if (occurrences !== null) {
    stats.push({ label: "GBIF records", value: formatCount(occurrences) });
  }
  return stats;
}

/**
 * Reads the Biodiversity Agent's rendered map out of a chat response's context.
 *
 * Returns undefined when the agent did not run, rendered nothing, or handed
 * back a URL the browser cannot load. That last case is the important one: the
 * agent used to report a `file://` path, which a page served over http is
 * blocked from loading, so the map silently never appeared. Anything that is
 * not http(s) is dropped here rather than mounted as a permanently blank frame.
 */
export function biodiversityMapFrom(
  context: Record<string, unknown>,
): BiodiversityMapSpec | undefined {
  const url = asString(context[BIODIVERSITY_MAP_KEY]);
  if (!url || !/^https?:\/\//i.test(url)) return undefined;

  const report = isRecord(context[BIODIVERSITY_REPORT_KEY])
    ? (context[BIODIVERSITY_REPORT_KEY] as Record<string, unknown>)
    : {};
  const findings = isRecord(report.findings)
    ? (report.findings as Record<string, unknown>)
    : {};

  const sourceAgents = Array.isArray(report.source_agents)
    ? report.source_agents.filter((agent): agent is string => typeof agent === "string")
    : [];

  const skill = skillFrom(findings, report);

  return {
    url,
    skill,
    speciesName: asString(findings.species_name),
    region: asString(findings.region),
    stats: statsFrom(skill, findings, report),
    sourceAgents,
    isIllustrative: sourceAgents.some((agent) => PLACEHOLDER_AGENTS.includes(agent)),
  };
}

/**
 * The context keys the Genome Agent publishes, per its card.json. Like every
 * other agent, its `output` is merged straight into the shared context.
 */
const GENOME_VISUALIZATION_KEY = "visualization";
const GENOME_METADATA_KEY = "genome_metadata";
const GENOME_SPECIES_KEY = "species_record";

function comparisonsFrom(value: unknown): GenomeComparison[] {
  if (!Array.isArray(value)) return [];
  return value.filter(isRecord).map((row) => ({
    scientificName: asString(row.scientific_name),
    commonName: asString(row.common_name),
    genomeSizeBp: asNumber(row.genome_size_bp),
    assemblyId: asString(row.assembly_id),
  }));
}

/**
 * Reads the Genome Agent's rendered chart out of a chat response's context.
 *
 * Returns undefined unless there is actual SVG to show. The agent reports a
 * `visualization` object on several paths that carry no picture at all - a
 * `protein_structure` request becomes a NEEDS_AGENT handoff, and a species
 * with no gene table completes with `chart_data: null` - so the presence of
 * the key says nothing about whether there is anything to render.
 */
export function genomeChartFrom(
  context: Record<string, unknown>,
): GenomeChartSpec | undefined {
  const visualization = context[GENOME_VISUALIZATION_KEY];
  if (!isRecord(visualization)) return undefined;

  const svg = asString(visualization.chart_svg);
  // Guard the shape as well as the presence: this is rendered into an <img>
  // data URI, and anything that is not really an SVG document would show as a
  // broken image rather than fail loudly.
  if (!svg || !svg.trimStart().startsWith("<svg")) return undefined;

  const metadata = isRecord(context[GENOME_METADATA_KEY])
    ? (context[GENOME_METADATA_KEY] as Record<string, unknown>)
    : {};
  const species = isRecord(context[GENOME_SPECIES_KEY])
    ? (context[GENOME_SPECIES_KEY] as Record<string, unknown>)
    : {};

  return {
    svg,
    // `common_name` currently repeats the combined "Scientific (common)"
    // string, so preferring the scientific name avoids showing it twice.
    speciesName: asString(species.scientific_name) ?? asString(species.common_name),
    assemblyId: asString(context.assembly_id) ?? asString(species.assembly_id),
    genomeSizeBp: asNumber(metadata.genome_size_bp),
    chromosomeCount: asNumber(metadata.chromosome_count),
    assemblyLevel: asString(metadata.assembly_level),
    comparisons: comparisonsFrom(visualization.comparisons),
    note: asString(visualization.note),
  };
}

/**
 * The context key the Literature Agent publishes its writing under, per its
 * card.json: the agent's `output` is `{discovery, writing}` and, like every
 * agent, that is merged straight into the shared context.
 */
const WRITING_KEY = "writing";

function asStringList(value: unknown): string[] {
  if (!Array.isArray(value)) return [];
  return value.map((item) => (typeof item === "string" ? item : String(item))).filter(Boolean);
}

/**
 * Reads the Literature Agent's generated text out of a chat response's context.
 *
 * The writing sub-orchestrator returns one of three shapes, so all three are
 * handled rather than assuming the common one:
 *
 *   writing_support     -> { draft, section, references_used, ... }
 *   publication_support -> { recommended_journals, based_on_draft }
 *   both                -> { writing: {...}, publication: {...} }
 *
 * Returns undefined unless there is something to show. The key is present on
 * paths that produced nothing at all - a failed run reports `draft: null` with
 * a notice - and a panel headed "Abstract" with no abstract in it is worse
 * than no panel, since the answer text already explains the failure.
 */
export function writingDraftFrom(
  context: Record<string, unknown>,
): WritingDraftSpec | undefined {
  const root = context[WRITING_KEY];
  if (!isRecord(root)) return undefined;

  const writing = isRecord(root.writing) ? root.writing : root;
  const publication = isRecord(root.publication) ? root.publication : root;

  const draft = asString(writing.draft);
  const recommendedJournals = asString(publication.recommended_journals);
  if (!draft && !recommendedJournals) return undefined;

  return {
    // Only the writing branch names a section; a venues-only answer has no
    // section of its own, so it is labelled for what it actually is.
    section: asString(writing.section) ?? (draft ? "Draft" : "Suggested journals"),
    draft,
    recommendedJournals,
    referencesUsed: asStringList(writing.references_used),
    referencesArePlaceholder: writing.references_are_placeholder === true,
    styleCorrected: writing.style_corrected === true,
    notice: asString(writing.notice) ?? asString(root.notice),
  };
}

/**
 * The context keys the Reconstruction Agent and the Genome Agent publish, per
 * their card.json files. Both are merged into the shared context by the
 * orchestrator, which is what lets one panel show a fill *and* the flanks it
 * sits between - the reconstruction response carries gap coordinates but not
 * flanking sequence, and a fill shown without its flanks is seven letters with
 * nothing to place them against.
 */
const RECONSTRUCTION_KEY = "reconstruction";
const TARGET_GAPS_KEY = "target_gaps";

/**
 * The evidence scores, keyed exactly as `CandidateScores` serialises them.
 *
 * These are the model's field names, not descriptions of them - the agent
 * sends `homology`, not `homology_score`. Getting one wrong costs nothing
 * loudly: `asNumber` returns null for a key that is not there, the score is
 * dropped, and a panel that should show six bars shows none with no error
 * anywhere. Ordered as a reader works through the argument: what was found,
 * how well it aligned, how conserved it is, how close the donor, what the
 * model thought, and whether it survived validation.
 */
const SCORE_LABELS: Array<[string, string]> = [
  ["homology", "homology"],
  ["alignment", "alignment"],
  ["conservation", "conservation"],
  ["evolutionary", "evolutionary"],
  ["evo2", "Evo 2"],
  ["validation", "validation"],
];

function scoresFrom(value: unknown): ReconstructionScore[] {
  if (!isRecord(value)) return [];
  const out: ReconstructionScore[] = [];
  for (const [key, label] of SCORE_LABELS) {
    if (!(key in value)) continue;
    // `evo2` is null whenever Evo 2 was not consulted, which is the normal
    // case - it breaks ties rather than scoring every candidate. The domain
    // is explicit that null and 0.0 mean different things, so the null is
    // carried through and labelled rather than collapsed into a zero bar or
    // dropped as though the signal had never existed.
    out.push({ label, value: asNumber(value[key]) });
  }
  return out;
}

/**
 * Flanking sequence for each gap, keyed by start coordinate.
 *
 * The two agents number gaps independently, so `gap_id` cannot be the join
 * key. Coordinates can be, but not exactly: the Genome Agent reports 1-based
 * inclusive starts and the Reconstruction Agent's `Gap` is 0-based, so the
 * same gap can arrive one apart. Both are indexed and looked up in turn rather
 * than assuming which convention survived the handoff.
 */
function flanksByStart(value: unknown): Map<number, { left: string; right: string }> {
  const out = new Map<number, { left: string; right: string }>();
  if (!Array.isArray(value)) return out;
  for (const entry of value) {
    if (!isRecord(entry)) continue;
    const start = asNumber(entry.start);
    if (start === null) continue;
    out.set(start, {
      left: asString(entry.left_flank) ?? "",
      right: asString(entry.right_flank) ?? "",
    });
  }
  return out;
}

function gapFrom(
  raw: Record<string, unknown>,
  flanks: Map<number, { left: string; right: string }>,
): ReconstructedGap {
  const candidate = isRecord(raw.selected_candidate) ? raw.selected_candidate : null;
  const status = (asString(raw.status) ?? "unknown").toLowerCase();
  const start = asNumber(raw.start);

  // See `flanksByStart`: try the coordinate as given, then one either side.
  const flank =
    start === null
      ? undefined
      : (flanks.get(start) ?? flanks.get(start + 1) ?? flanks.get(start - 1));

  return {
    gapId: asString(raw.gap_id) ?? "gap",
    start,
    end: asNumber(raw.end),
    lengthBp: asNumber(raw.length),
    status,
    resolved: status === "resolved" && candidate !== null,
    unresolvedReason: asString(raw.unresolved_reason),
    explanation: asString(raw.explanation),
    fill: candidate ? asString(candidate.sequence) : null,
    fillLengthBp: candidate ? asNumber(candidate.length) : null,
    confidence: candidate ? asNumber(candidate.confidence) : null,
    confidenceLevel: candidate ? asString(candidate.confidence_level) : null,
    isModelGenerated: candidate?.is_model_generated === true,
    supportingOrganisms: candidate ? asStringList(candidate.supporting_organisms) : [],
    supportingHits: candidate ? asStringList(candidate.supporting_hits) : [],
    scores: candidate ? scoresFrom(candidate.scores) : [],
    leftFlank: flank?.left ?? null,
    rightFlank: flank?.right ?? null,
  };
}

function selectionFrom(context: Record<string, unknown>): ReconstructionSelection | null {
  const policy = isRecord(context.selection_policy) ? context.selection_policy : {};
  const selection: ReconstructionSelection = {
    gapsFound: asNumber(context.gaps_found),
    gapsOverFloor: asNumber(context.gaps_over_floor),
    gapsSelected: asNumber(context.gaps_selected),
    recordsInAssembly: asNumber(policy.records_in_assembly),
    recordsOverCeiling: asNumber(policy.records_over_size_ceiling),
    minGapBp: asNumber(policy.min_gap_bp),
    assemblyGapBasesBp: asNumber(context.assembly_gap_bases_bp),
    assemblyGapFraction: asNumber(context.assembly_gap_fraction),
  };
  // Older Genome Agent builds send none of this. An all-empty object would
  // render as a row of dashes claiming to explain the selection.
  const hasAny = Object.values(selection).some((v) => v !== null);
  return hasAny ? selection : null;
}

/**
 * Reads the Reconstruction Agent's result out of a chat response's context.
 *
 * Returns undefined unless there are gaps to show. The key is present on paths
 * that produced nothing - a run that found no unresolved regions completes with
 * an empty `reconstructions` list - and a panel headed "Reconstruction" with no
 * gaps in it says less than the written answer already did.
 */
export function reconstructionFrom(
  context: Record<string, unknown>,
): ReconstructionSpec | undefined {
  const root = context[RECONSTRUCTION_KEY];
  if (!isRecord(root)) return undefined;

  const rawGaps = Array.isArray(root.reconstructions) ? root.reconstructions : [];
  if (rawGaps.length === 0) return undefined;

  const flanks = flanksByStart(context[TARGET_GAPS_KEY]);
  const gaps = rawGaps
    .filter(isRecord)
    .map((raw) => gapFrom(raw as Record<string, unknown>, flanks));

  const summary = isRecord(root.summary) ? root.summary : {};

  return {
    status: (asString(root.status) ?? "unknown").toLowerCase(),
    summary: asString(context.reconstruction_summary),
    scientificName: asString(root.scientific_name),
    assemblyId: asString(root.assembly_id),
    sequenceAccession: asString(root.sequence_accession),
    requestedGaps: asNumber(summary.requested_gaps) ?? gaps.length,
    resolvedGaps: asNumber(summary.resolved_gaps) ?? gaps.filter((g) => g.resolved).length,
    unresolvedGaps:
      asNumber(summary.unresolved_gaps) ?? gaps.filter((g) => !g.resolved).length,
    // Resolved first: one filled gap among nine skipped is the finding, and it
    // should not be buried under the nine.
    gaps: [...gaps].sort((a, b) => Number(b.resolved) - Number(a.resolved)),
    selection: selectionFrom(context),
    warnings: asStringList(root.warnings),
  };
}

/**
 * Reads the Molecular Comparison sub-agent's graph out of `similarity_network`.
 *
 * The agent builds this with NetworkX and ships `node_link_data`, so the shape
 * is `{nodes: [{id}], edges: [{source, target, score}]}`. Older NetworkX
 * spells the edge list `links`, and the agent's own public output contract
 * documents the field as a JSON *string*, so both are accepted rather than
 * silently yielding an empty graph.
 *
 * Returns null when there is nothing usable; the panel then falls back to
 * building a graph out of the flat `similarity_scores` list.
 */
function similarityNetworkFrom(raw: unknown): SimilarityNetwork | null {
  let value = raw;

  if (typeof value === "string") {
    try {
      value = JSON.parse(value);
    } catch {
      return null;
    }
  }

  if (!isRecord(value)) return null;

  const rawNodes = Array.isArray(value.nodes) ? value.nodes : [];
  const species = rawNodes
    .map((node) => (isRecord(node) ? asString(node.id) : asString(node)))
    .filter((name): name is string => Boolean(name));

  const rawEdges = Array.isArray(value.edges)
    ? value.edges
    : Array.isArray(value.links)
      ? value.links
      : [];

  const edges: SimilarityScore[] = rawEdges
    .filter(isRecord)
    .map((edge) => ({
      speciesA: asString(edge.source) ?? "",
      speciesB: asString(edge.target) ?? "",
      score: asNumber(edge.score) ?? 0,
    }))
    .filter((edge) => edge.speciesA && edge.speciesB);

  if (species.length === 0 && edges.length === 0) return null;

  return { species, edges };
}

/**
 * Reads the Evolution Agent's result out of a chat response's context.
 *
 * The Evolution Agent's output dict is merged flat into the shared context by
 * `worker_node`, so these are top-level keys - there is no wrapper object the
 * way `reconstruction` has one.
 *
 * Returns undefined unless there is a tree or at least one similarity score.
 * The clarification and failure paths set neither, and a panel headed
 * "Evolutionary analysis" containing only a species list says less than the
 * written answer already did.
 */
export function evolutionFrom(context: Record<string, unknown>): EvolutionSpec | undefined {
  const newick = asString(context.newick_tree);
  const rawScores = Array.isArray(context.similarity_scores) ? context.similarity_scores : [];

  if (!newick && rawScores.length === 0) return undefined;

  const similarityScores: SimilarityScore[] = rawScores
    .filter(isRecord)
    .map((raw) => ({
      speciesA: asString(raw.species_a) ?? "",
      speciesB: asString(raw.species_b) ?? "",
      score: asNumber(raw.score) ?? 0,
    }))
    .filter((edge) => edge.speciesA && edge.speciesB);

  const rawGroups = Array.isArray(context.species_groups) ? context.species_groups : [];
  const speciesGroups: SpeciesGroup[] = rawGroups
    .filter(isRecord)
    .map((raw, index) => ({
      groupId: asNumber(raw.group_id) ?? index,
      species: asStringList(raw.species),
      meanScore: asNumber(raw.mean_score),
    }))
    .filter((group) => group.species.length > 0);

  return {
    speciesList: asStringList(context.species_list),
    newick,
    model: asString(context.model),
    // Null is meaningful and must survive: it is what the agent reports when
    // UFBoot did not run, and rendering it as 0 would read as "no support"
    // rather than "not measured".
    overallConfidence: asNumber(context.overall_confidence),
    similarityScores,
    similarityNetwork: similarityNetworkFrom(context.similarity_network),
    speciesGroups,
    interpretation: asString(context.interpretation),
    warnings: asStringList(context.warnings),
  };
}
