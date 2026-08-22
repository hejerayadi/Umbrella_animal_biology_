import type {
  AgentActivity,
  AgentStatus,
  BiodiversityMapSpec,
  BiodiversityMapStat,
  BiodiversitySkill,
  ProteinDomain,
  ProteinSelection,
  ProteinViewerSpec,
  RecognitionCandidate,
  RecognitionProvenance,
  RecognitionResult,
} from "./umbrella-types";
import { apiRequest, apiUrl } from "./api-client";

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
  return `${ORCHESTRATOR_API_URL}${path.startsWith("/") ? "" : "/"}${path}`;
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
