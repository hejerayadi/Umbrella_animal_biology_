import type {
  AgentActivity,
  AgentStatus,
  ProteinDomain,
  ProteinSelection,
  ProteinViewerSpec,
} from "./umbrella-types";

/**
 * Base URL of the Python orchestrator API (backend/api.py).
 * Override with VITE_ORCHESTRATOR_API_URL in a .env file if it's not running
 * on the default local port.
 */
const ORCHESTRATOR_API_URL = import.meta.env.VITE_ORCHESTRATOR_API_URL ?? "http://localhost:8000";

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
  return `${ORCHESTRATOR_API_URL}/api/upload/${imageId}`;
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

  const response = await fetch(`${ORCHESTRATOR_API_URL}/api/upload`, {
    method: "POST",
    body, // no Content-Type header: the browser sets the multipart boundary
  });

  if (!response.ok) {
    // The backend's 400s carry a message written to be read by a user
    // ("That file is not a JPEG, PNG or WebP image"), so surface it rather
    // than replacing it with a status code.
    const detail = await response
      .json()
      .then((payload: { detail?: string }) => payload.detail)
      .catch(() => undefined);
    throw new Error(detail ?? `Upload failed (${response.status})`);
  }

  return (await response.json()) as UploadedImage;
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

  const response = await fetch(`${ORCHESTRATOR_API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, context }),
  });

  if (!response.ok) {
    throw new Error(`Orchestrator backend returned ${response.status}`);
  }

  return (await response.json()) as ChatResponse;
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
