import type { AgentActivity, AgentStatus } from "./umbrella-types";

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
}

/** Sends one chat message to the orchestrator and returns its final result. */
export async function askOrchestrator(query: string): Promise<ChatResponse> {
  const response = await fetch(`${ORCHESTRATOR_API_URL}/api/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query }),
  });

  if (!response.ok) {
    throw new Error(`Orchestrator backend returned ${response.status}`);
  }

  return (await response.json()) as ChatResponse;
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
