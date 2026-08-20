import type { AgentActivity, Conversation, Message, User } from "./umbrella-types";

export const MOCK_USER: User = {
  id: "usr_demo_01",
  name: "Dr. Amara Sørensen",
  email: "amara@umbrella.research",
  role: "Researcher",
  purpose: "Accelerate comparative genomics across endangered vertebrates",
  mainInterest: "Genome reconstruction and trait evolution",
  goals: "Publish a cross-species trait atlas within the year",
  researchInterests: ["Genome Reconstruction", "Trait Discovery", "Scientific Literature"],
  createdAt: "2026-03-11T09:12:00.000Z",
};

export const MOCK_CONVERSATIONS: Conversation[] = [
  {
    id: "cnv_001",
    title: "Draft genome scaffolding for Pantherinae",
    createdAt: "2026-07-24T08:20:00.000Z",
    updatedAt: "2026-07-29T16:40:00.000Z",
  },
  {
    id: "cnv_002",
    title: "Biodiversity decline signals in coastal reefs",
    createdAt: "2026-07-19T11:00:00.000Z",
    updatedAt: "2026-07-27T09:05:00.000Z",
  },
  {
    id: "cnv_003",
    title: "Protein fold comparison: myoglobin variants",
    createdAt: "2026-07-12T14:45:00.000Z",
    updatedAt: "2026-07-21T10:12:00.000Z",
  },
];

export const MOCK_MESSAGES: Message[] = [
  {
    id: "msg_001",
    conversationId: "cnv_001",
    sender: "user",
    content:
      "I have short-read data for three Pantherinae species. How should I approach reference-guided scaffolding, and what does the recent literature say about assembly bias?",
    timestamp: "2026-07-29T16:31:00.000Z",
  },
  {
    id: "msg_002",
    conversationId: "cnv_001",
    sender: "assistant",
    content: `## Recommended scaffolding strategy

Three agents contributed to this answer: **Genome Agent**, **Literature Agent**, and **Orchestrator**.

### 1. Assembly plan

| Stage | Tool class | Notes |
| --- | --- | --- |
| QC | read trimming | Drop \`Q < 20\` tails before assembly |
| Contigging | de Bruijn | k-mer sweep 31–71 |
| Scaffolding | reference-guided | Use *Felis catus* as anchor |

### 2. Bias to control for

- Reference bias inflates synteny agreement in conserved blocks.
- Repeat collapse is the dominant source of missing trait loci.

\`\`\`bash
# quick contiguity check
assembly-stats scaffolds.fa | grep -E "N50|number"
\`\`\`

> Literature Agent: 4 of 6 recent Pantherinae assemblies report inflated N50 when scaffolding against a single anchor genome.`,
    timestamp: "2026-07-29T16:33:00.000Z",
  },
  {
    id: "msg_003",
    conversationId: "cnv_002",
    sender: "user",
    content: "Summarise observed biodiversity shifts in warm-water reef systems over the last decade.",
    timestamp: "2026-07-27T09:01:00.000Z",
  },
  {
    id: "msg_004",
    conversationId: "cnv_002",
    sender: "assistant",
    content: `**Biodiversity Agent** compiled the following synthesis.

- Coral cover contraction is strongest between 0–12 m depth.
- Cryptobenthic fish richness declines faster than large-bodied richness.
- Recovery windows shorten when thermal anomalies occur < 4 years apart.

Next suggested step: run a species identification pass on your transect imagery.`,
    timestamp: "2026-07-27T09:05:00.000Z",
  },
  {
    id: "msg_005",
    conversationId: "cnv_003",
    sender: "user",
    content: "Compare the heme pocket geometry between terrestrial and diving mammal myoglobins.",
    timestamp: "2026-07-21T10:09:00.000Z",
  },
  {
    id: "msg_006",
    conversationId: "cnv_003",
    sender: "assistant",
    content: `**Protein Agent** returned a structural comparison.

1. Diving-mammal myoglobins show elevated net surface charge, reducing aggregation at high concentration.
2. Heme pocket volume is largely conserved; the functional divergence is electrostatic, not steric.
3. Distal histidine positioning varies by < 0.4 Å across the compared set.`,
    timestamp: "2026-07-21T10:12:00.000Z",
  },
];

export const MOCK_AGENT_ACTIVITY: AgentActivity[] = [
  {
    id: "act_001",
    conversationId: "cnv_001",
    agentName: "Orchestrator",
    status: "complete",
    description: "Planned a 4-step workflow across genome and literature agents",
    timestamp: "2026-07-29T16:31:10.000Z",
  },
  {
    id: "act_002",
    conversationId: "cnv_001",
    agentName: "Genome Agent",
    status: "complete",
    description: "Evaluated scaffolding strategies for short-read input",
    timestamp: "2026-07-29T16:31:40.000Z",
  },
  {
    id: "act_003",
    conversationId: "cnv_001",
    agentName: "Literature Agent",
    status: "complete",
    description: "Screened 18 recent assembly papers for bias reports",
    timestamp: "2026-07-29T16:32:20.000Z",
  },
  {
    id: "act_004",
    conversationId: "cnv_001",
    agentName: "Orchestrator",
    status: "complete",
    description: "Merged agent outputs and finalised the answer",
    timestamp: "2026-07-29T16:33:00.000Z",
  },
];

/** Simulated orchestration timeline used when a new message is sent. */
export const MOCK_ORCHESTRATION_PLAN = [
  { agentName: "Orchestrator", description: "Planning workflow..." },
  { agentName: "Genome Agent", description: "Selecting Genome Agent..." },
  { agentName: "Literature Agent", description: "Requesting Literature Agent..." },
  { agentName: "Orchestrator", description: "Combining responses..." },
  { agentName: "Orchestrator", description: "Finalizing answer..." },
] as const;

export const MOCK_ASSISTANT_REPLY = `### Synthesised answer

Umbrella coordinated **3 agents** to build this response. This is mock output — the orchestration layer will stream real events once a backend is connected.

- **Genome Agent** — inspected the relevant sequence context.
- **Literature Agent** — retrieved supporting publications.
- **Orchestrator** — reconciled the findings into a single narrative.

\`\`\`python
# illustrative snippet
from umbrella import Orchestrator

plan = Orchestrator().plan("compare trait loci across species")
for step in plan:
    print(step.agent, step.status)
\`\`\`

> Ask a follow-up to route the request to a different specialist agent.`;

export const SUGGESTED_PROMPTS = [
  "Reconstruct a draft genome from my short-read dataset",
  "Which traits diverge most across these three species?",
  "Summarise recent literature on reef biodiversity loss",
  "Visualise the structural difference between two protein variants",
];