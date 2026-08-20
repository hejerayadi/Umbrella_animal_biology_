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

export type MessageSender = "user" | "assistant";

export interface Message {
  id: string;
  conversationId: string;
  sender: MessageSender;
  content: string;
  timestamp: string;
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