import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
} from "react";

import { MOCK_AGENT_ACTIVITY, MOCK_CONVERSATIONS, MOCK_MESSAGES } from "./mock-data";
import {
  askOrchestrator,
  imageUrlFor,
  parseExecutionHistory,
  type UploadedImage,
} from "./orchestrator-client";
import type { AgentActivity, Conversation, Message, User } from "./umbrella-types";

const STORAGE_KEY = "umbrella.mock.state.v1";

interface PersistedState {
  user: User | null;
  conversations: Conversation[];
  messages: Message[];
  activities: AgentActivity[];
}

const initialState: PersistedState = {
  user: null,
  conversations: MOCK_CONVERSATIONS,
  messages: MOCK_MESSAGES,
  activities: MOCK_AGENT_ACTIVITY,
};

function readPersisted(): PersistedState {
  if (typeof window === "undefined") return initialState;
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return initialState;
    const parsed = JSON.parse(raw) as Partial<PersistedState>;
    return {
      user: parsed.user ?? null,
      conversations: parsed.conversations ?? MOCK_CONVERSATIONS,
      messages: parsed.messages ?? MOCK_MESSAGES,
      activities: parsed.activities ?? MOCK_AGENT_ACTIVITY,
    };
  } catch {
    return initialState;
  }
}

const uid = (prefix: string) =>
  `${prefix}_${Math.random().toString(36).slice(2, 8)}${Date.now().toString(36).slice(-4)}`;

interface UmbrellaContextValue extends PersistedState {
  hydrated: boolean;
  isThinking: boolean;
  streamingMessageId: string | null;
  signIn: (email: string) => void;
  signUp: (payload: Omit<User, "id" | "createdAt">) => void;
  signOut: () => void;
  createConversation: (title?: string) => Conversation;
  renameConversation: (id: string, title: string) => void;
  deleteConversation: (id: string) => void;
  sendMessage: (conversationId: string, content: string, image?: UploadedImage | null) => void;
  messagesFor: (conversationId: string) => Message[];
  activitiesFor: (conversationId: string) => AgentActivity[];
}

const UmbrellaContext = createContext<UmbrellaContextValue | null>(null);

export function UmbrellaProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<PersistedState>(initialState);
  const [hydrated, setHydrated] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);

  useEffect(() => {
    setState(readPersisted());
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }, [state, hydrated]);

  const signIn = useCallback((email: string) => {
    setState((prev) => ({
      ...prev,
      user:
        prev.user ??
        ({
          id: uid("usr"),
          name: email.split("@")[0] || "Researcher",
          email,
          role: "Researcher",
          purpose: "",
          mainInterest: "",
          goals: "",
          researchInterests: [],
          createdAt: new Date().toISOString(),
        } satisfies User),
    }));
  }, []);

  const signUp = useCallback((payload: Omit<User, "id" | "createdAt">) => {
    setState((prev) => ({
      ...prev,
      user: { ...payload, id: uid("usr"), createdAt: new Date().toISOString() },
    }));
  }, []);

  const signOut = useCallback(() => {
    setState((prev) => ({ ...prev, user: null }));
  }, []);

  const createConversation = useCallback((title = "New conversation") => {
    const now = new Date().toISOString();
    const conversation: Conversation = { id: uid("cnv"), title, createdAt: now, updatedAt: now };
    setState((prev) => ({ ...prev, conversations: [conversation, ...prev.conversations] }));
    return conversation;
  }, []);

  const renameConversation = useCallback((id: string, title: string) => {
    setState((prev) => ({
      ...prev,
      conversations: prev.conversations.map((c) =>
        c.id === id
          ? { ...c, title: title.trim() || c.title, updatedAt: new Date().toISOString() }
          : c,
      ),
    }));
  }, []);

  const deleteConversation = useCallback((id: string) => {
    setState((prev) => ({
      ...prev,
      conversations: prev.conversations.filter((c) => c.id !== id),
      messages: prev.messages.filter((m) => m.conversationId !== id),
      activities: prev.activities.filter((a) => a.conversationId !== id),
    }));
  }, []);

  const sendMessage = useCallback(
    (conversationId: string, content: string, image?: UploadedImage | null) => {
      const now = new Date().toISOString();
      const userMessage: Message = {
        id: uid("msg"),
        conversationId,
        sender: "user",
        content,
        timestamp: now,
        // Served back from the backend rather than kept as the composer's
        // blob: URL, which is revoked as soon as the attachment clears.
        ...(image ? { imageUrl: imageUrlFor(image.image_id), imageName: image.filename } : {}),
      };

      setState((prev) => ({
        ...prev,
        messages: [...prev.messages, userMessage],
        activities: prev.activities.filter((a) => a.conversationId !== conversationId),
        conversations: prev.conversations.map((c) =>
          c.id === conversationId
            ? {
                ...c,
                updatedAt: now,
                title:
                  c.title === "New conversation"
                    ? content.slice(0, 52) + (content.length > 52 ? "…" : "")
                    : c.title,
              }
            : c,
        ),
      }));

      setIsThinking(true);

      askOrchestrator(content, image)
        .then((response) => {
          const activities = parseExecutionHistory(response.execution_history, conversationId);
          const assistantId = uid("msg");

          setState((prev) => ({
            ...prev,
            activities: [
              ...prev.activities.filter((a) => a.conversationId !== conversationId),
              ...activities,
            ],
            messages: [
              ...prev.messages,
              {
                id: assistantId,
                conversationId,
                sender: "assistant",
                content: response.answer,
                timestamp: new Date().toISOString(),
              },
            ],
          }));
          setIsThinking(false);
          setStreamingMessageId(assistantId);
        })
        .catch((error: unknown) => {
          const assistantId = uid("msg");
          const message = error instanceof Error ? error.message : String(error);

          setState((prev) => ({
            ...prev,
            messages: [
              ...prev.messages,
              {
                id: assistantId,
                conversationId,
                sender: "assistant",
                content: `⚠️ Could not reach the orchestrator backend: ${message}`,
                timestamp: new Date().toISOString(),
              },
            ],
          }));
          setIsThinking(false);
        });
    },
    [],
  );

  const messagesFor = useCallback(
    (conversationId: string) => state.messages.filter((m) => m.conversationId === conversationId),
    [state.messages],
  );

  const activitiesFor = useCallback(
    (conversationId: string) => state.activities.filter((a) => a.conversationId === conversationId),
    [state.activities],
  );

  const value = useMemo<UmbrellaContextValue>(
    () => ({
      ...state,
      hydrated,
      isThinking,
      streamingMessageId,
      signIn,
      signUp,
      signOut,
      createConversation,
      renameConversation,
      deleteConversation,
      sendMessage,
      messagesFor,
      activitiesFor,
    }),
    [
      state,
      hydrated,
      isThinking,
      streamingMessageId,
      signIn,
      signUp,
      signOut,
      createConversation,
      renameConversation,
      deleteConversation,
      sendMessage,
      messagesFor,
      activitiesFor,
    ],
  );

  return <UmbrellaContext.Provider value={value}>{children}</UmbrellaContext.Provider>;
}

export function useUmbrella() {
  const ctx = useContext(UmbrellaContext);
  if (!ctx) throw new Error("useUmbrella must be used inside UmbrellaProvider");
  return ctx;
}
