import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react";

import { MOCK_AGENT_ACTIVITY, MOCK_CONVERSATIONS, MOCK_MESSAGES } from "./mock-data";
import {
  generatedImageFrom,
  imageUrlFor,
  parseExecutionHistory,
  streamOrchestrator,
  biodiversityMapFrom,
  evolutionFrom,
  genomeChartFrom,
  proteinViewerFrom,
  reconstructionFrom,
  recognitionFrom,
  writingDraftFrom,
  type ChatResponse,
  type OrchestratorEvent,
  type UploadedImage,
} from "./orchestrator-client";
import type { AgentActivity, AgentStatus, Conversation, Message } from "./umbrella-types";

const STORAGE_KEY = "umbrella.mock.state.v1";

interface PersistedState {
  conversations: Conversation[];
  messages: Message[];
  activities: AgentActivity[];
}

const initialState: PersistedState = {
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

/**
 * Everything the agents produced, pulled out of one finished response.
 *
 * Kept on the assistant message rather than in a side channel, so the panels
 * are still there after a reload exactly like the text they belong to.
 */
function panelsFrom(response: ChatResponse): Partial<Message> {
  const viewer = proteinViewerFrom(response.context);
  const generated = generatedImageFrom(response);
  const recognition = recognitionFrom(response.context);
  const map = biodiversityMapFrom(response.context);
  const chart = genomeChartFrom(response.context);
  const writing = writingDraftFrom(response.context);
  const reconstruction = reconstructionFrom(response.context);
  const evolution = evolutionFrom(response.context);

  return {
    ...(viewer ? { proteinViewer: viewer } : {}),
    ...(generated ? { generatedImageUrl: generated } : {}),
    ...(recognition ? { recognition } : {}),
    ...(map ? { biodiversityMap: map } : {}),
    ...(chart ? { genomeChart: chart } : {}),
    ...(writing ? { writingDraft: writing } : {}),
    ...(reconstruction ? { reconstruction } : {}),
    ...(evolution ? { evolution } : {}),
  };
}

const STEP_STATUS: Record<"running" | "done" | "failed", AgentStatus> = {
  running: "running",
  done: "complete",
  failed: "failed",
};

/**
 * Applies one live step event to the timeline.
 *
 * The backend sends the same `id` twice - once when a step starts and once
 * with its outcome - so a step already on the list is updated in place. That
 * is what turns the panel into a running commentary instead of a list that
 * doubles in length.
 */
function applyStep(
  activities: AgentActivity[],
  conversationId: string,
  event: Extract<OrchestratorEvent, { type: "step" }>,
): AgentActivity[] {
  const id = `${conversationId}:${event.id}`;
  const existing = activities.find((activity) => activity.id === id);

  if (!existing) {
    return [
      ...activities,
      {
        id,
        conversationId,
        agentName: event.agent,
        status: STEP_STATUS[event.state],
        description: event.text,
        timestamp: new Date().toISOString(),
      },
    ];
  }

  return activities.map((activity) =>
    activity.id === id
      ? { ...activity, status: STEP_STATUS[event.state], description: event.text }
      : activity,
  );
}

interface UmbrellaContextValue extends PersistedState {
  hydrated: boolean;
  isThinking: boolean;
  streamingMessageId: string | null;
  createConversation: (title?: string) => Conversation;
  renameConversation: (id: string, title: string) => void;
  deleteConversation: (id: string) => void;
  sendMessage: (conversationId: string, content: string, image?: UploadedImage | null) => void;
  messagesFor: (conversationId: string) => Message[];
  activitiesFor: (conversationId: string) => AgentActivity[];
  /**
   * The answer currently being written, or null when nothing is streaming.
   *
   * Deliberately not part of the persisted message list: it changes on every
   * token, and `state` is written to localStorage on every change.
   */
  liveAnswerFor: (conversationId: string) => string | null;
}

const UmbrellaContext = createContext<UmbrellaContextValue | null>(null);

export function UmbrellaProvider({ children }: { children: ReactNode }) {
  const [state, setState] = useState<PersistedState>(initialState);
  const [hydrated, setHydrated] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const [streamingMessageId, setStreamingMessageId] = useState<string | null>(null);
  // The run in progress. Held outside `state` so that a token arriving every
  // few milliseconds does not rewrite localStorage every few milliseconds.
  const [liveConversationId, setLiveConversationId] = useState<string | null>(null);
  const [liveActivities, setLiveActivities] = useState<AgentActivity[]>([]);
  const [liveAnswer, setLiveAnswer] = useState<string | null>(null);

  // The same list as `liveActivities`, readable synchronously. Events arrive
  // from a promise callback that closes over the first render's state, and the
  // committing step at the end needs the final list, not that stale one.
  const liveActivitiesRef = useRef<AgentActivity[]>([]);
  const updateLiveActivities = useCallback(
    (update: (previous: AgentActivity[]) => AgentActivity[]) => {
      liveActivitiesRef.current = update(liveActivitiesRef.current);
      setLiveActivities(liveActivitiesRef.current);
    },
    [],
  );

  useEffect(() => {
    setState(readPersisted());
    setHydrated(true);
  }, []);

  useEffect(() => {
    if (!hydrated) return;
    window.localStorage.setItem(STORAGE_KEY, JSON.stringify(state));
  }, [state, hydrated]);

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
      setStreamingMessageId(null);
      setLiveConversationId(conversationId);
      updateLiveActivities(() => []);
      setLiveAnswer(null);

      // Accumulated here rather than read back off React state: `setState` is
      // asynchronous, and the next token would otherwise append to whatever
      // the last render happened to see.
      let answer = "";
      // True once the orchestrator has streamed at least one token. It decides
      // whether the finished message needs the typewriter effect - text the
      // user already watched arrive must not be typed out a second time.
      let streamed = false;

      const onEvent = (event: OrchestratorEvent) => {
        switch (event.type) {
          case "step":
            updateLiveActivities((prev) => applyStep(prev, conversationId, event));
            break;
          case "thought":
            updateLiveActivities((prev) =>
              prev.map((activity) =>
                activity.id === `${conversationId}:${event.id}`
                  ? { ...activity, thought: event.text }
                  : activity,
              ),
            );
            break;
          case "answer":
            answer += event.delta;
            streamed = true;
            // The steps have had their turn; from the first token of the
            // answer the panel is history and the message is the live thing.
            setIsThinking(false);
            setLiveAnswer(answer);
            break;
          default:
            break;
        }
      };

      streamOrchestrator(content, image, onEvent)
        .then((response) => {
          const assistantId = uid("msg");
          // The live steps when the run streamed; the after-the-fact history
          // when it fell back to the blocking route, which reports what
          // happened only once it is all over.
          const activities = streamed
            ? liveActivitiesRef.current
            : parseExecutionHistory(response.execution_history, conversationId);

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
                ...panelsFrom(response),
              },
            ],
          }));
          setIsThinking(false);
          setLiveConversationId(null);
          setLiveAnswer(null);
          if (!streamed) setStreamingMessageId(assistantId);
        })
        .catch((error: unknown) => {
          const message = error instanceof Error ? error.message : String(error);

          setState((prev) => ({
            ...prev,
            messages: [
              ...prev.messages,
              {
                id: uid("msg"),
                conversationId,
                sender: "assistant",
                content: `⚠️ Could not reach the orchestrator backend: ${message}`,
                timestamp: new Date().toISOString(),
              },
            ],
          }));
          setIsThinking(false);
          setLiveConversationId(null);
          setLiveAnswer(null);
        });
    },
    [updateLiveActivities],
  );

  const messagesFor = useCallback(
    (conversationId: string) => state.messages.filter((m) => m.conversationId === conversationId),
    [state.messages],
  );

  const activitiesFor = useCallback(
    (conversationId: string) =>
      // While a run is in flight its live steps ARE the timeline; the stored
      // ones belong to the previous message and are replaced at the end.
      conversationId === liveConversationId
        ? liveActivities
        : state.activities.filter((a) => a.conversationId === conversationId),
    [state.activities, liveConversationId, liveActivities],
  );

  const liveAnswerFor = useCallback(
    (conversationId: string) => (conversationId === liveConversationId ? liveAnswer : null),
    [liveConversationId, liveAnswer],
  );

  const value = useMemo<UmbrellaContextValue>(
    () => ({
      ...state,
      hydrated,
      isThinking,
      streamingMessageId,
      createConversation,
      renameConversation,
      deleteConversation,
      sendMessage,
      messagesFor,
      activitiesFor,
      liveAnswerFor,
    }),
    [
      state,
      hydrated,
      isThinking,
      streamingMessageId,
      createConversation,
      renameConversation,
      deleteConversation,
      sendMessage,
      messagesFor,
      activitiesFor,
      liveAnswerFor,
    ],
  );

  return <UmbrellaContext.Provider value={value}>{children}</UmbrellaContext.Provider>;
}

export function useUmbrella() {
  const ctx = useContext(UmbrellaContext);
  if (!ctx) throw new Error("useUmbrella must be used inside UmbrellaProvider");
  return ctx;
}
