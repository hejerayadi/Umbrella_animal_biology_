const DEFAULT_API_BASE =
  typeof window === "undefined"
    ? "http://localhost:8000"
    : `${window.location.protocol}//${window.location.hostname}:8000`;

const API_BASE = import.meta.env.VITE_ORCHESTRATOR_API_URL ?? DEFAULT_API_BASE;

export interface ApiMeta {
  api_version: string;
  timestamp: string;
  request_id?: string | null;
  duration_ms?: number | null;
  pagination?: { page: number; page_size: number; total: number } | null;
}

interface ApiEnvelope<T> {
  data: T | null;
  meta: ApiMeta;
  error: {
    code: string;
    title: string;
    status: number;
    detail: string;
    errors: Array<Record<string, unknown>>;
  } | null;
}

export interface ApiResult<T> {
  data: T;
  meta: ApiMeta;
}

export class ApiClientError extends Error {
  constructor(
    message: string,
    public readonly status: number,
    public readonly code: string,
  ) {
    super(message);
  }
}

let csrfToken: string | null = null;

export function setCsrfToken(token: string | null) {
  csrfToken = token;
}

export function apiUrl(path: string): string {
  return `${API_BASE}${path.startsWith("/") ? "" : "/"}${path}`;
}

export async function apiRequest<T>(path: string, init: RequestInit = {}): Promise<T> {
  return (await apiRequestWithMeta<T>(path, init)).data;
}

/**
 * Same auth, CSRF and cookie handling as `apiRequest`, but hands back the raw
 * `Response` with its body still unread.
 *
 * `apiRequest` awaits `response.json()`, which for a streaming endpoint means
 * waiting for the last byte - exactly what a stream exists to avoid. Callers
 * read `response.body` themselves.
 *
 * An error response is still JSON, and still in the standard envelope, so it
 * is unwrapped here and thrown the same way `apiRequest` would.
 */
export async function apiStream(path: string, init: RequestInit = {}): Promise<Response> {
  const method = (init.method ?? "GET").toUpperCase();
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (!["GET", "HEAD", "OPTIONS"].includes(method) && csrfToken) {
    headers.set("X-CSRF-Token", csrfToken);
  }
  const response = await fetch(apiUrl(path), { ...init, headers, credentials: "include" });
  if (!response.ok) {
    const envelope = (await response.json().catch(() => null)) as ApiEnvelope<never> | null;
    throw new ApiClientError(
      envelope?.error?.detail ?? `Request failed (${response.status})`,
      response.status,
      envelope?.error?.code ?? "HTTP_ERROR",
    );
  }
  return response;
}

export async function apiRequestWithMeta<T>(
  path: string,
  init: RequestInit = {},
): Promise<ApiResult<T>> {
  const method = (init.method ?? "GET").toUpperCase();
  const headers = new Headers(init.headers);
  if (init.body && !(init.body instanceof FormData) && !headers.has("Content-Type")) {
    headers.set("Content-Type", "application/json");
  }
  if (!["GET", "HEAD", "OPTIONS"].includes(method) && csrfToken) {
    headers.set("X-CSRF-Token", csrfToken);
  }
  const response = await fetch(apiUrl(path), { ...init, headers, credentials: "include" });
  const envelope = (await response.json().catch(() => null)) as ApiEnvelope<T> | null;
  if (!response.ok || !envelope || envelope.error) {
    throw new ApiClientError(
      envelope?.error?.detail ?? `Request failed (${response.status})`,
      response.status,
      envelope?.error?.code ?? "HTTP_ERROR",
    );
  }
  return { data: envelope.data as T, meta: envelope.meta };
}
