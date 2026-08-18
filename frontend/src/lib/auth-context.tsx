import { createContext, useCallback, useContext, useEffect, useMemo, useState, type ReactNode } from "react";

import { ApiClientError, apiRequest, setCsrfToken } from "./api-client";
import type { User } from "./umbrella-types";

export type AuthStep =
  | "AUTHENTICATED"
  | "MFA_REQUIRED"
  | "MFA_ENROLLMENT_REQUIRED"
  | "PASSWORD_CHANGE_REQUIRED";

/**
 * Single source of truth for where a given auth step should live. Every page
 * that gates itself on `nextStep` (sign-in, MFA, change-password) must route
 * through this instead of hand-rolling its own "redirect if not X" check —
 * otherwise a page's own post-action navigate() races with its effect the
 * moment `nextStep` changes (e.g. MFA verify succeeds -> nextStep flips to
 * AUTHENTICATED -> the MFA page's old guard treated that as "not MFA_REQUIRED"
 * and bounced back to /signin, undoing the navigate to /chat).
 */
export function resolveAuthDestination(step: AuthStep | null) {
  if (step === "AUTHENTICATED") return "/chat" as const;
  if (step === "PASSWORD_CHANGE_REQUIRED") return "/change-password" as const;
  if (step === "MFA_REQUIRED" || step === "MFA_ENROLLMENT_REQUIRED") return "/mfa" as const;
  return "/signin" as const;
}

interface AuthResult {
  next_step: AuthStep;
  csrf_token: string;
  user?: User | null;
  recovery_codes?: string[];
}

export interface RegistrationPayload {
  email: string;
  password: string;
  full_name: string;
  institution: string;
  professional_title: string;
  country: string;
  orcid?: string;
  motivation: string;
  specialties: string[];
}

interface AuthContextValue {
  user: User | null;
  loading: boolean;
  nextStep: AuthStep | null;
  login: (email: string, password: string) => Promise<AuthStep>;
  register: (payload: RegistrationPayload) => Promise<void>;
  logout: () => Promise<void>;
  changePassword: (newPassword: string, currentPassword?: string) => Promise<AuthStep>;
  beginMfaEnrollment: () => Promise<{ secret: string; provisioning_uri: string }>;
  confirmMfaEnrollment: (code: string) => Promise<string[]>;
  verifyMfa: (code: string, method?: "totp" | "recovery_code") => Promise<void>;
  refreshSession: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [nextStep, setNextStep] = useState<AuthStep | null>(null);

  const applyResult = useCallback((result: AuthResult) => {
    setCsrfToken(result.csrf_token);
    setNextStep(result.next_step);
    setUser(result.next_step === "AUTHENTICATED" ? (result.user ?? null) : null);
    return result.next_step;
  }, []);

  const refreshSession = useCallback(async () => {
    try {
      applyResult(await apiRequest<AuthResult>("/api/v1/auth/session"));
    } catch (error) {
      if (!(error instanceof ApiClientError) || error.status !== 401) console.error(error);
      setCsrfToken(null);
      setUser(null);
      setNextStep(null);
    } finally {
      setLoading(false);
    }
  }, [applyResult]);

  useEffect(() => {
    void refreshSession();
  }, [refreshSession]);

  const login = useCallback(async (email: string, password: string) => {
    const result = await apiRequest<AuthResult>("/api/v1/auth/login", {
      method: "POST",
      body: JSON.stringify({ email, password }),
    });
    return applyResult(result);
  }, [applyResult]);

  const register = useCallback(async (payload: RegistrationPayload) => {
    await apiRequest("/api/v1/auth/register", { method: "POST", body: JSON.stringify(payload) });
  }, []);

  const logout = useCallback(async () => {
    try {
      await apiRequest("/api/v1/auth/logout", { method: "POST" });
    } finally {
      setCsrfToken(null);
      setUser(null);
      setNextStep(null);
    }
  }, []);

  const changePassword = useCallback(async (newPassword: string, currentPassword?: string) => {
    const result = await apiRequest<AuthResult>("/api/v1/auth/password/change", {
      method: "POST",
      body: JSON.stringify({ new_password: newPassword, current_password: currentPassword }),
    });
    return applyResult(result);
  }, [applyResult]);

  const beginMfaEnrollment = useCallback(
    () => apiRequest<{ secret: string; provisioning_uri: string }>("/api/v1/auth/mfa/enrollment", { method: "POST" }),
    [],
  );

  const confirmMfaEnrollment = useCallback(async (code: string) => {
    const result = await apiRequest<AuthResult>("/api/v1/auth/mfa/enrollment/confirm", {
      method: "POST",
      body: JSON.stringify({ code, method: "totp" }),
    });
    applyResult(result);
    return result.recovery_codes ?? [];
  }, [applyResult]);

  const verifyMfa = useCallback(async (code: string, method: "totp" | "recovery_code" = "totp") => {
    applyResult(await apiRequest<AuthResult>("/api/v1/auth/mfa/verify", {
      method: "POST",
      body: JSON.stringify({ code, method }),
    }));
  }, [applyResult]);

  const value = useMemo(() => ({
    user, loading, nextStep, login, register, logout, changePassword,
    beginMfaEnrollment, confirmMfaEnrollment, verifyMfa, refreshSession,
  }), [
    user, loading, nextStep, login, register, logout, changePassword,
    beginMfaEnrollment, confirmMfaEnrollment, verifyMfa, refreshSession,
  ]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used inside AuthProvider");
  return context;
}

