import { Link, createFileRoute, useNavigate } from "@tanstack/react-router";
import {
  ArrowLeft,
  ArrowRight,
  Check,
  Eye,
  EyeOff,
  KeyRound,
  Loader2,
  LockKeyhole,
} from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useState, type FormEvent } from "react";

import { AuthBrandPanel, AuthTrustNote } from "@/components/umbrella/auth-brand-panel";
import { Reveal, Stagger, StaggerItem } from "@/components/umbrella/landing/reveal";
import { UmbrellaLogo } from "@/components/umbrella/logo";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { apiRequest } from "@/lib/api-client";

const TITLE = "Reset password - Umbrella Animal BioHub";
const DESCRIPTION = "Choose a new password for your Umbrella account.";
const EASE = [0.22, 1, 0.36, 1] as const;

export const Route = createFileRoute("/reset-password")({
  validateSearch: (search: Record<string, unknown>) => ({
    token: typeof search.token === "string" ? search.token : "",
  }),
  head: () => ({
    meta: [
      { title: TITLE },
      { name: "description", content: DESCRIPTION },
      { property: "og:title", content: TITLE },
      { property: "og:description", content: DESCRIPTION },
    ],
  }),
  component: ResetPassword,
});

function ResetPassword() {
  const { token } = Route.useSearch();
  const navigate = useNavigate();
  const reduceMotion = useReducedMotion();
  const [password, setPassword] = useState("");
  const [confirmation, setConfirmation] = useState("");
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  const hasValidLength = password.length >= 12 && password.length <= 128;
  const passwordsMatch = confirmation.length > 0 && password === confirmation;
  const canSubmit = Boolean(token) && hasValidLength && passwordsMatch && !submitting;

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");

    if (!token) {
      setError("This password reset link is missing or invalid. Request a new link to continue.");
      return;
    }
    if (!hasValidLength) {
      setError("Your password must contain between 12 and 128 characters.");
      return;
    }
    if (!passwordsMatch) {
      setError("Passwords do not match.");
      return;
    }

    setSubmitting(true);
    try {
      await apiRequest("/api/v1/auth/password/reset", {
        method: "POST",
        body: JSON.stringify({ token, new_password: password }),
      });
      navigate({ to: "/signin", replace: true });
    } catch (value) {
      setError(value instanceof Error ? value.message : "Password reset failed.");
    } finally {
      setSubmitting(false);
    }
  }

  return (
    <div className="auth-page auth-login-page">
      <div className="auth-mobile-brand">
        <Link to="/">
          <UmbrellaLogo detailed />
        </Link>
      </div>
      <main className="auth-login-layout">
        <AuthBrandPanel />
        <motion.section
          className="auth-login-column"
          initial={reduceMotion ? false : { opacity: 0, x: 44 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.65, delay: 0.1, ease: EASE }}
        >
          <div className="auth-card auth-login-card">
            <Reveal as="fade-up" eager duration={0.5} delay={0.2}>
              <div className="auth-reset-heading text-center">
                <span className="auth-reset-icon" aria-hidden="true">
                  <KeyRound />
                </span>
                <h1 className="text-3xl font-semibold sm:text-4xl">Choose a new password</h1>
                <p className="mt-3 text-base text-muted-foreground">
                  Create a strong password to secure your Umbrella account.
                </p>
              </div>
            </Reveal>

            <form className="mt-8" onSubmit={handleSubmit}>
              <Stagger className="space-y-5" delay={0.32} stagger={0.09} eager>
                {!token && (
                  <StaggerItem>
                    <p role="alert" className="auth-error">
                      This reset link is missing or invalid. Please request a new password reset
                      email.
                    </p>
                  </StaggerItem>
                )}

                <StaggerItem>
                  <PasswordField
                    id="reset-password"
                    label="New password"
                    value={password}
                    onChange={setPassword}
                    placeholder="Enter your new password"
                  />
                </StaggerItem>
                <StaggerItem>
                  <PasswordField
                    id="reset-password-confirmation"
                    label="Confirm new password"
                    value={confirmation}
                    onChange={setConfirmation}
                    placeholder="Confirm your new password"
                  />
                </StaggerItem>

                <StaggerItem>
                  <div className="auth-password-checks" aria-live="polite">
                    <span className={hasValidLength ? "is-valid" : undefined}>
                      <Check aria-hidden="true" />
                      12 to 128 characters
                    </span>
                    <span className={passwordsMatch ? "is-valid" : undefined}>
                      <Check aria-hidden="true" />
                      Passwords match
                    </span>
                  </div>
                </StaggerItem>

                <AnimatePresence mode="popLayout">
                  {error && (
                    <motion.p
                      key={error}
                      role="alert"
                      className="auth-error"
                      initial={reduceMotion ? false : { opacity: 0, y: -8, scale: 0.98 }}
                      animate={{ opacity: 1, y: 0, scale: 1 }}
                      exit={reduceMotion ? undefined : { opacity: 0, height: 0, marginTop: 0 }}
                      transition={{ duration: 0.25, ease: EASE }}
                    >
                      {error}
                    </motion.p>
                  )}
                </AnimatePresence>

                <StaggerItem as="zoom-in">
                  <motion.div
                    whileHover={reduceMotion || !canSubmit ? undefined : { scale: 1.015 }}
                    whileTap={reduceMotion || !canSubmit ? undefined : { scale: 0.985 }}
                    transition={{ duration: 0.15, ease: EASE }}
                  >
                    <Button
                      type="submit"
                      size="lg"
                      className="auth-primary-button w-full"
                      disabled={!canSubmit}
                    >
                      {submitting ? (
                        <Loader2 className="animate-spin" />
                      ) : (
                        <>
                          Reset password <ArrowRight />
                        </>
                      )}
                    </Button>
                  </motion.div>
                </StaggerItem>
              </Stagger>
            </form>

            <Reveal as="fade-up" eager duration={0.5} delay={0.85}>
              <p className="mt-8 text-center text-sm text-muted-foreground">
                <Link
                  to={token ? "/signin" : "/forgot-password"}
                  className="inline-flex items-center gap-2 font-semibold text-primary hover:underline"
                >
                  <ArrowLeft aria-hidden="true" className="size-4" />
                  {token ? "Back to sign in" : "Request a new reset link"}
                </Link>
              </p>
            </Reveal>
          </div>
          <AuthTrustNote />
        </motion.section>
      </main>
      <footer className="auth-footer">© 2026 Umbrella Animal BioHub. All rights reserved.</footer>
    </div>
  );
}

function PasswordField({
  id,
  label,
  value,
  onChange,
  placeholder,
}: {
  id: string;
  label: string;
  value: string;
  onChange: (value: string) => void;
  placeholder: string;
}) {
  const reduceMotion = useReducedMotion();
  const [visible, setVisible] = useState(false);

  return (
    <div className="space-y-2">
      <Label htmlFor={id}>{label}</Label>
      <div className="auth-input-wrap">
        <span className="auth-input-icon" aria-hidden="true">
          <LockKeyhole />
        </span>
        <Input
          id={id}
          type={visible ? "text" : "password"}
          autoComplete="new-password"
          minLength={12}
          maxLength={128}
          value={value}
          onChange={(event) => onChange(event.target.value)}
          placeholder={placeholder}
          required
        />
        <button
          type="button"
          className="auth-input-action"
          onClick={() => setVisible((current) => !current)}
          aria-label={visible ? `Hide ${label.toLowerCase()}` : `Show ${label.toLowerCase()}`}
        >
          <AnimatePresence mode="wait" initial={false}>
            <motion.span
              key={visible ? "hide" : "show"}
              className="grid place-items-center"
              initial={reduceMotion ? false : { opacity: 0, scale: 0.6, rotate: -45 }}
              animate={{ opacity: 1, scale: 1, rotate: 0 }}
              exit={reduceMotion ? undefined : { opacity: 0, scale: 0.6, rotate: 45 }}
              transition={{ duration: 0.18, ease: EASE }}
            >
              {visible ? <EyeOff /> : <Eye />}
            </motion.span>
          </AnimatePresence>
        </button>
      </div>
    </div>
  );
}
