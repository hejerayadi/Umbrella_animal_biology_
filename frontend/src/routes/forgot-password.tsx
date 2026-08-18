import { Link, createFileRoute } from "@tanstack/react-router";
import { ArrowLeft, ArrowRight, Loader2, Mail, MailCheck } from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useState, type FormEvent } from "react";

import { AuthBrandPanel, AuthTrustNote } from "@/components/umbrella/auth-brand-panel";
import { Reveal, Stagger, StaggerItem } from "@/components/umbrella/landing/reveal";
import { UmbrellaLogo } from "@/components/umbrella/logo";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { apiRequest } from "@/lib/api-client";

const TITLE = "Forgot password - Umbrella Animal BioHub";
const DESCRIPTION = "Request a secure password reset link for your Umbrella account.";
const EASE = [0.22, 1, 0.36, 1] as const;

export const Route = createFileRoute("/forgot-password")({
  head: () => ({
    meta: [
      { title: TITLE },
      { name: "description", content: DESCRIPTION },
      { property: "og:title", content: TITLE },
      { property: "og:description", content: DESCRIPTION },
    ],
  }),
  component: ForgotPassword,
});

function ForgotPassword() {
  const reduceMotion = useReducedMotion();
  const [email, setEmail] = useState("");
  const [sent, setSent] = useState(false);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setSubmitting(true);

    try {
      await apiRequest("/api/v1/auth/password/forgot", {
        method: "POST",
        body: JSON.stringify({ email }),
      });
      setSent(true);
    } catch (value) {
      setError(value instanceof Error ? value.message : "Unable to request a reset link.");
    } finally {
      setSubmitting(false);
    }
  }

  function resetForm() {
    setSent(false);
    setError("");
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
            <AnimatePresence mode="wait" initial={false}>
              {sent ? (
                <SentConfirmation
                  key="sent"
                  email={email}
                  onUseAnotherEmail={resetForm}
                  reduceMotion={Boolean(reduceMotion)}
                />
              ) : (
                <motion.div
                  key="request"
                  initial={reduceMotion ? false : { opacity: 0, y: 14 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={reduceMotion ? undefined : { opacity: 0, y: -12 }}
                  transition={{ duration: 0.35, ease: EASE }}
                >
                  <Reveal as="fade-up" eager duration={0.5} delay={0.2}>
                    <div className="auth-reset-heading text-center">
                      <span className="auth-reset-icon" aria-hidden="true">
                        <Mail />
                      </span>
                      <h1 className="text-3xl font-semibold sm:text-4xl">Forgot your password?</h1>
                      <p className="mt-3 text-base text-muted-foreground">
                        Enter your email and we will send you a secure, short-lived reset link.
                      </p>
                    </div>
                  </Reveal>

                  <form className="mt-9" onSubmit={handleSubmit}>
                    <Stagger className="space-y-5" delay={0.32} stagger={0.09} eager>
                      <StaggerItem>
                        <div className="space-y-2">
                          <Label htmlFor="forgot-password-email">Email address</Label>
                          <div className="auth-input-wrap">
                            <span className="auth-input-icon" aria-hidden="true">
                              <Mail />
                            </span>
                            <Input
                              id="forgot-password-email"
                              type="email"
                              autoComplete="email"
                              value={email}
                              onChange={(event) => setEmail(event.target.value)}
                              placeholder="you@example.com"
                              required
                            />
                          </div>
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
                            exit={
                              reduceMotion ? undefined : { opacity: 0, height: 0, marginTop: 0 }
                            }
                            transition={{ duration: 0.25, ease: EASE }}
                          >
                            {error}
                          </motion.p>
                        )}
                      </AnimatePresence>

                      <StaggerItem as="zoom-in">
                        <motion.div
                          whileHover={reduceMotion || submitting ? undefined : { scale: 1.015 }}
                          whileTap={reduceMotion || submitting ? undefined : { scale: 0.985 }}
                          transition={{ duration: 0.15, ease: EASE }}
                        >
                          <Button
                            type="submit"
                            size="lg"
                            className="auth-primary-button w-full"
                            disabled={submitting}
                          >
                            {submitting ? (
                              <Loader2 className="animate-spin" />
                            ) : (
                              <>
                                Send reset link <ArrowRight />
                              </>
                            )}
                          </Button>
                        </motion.div>
                      </StaggerItem>
                    </Stagger>
                  </form>

                  <Reveal as="fade-up" eager duration={0.5} delay={0.7}>
                    <p className="mt-8 text-center text-sm text-muted-foreground">
                      <Link
                        to="/signin"
                        className="inline-flex items-center gap-2 font-semibold text-primary hover:underline"
                      >
                        <ArrowLeft aria-hidden="true" className="size-4" />
                        Back to sign in
                      </Link>
                    </p>
                  </Reveal>
                </motion.div>
              )}
            </AnimatePresence>
          </div>
          <AuthTrustNote />
        </motion.section>
      </main>
      <footer className="auth-footer">© 2026 Umbrella Animal BioHub. All rights reserved.</footer>
    </div>
  );
}

function SentConfirmation({
  email,
  onUseAnotherEmail,
  reduceMotion,
}: {
  email: string;
  onUseAnotherEmail: () => void;
  reduceMotion: boolean;
}) {
  return (
    <motion.div
      className="auth-forgot-success text-center"
      initial={reduceMotion ? false : { opacity: 0, scale: 0.96, y: 14 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      exit={reduceMotion ? undefined : { opacity: 0, scale: 0.98, y: -10 }}
      transition={{ duration: 0.45, ease: EASE }}
    >
      <motion.span
        className="auth-reset-icon"
        aria-hidden="true"
        initial={reduceMotion ? false : { scale: 0.6, rotate: -12 }}
        animate={{ scale: 1, rotate: 0 }}
        transition={{ duration: 0.5, delay: 0.1, ease: EASE }}
      >
        <MailCheck />
      </motion.span>
      <h1 className="text-3xl font-semibold sm:text-4xl">Check your inbox</h1>
      <p className="mt-4 text-base leading-7 text-muted-foreground">
        If an eligible account exists for <strong className="text-foreground">{email}</strong>, a
        secure password reset link is on its way.
      </p>
      <p className="mt-3 text-sm text-muted-foreground">
        The link expires shortly. Check your spam folder if it does not arrive.
      </p>
      <Button asChild size="lg" className="auth-primary-button mt-8 w-full">
        <Link to="/signin">
          Back to sign in <ArrowRight />
        </Link>
      </Button>
      <button
        type="button"
        className="mt-5 text-sm font-semibold text-primary hover:underline"
        onClick={onUseAnotherEmail}
      >
        Use another email address
      </button>
    </motion.div>
  );
}
