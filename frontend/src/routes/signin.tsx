import { Link, createFileRoute, useNavigate } from "@tanstack/react-router";
import { ArrowRight, Eye, EyeOff, Loader2, LockKeyhole, Mail } from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useState, type FormEvent, type ReactNode } from "react";

import { AuthBrandPanel, AuthTrustNote } from "@/components/umbrella/auth-brand-panel";
import { UmbrellaLogo } from "@/components/umbrella/logo";
import { Reveal, Stagger, StaggerItem } from "@/components/umbrella/landing/reveal";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { resolveAuthDestination, useAuth } from "@/lib/auth-context";

const TITLE = "Sign in - Umbrella Animal BioHub";
const DESCRIPTION = "Sign in to continue your animal genomics research with Umbrella.";
const EASE = [0.22, 1, 0.36, 1] as const;

export const Route = createFileRoute("/signin")({
  head: () => ({
    meta: [
      { title: TITLE },
      { name: "description", content: DESCRIPTION },
      { property: "og:title", content: TITLE },
      { property: "og:description", content: DESCRIPTION },
    ],
  }),
  component: SignIn,
});

function SignIn() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const reduceMotion = useReducedMotion();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState("");
  const [submitting, setSubmitting] = useState(false);

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
              <div className="text-center">
                <h1 className="text-3xl font-semibold sm:text-4xl">Welcome back</h1>
                <p className="mt-3 text-base text-muted-foreground sm:text-lg">
                  Sign in to continue to <span className="font-medium text-primary">Umbrella</span>
                </p>
              </div>
            </Reveal>

            <form
              className="mt-9"
              onSubmit={async (event: FormEvent) => {
                event.preventDefault();
                setSubmitting(true);
                setError("");
                try {
                  const step = await login(email, password);
                  navigate({ to: resolveAuthDestination(step) });
                } catch (value) {
                  setError(value instanceof Error ? value.message : "Sign in failed.");
                } finally {
                  setSubmitting(false);
                }
              }}
            >
              <Stagger className="space-y-5" delay={0.32} stagger={0.09} eager>
                <StaggerItem>
                  <AuthField icon={<Mail />} label="Email address" htmlFor="signin-email">
                    <Input
                      id="signin-email"
                      type="email"
                      autoComplete="email"
                      value={email}
                      onChange={(event) => setEmail(event.target.value)}
                      placeholder="you@example.com"
                      required
                    />
                  </AuthField>
                </StaggerItem>
                <StaggerItem>
                  <AuthField icon={<LockKeyhole />} label="Password" htmlFor="signin-password">
                    <Input
                      id="signin-password"
                      type={showPassword ? "text" : "password"}
                      autoComplete="current-password"
                      value={password}
                      onChange={(event) => setPassword(event.target.value)}
                      placeholder="Enter your password"
                      required
                    />
                    <button
                      type="button"
                      className="auth-input-action"
                      onClick={() => setShowPassword((value) => !value)}
                      aria-label={showPassword ? "Hide password" : "Show password"}
                    >
                      <AnimatePresence mode="wait" initial={false}>
                        <motion.span
                          key={showPassword ? "hide" : "show"}
                          className="grid place-items-center"
                          initial={reduceMotion ? false : { opacity: 0, scale: 0.6, rotate: -45 }}
                          animate={{ opacity: 1, scale: 1, rotate: 0 }}
                          exit={reduceMotion ? undefined : { opacity: 0, scale: 0.6, rotate: 45 }}
                          transition={{ duration: 0.18, ease: EASE }}
                        >
                          {showPassword ? <EyeOff /> : <Eye />}
                        </motion.span>
                      </AnimatePresence>
                    </button>
                  </AuthField>
                </StaggerItem>
                <StaggerItem className="flex justify-end">
                  <Link
                    to="/forgot-password"
                    className="text-sm font-medium text-primary hover:underline"
                  >
                    Forgot password?
                  </Link>
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
                    whileHover={reduceMotion ? undefined : { scale: 1.015 }}
                    whileTap={reduceMotion ? undefined : { scale: 0.985 }}
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
                          Sign in <ArrowRight />
                        </>
                      )}
                    </Button>
                  </motion.div>
                </StaggerItem>
              </Stagger>
            </form>

            <Reveal as="fade-up" eager duration={0.5} delay={0.85}>
              <p className="mt-8 text-center text-sm text-muted-foreground">
                Don't have an account?{" "}
                <Link to="/signup" className="font-semibold text-primary hover:underline">
                  Sign up
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

function AuthField({
  icon,
  label,
  htmlFor,
  children,
}: {
  icon: ReactNode;
  label: string;
  htmlFor: string;
  children: ReactNode;
}) {
  return (
    <div className="space-y-2">
      <Label htmlFor={htmlFor}>{label}</Label>
      <div className="auth-input-wrap">
        <span className="auth-input-icon" aria-hidden="true">
          {icon}
        </span>
        {children}
      </div>
    </div>
  );
}
