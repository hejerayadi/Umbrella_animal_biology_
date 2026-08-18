import { Link, createFileRoute, useNavigate } from "@tanstack/react-router";
import {
  ArrowRight,
  Check,
  Copy,
  Download,
  KeyRound,
  Loader2,
  QrCode,
  ShieldCheck,
  Smartphone,
} from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { REGEXP_ONLY_DIGITS } from "input-otp";
import { QRCodeSVG } from "qrcode.react";
import { useEffect, useRef, useState, type FormEvent } from "react";

import { AuthBrandPanel, AuthTrustNote } from "@/components/umbrella/auth-brand-panel";
import { Reveal } from "@/components/umbrella/landing/reveal";
import { UmbrellaLogo } from "@/components/umbrella/logo";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { InputOTP, InputOTPGroup, InputOTPSlot } from "@/components/ui/input-otp";
import { Label } from "@/components/ui/label";
import { resolveAuthDestination, useAuth } from "@/lib/auth-context";

const TITLE = "Two-factor authentication - Umbrella Animal BioHub";
const DESCRIPTION = "Secure your Umbrella administrator account with two-factor authentication.";
const EASE = [0.22, 1, 0.36, 1] as const;

interface EnrollmentData {
  secret: string;
  provisioning_uri: string;
}

export const Route = createFileRoute("/mfa")({
  head: () => ({
    meta: [
      { title: TITLE },
      { name: "description", content: DESCRIPTION },
      { property: "og:title", content: TITLE },
      { property: "og:description", content: DESCRIPTION },
    ],
  }),
  component: Mfa,
});

function Mfa() {
  const { loading, nextStep, beginMfaEnrollment, confirmMfaEnrollment, verifyMfa } = useAuth();
  const navigate = useNavigate();
  const reduceMotion = useReducedMotion();
  const enrollmentRequested = useRef(false);
  const [code, setCode] = useState("");
  const [enrollmentData, setEnrollmentData] = useState<EnrollmentData | null>(null);
  const [recoveryCodes, setRecoveryCodes] = useState<string[]>([]);
  const [useRecovery, setUseRecovery] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState("");

  const enrollment = nextStep === "MFA_ENROLLMENT_REQUIRED";

  useEffect(() => {
    if (loading || recoveryCodes.length) return;

    // This effect is the only thing allowed to navigate away from /mfa.
    // Handlers below only call the API and update auth state; once nextStep
    // changes, this re-runs and decides where to go - no competing navigate.
    const target = resolveAuthDestination(nextStep);
    if (target !== "/mfa") {
      navigate({ to: target, replace: true });
      return;
    }

    if (nextStep === "MFA_ENROLLMENT_REQUIRED" && !enrollmentRequested.current) {
      enrollmentRequested.current = true;
      void beginMfaEnrollment()
        .then(setEnrollmentData)
        .catch((value) => {
          enrollmentRequested.current = false;
          setError(value instanceof Error ? value.message : "Unable to start MFA enrollment.");
        });
    }
  }, [beginMfaEnrollment, loading, navigate, nextStep, recoveryCodes.length]);

  async function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setError("");
    setSubmitting(true);

    try {
      if (enrollment) {
        setRecoveryCodes(await confirmMfaEnrollment(code));
      } else {
        // No navigate here: verifyMfa() flips nextStep to AUTHENTICATED,
        // and the effect above reacts to that and moves on to /chat.
        await verifyMfa(code, useRecovery ? "recovery_code" : "totp");
      }
    } catch (value) {
      setError(value instanceof Error ? value.message : "Verification failed.");
    } finally {
      setSubmitting(false);
    }
  }

  async function copySecret() {
    if (!enrollmentData?.secret) return;
    try {
      await navigator.clipboard.writeText(enrollmentData.secret);
      setCopied(true);
      window.setTimeout(() => setCopied(false), 1800);
    } catch {
      setError("Unable to copy the setup key. Select and copy it manually.");
    }
  }

  function selectVerificationMethod(recovery: boolean) {
    setUseRecovery(recovery);
    setCode("");
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
          <div className="auth-card auth-login-card auth-mfa-card">
            <AnimatePresence mode="wait" initial={false}>
              {loading ||
              (!recoveryCodes.length &&
                nextStep !== "MFA_REQUIRED" &&
                nextStep !== "MFA_ENROLLMENT_REQUIRED") ? (
                <MfaLoading key="loading" reduceMotion={Boolean(reduceMotion)} />
              ) : recoveryCodes.length ? (
                <RecoveryCodes
                  key="recovery-codes"
                  codes={recoveryCodes}
                  onContinue={() => navigate({ to: "/chat", replace: true })}
                  reduceMotion={Boolean(reduceMotion)}
                />
              ) : (
                <motion.div
                  key={enrollment ? "enrollment" : "verification"}
                  initial={reduceMotion ? false : { opacity: 0, y: 14 }}
                  animate={{ opacity: 1, y: 0 }}
                  exit={reduceMotion ? undefined : { opacity: 0, y: -12 }}
                  transition={{ duration: 0.4, ease: EASE }}
                >
                  <Reveal as="fade-up" eager duration={0.5} delay={0.15}>
                    <div className="auth-reset-heading text-center">
                      <span className="auth-reset-icon" aria-hidden="true">
                        {enrollment ? <QrCode /> : <ShieldCheck />}
                      </span>
                      <h1 className="text-3xl font-semibold sm:text-4xl">
                        {enrollment ? "Set up your authenticator" : "Verify it is you"}
                      </h1>
                      <p className="mt-3 text-base text-muted-foreground">
                        {enrollment
                          ? "Scan the QR code, then enter the six-digit code from your app."
                          : "Enter a fresh code from your authenticator app to continue."}
                      </p>
                    </div>
                  </Reveal>

                  {enrollment && (
                    <EnrollmentSetup
                      data={enrollmentData}
                      copied={copied}
                      onCopySecret={copySecret}
                    />
                  )}

                  {!enrollment && (
                    <div className="auth-choice-control auth-mfa-methods mt-7">
                      <button
                        type="button"
                        className={!useRecovery ? "is-selected" : undefined}
                        onClick={() => selectVerificationMethod(false)}
                      >
                        <Smartphone aria-hidden="true" />
                        Authenticator app
                      </button>
                      <button
                        type="button"
                        className={useRecovery ? "is-selected" : undefined}
                        onClick={() => selectVerificationMethod(true)}
                      >
                        <KeyRound aria-hidden="true" />
                        Recovery code
                      </button>
                    </div>
                  )}

                  <form className={enrollment ? "mt-7" : "mt-6"} onSubmit={handleSubmit}>
                    <div className="space-y-3">
                      <Label htmlFor={useRecovery ? "mfa-recovery-code" : "mfa-code"}>
                        {useRecovery ? "Recovery code" : "Authentication code"}
                      </Label>
                      {useRecovery ? (
                        <div className="auth-input-wrap">
                          <span className="auth-input-icon" aria-hidden="true">
                            <KeyRound />
                          </span>
                          <Input
                            id="mfa-recovery-code"
                            autoComplete="one-time-code"
                            value={code}
                            onChange={(event) => setCode(event.target.value)}
                            placeholder="Enter one recovery code"
                            required
                          />
                        </div>
                      ) : (
                        <InputOTP
                          id="mfa-code"
                          value={code}
                          onChange={setCode}
                          maxLength={6}
                          pattern={REGEXP_ONLY_DIGITS}
                          autoComplete="one-time-code"
                          containerClassName="auth-mfa-otp"
                          aria-label="Six-digit authentication code"
                        >
                          <InputOTPGroup>
                            {Array.from({ length: 6 }, (_, index) => (
                              <InputOTPSlot
                                key={index}
                                index={index}
                                className="auth-mfa-otp-slot"
                              />
                            ))}
                          </InputOTPGroup>
                        </InputOTP>
                      )}
                    </div>

                    <AnimatePresence mode="popLayout">
                      {error && (
                        <motion.p
                          key={error}
                          role="alert"
                          className="auth-error mt-5"
                          initial={reduceMotion ? false : { opacity: 0, y: -8, scale: 0.98 }}
                          animate={{ opacity: 1, y: 0, scale: 1 }}
                          exit={reduceMotion ? undefined : { opacity: 0, height: 0, marginTop: 0 }}
                          transition={{ duration: 0.25, ease: EASE }}
                        >
                          {error}
                        </motion.p>
                      )}
                    </AnimatePresence>

                    <motion.div
                      className="mt-6"
                      whileHover={reduceMotion || submitting ? undefined : { scale: 1.015 }}
                      whileTap={reduceMotion || submitting ? undefined : { scale: 0.985 }}
                      transition={{ duration: 0.15, ease: EASE }}
                    >
                      <Button
                        type="submit"
                        size="lg"
                        className="auth-primary-button w-full"
                        disabled={
                          submitting ||
                          (enrollment && !enrollmentData) ||
                          (!useRecovery && code.length !== 6) ||
                          (useRecovery && !code.trim())
                        }
                      >
                        {submitting ? (
                          <Loader2 className="animate-spin" />
                        ) : (
                          <>
                            {enrollment ? "Confirm and enable MFA" : "Verify and continue"}
                            <ArrowRight />
                          </>
                        )}
                      </Button>
                    </motion.div>
                  </form>
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

function MfaLoading({ reduceMotion }: { reduceMotion: boolean }) {
  return (
    <motion.div
      className="auth-mfa-page-loading"
      initial={reduceMotion ? false : { opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={reduceMotion ? undefined : { opacity: 0 }}
    >
      <Loader2 className="animate-spin" aria-hidden="true" />
      <p>Preparing secure verification...</p>
    </motion.div>
  );
}

function EnrollmentSetup({
  data,
  copied,
  onCopySecret,
}: {
  data: EnrollmentData | null;
  copied: boolean;
  onCopySecret: () => void;
}) {
  return (
    <div className="auth-mfa-setup">
      <div className="auth-mfa-qr" aria-live="polite">
        {data ? (
          <QRCodeSVG
            value={data.provisioning_uri}
            size={184}
            level="M"
            marginSize={4}
            bgColor="#ffffff"
            fgColor="#211d1b"
            title="Scan to add Umbrella to your authenticator app"
          />
        ) : (
          <span className="auth-mfa-loading">
            <Loader2 className="animate-spin" aria-hidden="true" />
            <span className="sr-only">Generating QR code</span>
          </span>
        )}
      </div>
      <div className="auth-mfa-setup-copy">
        <div>
          <span>01</span>
          <p>
            <strong>Scan the QR code</strong>
            Open your authenticator app and add a new account.
          </p>
        </div>
        <div>
          <span>02</span>
          <p>
            <strong>Enter the generated code</strong>
            Use the current six-digit code to confirm setup.
          </p>
        </div>
        <div className="auth-mfa-manual-key">
          <small>Cannot scan? Enter this setup key manually.</small>
          <div>
            <code>{data?.secret ?? "Generating secure key..."}</code>
            <button
              type="button"
              onClick={onCopySecret}
              disabled={!data}
              aria-label="Copy manual setup key"
              title="Copy setup key"
            >
              {copied ? <Check /> : <Copy />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

function RecoveryCodes({
  codes,
  onContinue,
  reduceMotion,
}: {
  codes: string[];
  onContinue: () => void;
  reduceMotion: boolean;
}) {
  const [copied, setCopied] = useState(false);

  async function copyCodes() {
    await navigator.clipboard.writeText(codes.join("\n"));
    setCopied(true);
    window.setTimeout(() => setCopied(false), 1800);
  }

  function downloadCodes() {
    const file = new Blob(
      ["Umbrella Animal BioHub - MFA recovery codes\n\n", codes.join("\n"), "\n"],
      { type: "text/plain;charset=utf-8" },
    );
    const url = URL.createObjectURL(file);
    const link = document.createElement("a");
    link.href = url;
    link.download = "umbrella-recovery-codes.txt";
    link.click();
    URL.revokeObjectURL(url);
  }

  return (
    <motion.div
      className="auth-recovery-view text-center"
      initial={reduceMotion ? false : { opacity: 0, scale: 0.96, y: 14 }}
      animate={{ opacity: 1, scale: 1, y: 0 }}
      transition={{ duration: 0.45, ease: EASE }}
    >
      <span className="auth-reset-icon" aria-hidden="true">
        <ShieldCheck />
      </span>
      <h1 className="text-3xl font-semibold sm:text-4xl">Save your recovery codes</h1>
      <p className="mt-3 text-base text-muted-foreground">
        Each code can be used once if you lose access to your authenticator. They will not be shown
        again.
      </p>
      <div className="auth-recovery-grid">
        {codes.map((recoveryCode) => (
          <code key={recoveryCode}>{recoveryCode}</code>
        ))}
      </div>
      <div className="auth-recovery-actions">
        <Button type="button" variant="outline" onClick={copyCodes}>
          {copied ? <Check /> : <Copy />}
          {copied ? "Copied" : "Copy codes"}
        </Button>
        <Button type="button" variant="outline" onClick={downloadCodes}>
          <Download /> Download
        </Button>
      </div>
      <Button
        type="button"
        size="lg"
        className="auth-primary-button mt-6 w-full"
        onClick={onContinue}
      >
        I saved my codes <ArrowRight />
      </Button>
    </motion.div>
  );
}
