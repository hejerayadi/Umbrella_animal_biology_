import { Link, createFileRoute } from "@tanstack/react-router";
import {
  ArrowLeft,
  ArrowRight,
  BookOpen,
  Check,
  CheckCircle2,
  Crosshair,
  Dna,
  Eye,
  EyeOff,
  GitFork,
  GraduationCap,
  LibraryBig,
  Loader2,
  LockKeyhole,
  MailCheck,
  Microscope,
  Network,
  Orbit,
  RotateCw,
  ShieldCheck,
  Sparkles,
  Trees,
  UserRound,
  X,
} from "lucide-react";
import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import { useState, type ReactNode } from "react";

import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Textarea } from "@/components/ui/textarea";
import { AuthTrustNote } from "@/components/umbrella/auth-brand-panel";
import { FieldHint } from "@/components/umbrella/field-hint";
import { UmbrellaLogo } from "@/components/umbrella/logo";
import { apiRequest } from "@/lib/api-client";
import { useAuth } from "@/lib/auth-context";
import { cn } from "@/lib/utils";
import { evaluatePasswordPolicy, getPasswordStrength } from "@/lib/password-strength";

const EASE = [0.22, 1, 0.36, 1] as const;
const STEPS = ["Create account", "Your profile", "Organization", "Verify email"] as const;
const ROLES = [
  { label: "Researcher", icon: Microscope },
  { label: "Student", icon: GraduationCap },
  { label: "Educator", icon: BookOpen },
  { label: "Other", icon: UserRound },
] as const;
const INTERESTS = [
  {
    label: "Genome analysis",
    description: "Assembly and reconstruction",
    icon: Dna,
  },
  {
    label: "Biodiversity",
    description: "Species and ecosystem signals",
    icon: Trees,
  },
  {
    label: "Evolution",
    description: "Phylogeny and divergence",
    icon: GitFork,
  },
  {
    label: "Trait discovery",
    description: "Phenotypes and candidate loci",
    icon: Crosshair,
  },
  {
    label: "Protein science",
    description: "Structure and function",
    icon: Orbit,
  },
  {
    label: "Scientific literature",
    description: "Evidence and synthesis",
    icon: LibraryBig,
  },
  {
    label: "AI orchestration",
    description: "Multi-agent research systems",
    icon: Network,
  },
  {
    label: "Other",
    description: "Add your own specialty",
    icon: Sparkles,
  },
] as const;

type FormState = {
  fullName: string;
  email: string;
  password: string;
  confirmPassword: string;
  role: string;
  fieldInterests: string[];
  otherInterest: string;
  country: string;
  motivation: string;
  organizationMode: "join" | "create";
  institution: string;
  department: string;
  orcid: string;
};

const INITIAL_FORM: FormState = {
  fullName: "",
  email: "",
  password: "",
  confirmPassword: "",
  role: "",
  fieldInterests: [],
  otherInterest: "",
  country: "",
  motivation: "",
  organizationMode: "join",
  institution: "",
  department: "",
  orcid: "",
};

export const Route = createFileRoute("/signup")({
  head: () => ({
    meta: [
      { title: "Create your account - Umbrella Animal BioHub" },
      {
        name: "description",
        content: "Apply to join Umbrella's animal genomics research platform.",
      },
    ],
  }),
  component: SignUp,
});

function SignUp() {
  const { register } = useAuth();
  const reduceMotion = useReducedMotion();
  const [step, setStep] = useState(1);
  const [form, setForm] = useState<FormState>(INITIAL_FORM);
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [submitting, setSubmitting] = useState(false);
  const [resending, setResending] = useState(false);
  const [notice, setNotice] = useState("");
  const [error, setError] = useState("");

  const update = <K extends keyof FormState>(field: K, value: FormState[K]) =>
    setForm((current) => ({ ...current, [field]: value }));

  const toggleInterest = (interest: string) =>
    setForm((current) => ({
      ...current,
      fieldInterests: current.fieldInterests.includes(interest)
        ? current.fieldInterests.filter((item) => item !== interest)
        : [...current.fieldInterests, interest],
    }));

  const strength = getPasswordStrength(form.password);
  const passwordPolicy = evaluatePasswordPolicy(form.password, form.email, form.fullName);
  const confirmMatches = form.confirmPassword.length > 0 && form.password === form.confirmPassword;

  const accountValid =
    form.fullName.trim().length >= 2 &&
    /^\S+@\S+\.\S+$/.test(form.email) &&
    passwordPolicy.valid &&
    form.password === form.confirmPassword;
  const roleValid = Boolean(form.role);
  const countryValid = form.country.trim().length === 2;
  const motivationValid = form.motivation.trim().length >= 20;
  const profileValid = roleValid && countryValid && motivationValid;

  const submitApplication = async (skipOrganization = false) => {
    setSubmitting(true);
    setError("");
    try {
      const institution = skipOrganization
        ? "Independent researcher"
        : [form.institution.trim(), form.department.trim()].filter(Boolean).join(" - ");
      await register({
        email: form.email,
        password: form.password,
        full_name: form.fullName.trim(),
        institution,
        professional_title: form.role,
        country: form.country.toUpperCase(),
        orcid: form.orcid.trim() || undefined,
        motivation: form.motivation.trim(),
        specialties: [
          ...form.fieldInterests.filter((interest) => interest !== "Other"),
          ...(form.fieldInterests.includes("Other") ? [form.otherInterest.trim() || "Other"] : []),
        ],
      });
      setStep(4);
    } catch (value) {
      setError(value instanceof Error ? value.message : "Application could not be submitted.");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="auth-page auth-signup-page">
      <header className="auth-signup-header">
        <Link to="/" aria-label="Umbrella home">
          <UmbrellaLogo detailed />
        </Link>
        <p>
          One platform. All animal genomics. <span>Powered by AI.</span>
        </p>
        <Link to="/signin" className="auth-header-link">
          Sign in
        </Link>
      </header>

      <main className="auth-signup-main">
        <RegistrationStepper current={step} />
        <div className="auth-card auth-signup-card">
          <AnimatePresence mode="wait" initial={false}>
            <motion.div
              key={step}
              initial={reduceMotion ? false : { opacity: 0, y: 14 }}
              animate={{ opacity: 1, y: 0 }}
              exit={reduceMotion ? undefined : { opacity: 0, y: -12 }}
              transition={{ duration: 0.35, ease: EASE }}
            >
              {step === 1 && (
                <form
                  onSubmit={(event) => {
                    event.preventDefault();
                    if (accountValid) setStep(2);
                  }}
                >
                  <StepHeading eyebrow="Step 1 of 4" title="Create your account">
                    Start your application to join the Umbrella research community.
                  </StepHeading>
                  <div className="auth-form-grid mt-8">
                    <Field
                      label="Full name"
                      htmlFor="full-name"
                      hint="Your full name as it will appear on your Umbrella profile and shared research threads."
                    >
                      <Input
                        id="full-name"
                        autoComplete="name"
                        value={form.fullName}
                        onChange={(e) => update("fullName", e.target.value)}
                        placeholder="Dr. Alex Morgan"
                        required
                      />
                    </Field>
                    <Field
                      label="Email address"
                      htmlFor="signup-email"
                      hint="We'll send your verification link here. Use an address you check regularly."
                    >
                      <Input
                        id="signup-email"
                        type="email"
                        autoComplete="email"
                        value={form.email}
                        onChange={(e) => update("email", e.target.value)}
                        placeholder="you@example.com"
                        required
                      />
                    </Field>
                    <Field
                      label="Password"
                      htmlFor="signup-password"
                      hint="At least 12 characters. Mixing uppercase, lowercase, numbers, and symbols makes it stronger."
                    >
                      <div className="auth-input-wrap">
                        <span className="auth-input-icon" aria-hidden="true">
                          <LockKeyhole />
                        </span>
                        <Input
                          id="signup-password"
                          type={showPassword ? "text" : "password"}
                          autoComplete="new-password"
                          minLength={12}
                          value={form.password}
                          onChange={(e) => update("password", e.target.value)}
                          placeholder="At least 12 characters"
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
                              initial={
                                reduceMotion ? false : { opacity: 0, scale: 0.6, rotate: -45 }
                              }
                              animate={{ opacity: 1, scale: 1, rotate: 0 }}
                              exit={
                                reduceMotion ? undefined : { opacity: 0, scale: 0.6, rotate: 45 }
                              }
                              transition={{ duration: 0.18, ease: EASE }}
                            >
                              {showPassword ? <EyeOff /> : <Eye />}
                            </motion.span>
                          </AnimatePresence>
                        </button>
                      </div>
                      <AnimatePresence>
                        {form.password && (
                          <motion.div
                            className="auth-password-feedback"
                            initial={reduceMotion ? false : { opacity: 0, height: 0 }}
                            animate={{ opacity: 1, height: "auto" }}
                            exit={reduceMotion ? undefined : { opacity: 0, height: 0 }}
                            transition={{ duration: 0.25, ease: EASE }}
                          >
                            <div className={cn("auth-strength-meter", `is-${strength.score}`)}>
                              <div className="auth-strength-track">
                                {[1, 2, 3, 4].map((segment) => (
                                  <motion.span
                                    key={segment}
                                    className={segment <= strength.score ? "is-filled" : undefined}
                                    initial={reduceMotion ? false : { scaleX: 0 }}
                                    animate={{ scaleX: segment <= strength.score ? 1 : 0.2 }}
                                    transition={{
                                      duration: 0.3,
                                      delay: segment * 0.04,
                                      ease: EASE,
                                    }}
                                  />
                                ))}
                              </div>
                              <span className="auth-strength-label">{strength.label}</span>
                            </div>
                            <p
                              className={cn(
                                "auth-policy-hint",
                                passwordPolicy.valid ? "is-valid" : "is-invalid",
                              )}
                              role={passwordPolicy.valid ? undefined : "alert"}
                            >
                              {passwordPolicy.valid ? (
                                <>
                                  <Check aria-hidden="true" /> Password meets the account policy
                                </>
                              ) : (
                                <>
                                  <X aria-hidden="true" /> {passwordPolicy.message}
                                </>
                              )}
                            </p>
                          </motion.div>
                        )}
                      </AnimatePresence>
                    </Field>
                    <Field
                      label="Confirm password"
                      htmlFor="confirm-password"
                      hint="Re-enter the exact same password to confirm there's no typo."
                    >
                      <div className="auth-input-wrap">
                        <span className="auth-input-icon" aria-hidden="true">
                          <LockKeyhole />
                        </span>
                        <Input
                          id="confirm-password"
                          type={showConfirmPassword ? "text" : "password"}
                          autoComplete="new-password"
                          value={form.confirmPassword}
                          onChange={(e) => update("confirmPassword", e.target.value)}
                          placeholder="Repeat your password"
                          required
                        />
                        <button
                          type="button"
                          className="auth-input-action"
                          onClick={() => setShowConfirmPassword((value) => !value)}
                          aria-label={showConfirmPassword ? "Hide password" : "Show password"}
                        >
                          <AnimatePresence mode="wait" initial={false}>
                            <motion.span
                              key={showConfirmPassword ? "hide" : "show"}
                              className="grid place-items-center"
                              initial={
                                reduceMotion ? false : { opacity: 0, scale: 0.6, rotate: -45 }
                              }
                              animate={{ opacity: 1, scale: 1, rotate: 0 }}
                              exit={
                                reduceMotion ? undefined : { opacity: 0, scale: 0.6, rotate: 45 }
                              }
                              transition={{ duration: 0.18, ease: EASE }}
                            >
                              {showConfirmPassword ? <EyeOff /> : <Eye />}
                            </motion.span>
                          </AnimatePresence>
                        </button>
                      </div>
                      <AnimatePresence mode="wait">
                        {form.confirmPassword && (
                          <motion.p
                            key={confirmMatches ? "match" : "mismatch"}
                            className={cn(
                              "auth-match-hint",
                              confirmMatches ? "is-match" : "is-mismatch",
                            )}
                            initial={reduceMotion ? false : { opacity: 0, y: -4 }}
                            animate={{ opacity: 1, y: 0 }}
                            exit={reduceMotion ? undefined : { opacity: 0 }}
                            transition={{ duration: 0.2, ease: EASE }}
                          >
                            {confirmMatches ? (
                              <>
                                <Check aria-hidden="true" /> Passwords match
                              </>
                            ) : (
                              <>
                                <X aria-hidden="true" /> Passwords do not match
                              </>
                            )}
                          </motion.p>
                        )}
                      </AnimatePresence>
                    </Field>
                  </div>
                  <StepActions>
                    <span className="text-sm text-muted-foreground">
                      Already have an account?{" "}
                      <Link to="/signin" className="font-semibold text-primary">
                        Sign in
                      </Link>
                    </span>
                    <Button
                      type="submit"
                      size="lg"
                      className="auth-primary-button"
                      disabled={!accountValid}
                    >
                      Continue <ArrowRight />
                    </Button>
                  </StepActions>
                </form>
              )}

              {step === 2 && (
                <form
                  onSubmit={(event) => {
                    event.preventDefault();
                    if (profileValid) setStep(3);
                  }}
                >
                  <StepHeading eyebrow="Step 2 of 4" title="Tell us about yourself">
                    Help administrators understand your research background.
                  </StepHeading>
                  <div className="mt-8 space-y-6">
                    <fieldset>
                      <legend className="mb-3 text-sm font-medium">
                        Which best describes you?
                      </legend>
                      <div className="auth-role-grid">
                        {ROLES.map(({ label, icon: Icon }) => (
                          <button
                            key={label}
                            type="button"
                            className={cn("auth-role-option", form.role === label && "is-selected")}
                            onClick={() => update("role", label)}
                            aria-pressed={form.role === label}
                          >
                            <Icon />
                            <span>{label}</span>
                            {form.role === label && <Check />}
                          </button>
                        ))}
                      </div>
                    </fieldset>
                    <fieldset>
                      <div className="auth-interest-legend">
                        <legend>Fields of interest (optional)</legend>
                        <span>{form.fieldInterests.length} selected</span>
                      </div>
                      <p className="auth-interest-help">
                        Select every area relevant to your work. This helps Umbrella route research
                        questions to the right specialist agents.
                      </p>
                      <div className="auth-interest-grid">
                        {INTERESTS.map(({ label, description, icon: Icon }) => {
                          const selected = form.fieldInterests.includes(label);
                          return (
                            <button
                              key={label}
                              type="button"
                              className={cn("auth-interest-option", selected && "is-selected")}
                              aria-pressed={selected}
                              onClick={() => toggleInterest(label)}
                            >
                              <span className="auth-interest-icon">
                                <Icon />
                              </span>
                              <span>
                                <strong>{label}</strong>
                                <small>{description}</small>
                              </span>
                              <span className="auth-interest-check" aria-hidden="true">
                                <Check />
                              </span>
                            </button>
                          );
                        })}
                      </div>
                    </fieldset>
                    <AnimatePresence initial={false}>
                      {form.fieldInterests.includes("Other") && (
                        <motion.div
                          initial={reduceMotion ? false : { opacity: 0, height: 0 }}
                          animate={{ opacity: 1, height: "auto" }}
                          exit={reduceMotion ? undefined : { opacity: 0, height: 0 }}
                          className="auth-other-interest"
                        >
                          <Field
                            label="Your field of interest"
                            htmlFor="other-interest"
                            hint="Use a concise scientific specialty, for example Wildlife epidemiology."
                          >
                            <Input
                              id="other-interest"
                              value={form.otherInterest}
                              onChange={(event) => update("otherInterest", event.target.value)}
                              placeholder="e.g. Wildlife epidemiology"
                              maxLength={100}
                              autoFocus
                            />
                          </Field>
                        </motion.div>
                      )}
                    </AnimatePresence>
                    <div className="auth-profile-country">
                      <Field
                        label="Country code"
                        htmlFor="country"
                        hint="Two-letter ISO country code, e.g. TN for Tunisia, US for United States, FR for France."
                      >
                        <Input
                          id="country"
                          value={form.country}
                          onChange={(e) =>
                            update("country", e.target.value.toUpperCase().slice(0, 2))
                          }
                          placeholder="TN"
                          maxLength={2}
                          required
                        />
                      </Field>
                    </div>
                    <Field
                      label="How will you use Umbrella?"
                      htmlFor="motivation"
                      hint="A couple of sentences (20+ characters) about your research goals — this helps administrators review your application faster."
                    >
                      <Textarea
                        id="motivation"
                        rows={4}
                        minLength={20}
                        maxLength={4000}
                        value={form.motivation}
                        onChange={(e) => update("motivation", e.target.value)}
                        placeholder="Tell us briefly about your research goals..."
                        required
                      />
                    </Field>
                    <div className="auth-requirement-checks" aria-live="polite">
                      <span className={roleValid ? "is-valid" : undefined}>
                        <Check aria-hidden="true" />
                        Select the option that best describes you
                      </span>
                      <span className={countryValid ? "is-valid" : undefined}>
                        <Check aria-hidden="true" />
                        Two-letter country code ({form.country.trim().length}/2)
                      </span>
                      <span className={motivationValid ? "is-valid" : undefined}>
                        <Check aria-hidden="true" />
                        Motivation ({Math.min(form.motivation.trim().length, 20)}/20 characters
                        minimum)
                      </span>
                    </div>
                  </div>
                  <StepActions>
                    <Button type="button" variant="ghost" onClick={() => setStep(1)}>
                      <ArrowLeft /> Back
                    </Button>
                    <Button
                      type="submit"
                      size="lg"
                      className="auth-primary-button"
                      disabled={!profileValid}
                    >
                      Continue <ArrowRight />
                    </Button>
                  </StepActions>
                </form>
              )}

              {step === 3 && (
                <form
                  onSubmit={(event) => {
                    event.preventDefault();
                    if (form.institution.trim().length >= 2) void submitApplication();
                  }}
                >
                  <StepHeading eyebrow="Step 3 of 4" title="Add your organization">
                    Connect your account to a university, laboratory, or research organization.
                  </StepHeading>
                  <div className="mt-8 space-y-6">
                    <div
                      className="auth-choice-control"
                      role="radiogroup"
                      aria-label="Organization action"
                    >
                      {(["join", "create"] as const).map((mode) => (
                        <button
                          key={mode}
                          type="button"
                          role="radio"
                          aria-checked={form.organizationMode === mode}
                          className={cn(form.organizationMode === mode && "is-selected")}
                          onClick={() => update("organizationMode", mode)}
                        >
                          {mode === "join"
                            ? "Join an existing organization"
                            : "Create a new organization"}
                        </button>
                      ))}
                    </div>
                    <div className="auth-form-grid">
                      <Field
                        label="Organization name"
                        htmlFor="institution"
                        hint="The university, lab, company, or institute you're affiliated with."
                      >
                        <Input
                          id="institution"
                          value={form.institution}
                          onChange={(e) => update("institution", e.target.value)}
                          placeholder="University or institute"
                          required
                        />
                      </Field>
                      <Field
                        label="Department or lab"
                        htmlFor="department"
                        hint="Optional — your specific team, lab, or research group."
                      >
                        <Input
                          id="department"
                          value={form.department}
                          onChange={(e) => update("department", e.target.value)}
                          placeholder="Comparative Genomics Lab"
                        />
                      </Field>
                    </div>
                    <Field
                      label="ORCID (optional)"
                      htmlFor="orcid"
                      hint="Optional — your 16-digit ORCID researcher identifier, formatted like 0000-0000-0000-0000."
                    >
                      <Input
                        id="orcid"
                        value={form.orcid}
                        onChange={(e) => update("orcid", e.target.value)}
                        placeholder="0000-0000-0000-0000"
                      />
                    </Field>
                    {error && (
                      <p role="alert" className="auth-error">
                        {error}
                      </p>
                    )}
                  </div>
                  <StepActions>
                    <Button type="button" variant="ghost" onClick={() => setStep(2)}>
                      <ArrowLeft /> Back
                    </Button>
                    <div className="flex flex-wrap items-center justify-end gap-3">
                      <Button
                        type="button"
                        variant="link"
                        disabled={submitting}
                        onClick={() => void submitApplication(true)}
                      >
                        Skip this step
                      </Button>
                      <Button
                        type="submit"
                        size="lg"
                        className="auth-primary-button"
                        disabled={submitting || form.institution.trim().length < 2}
                      >
                        {submitting ? (
                          <Loader2 className="animate-spin" />
                        ) : (
                          <>
                            Continue <ArrowRight />
                          </>
                        )}
                      </Button>
                    </div>
                  </StepActions>
                </form>
              )}

              {step === 4 && (
                <div className="auth-verify-panel">
                  <div className="auth-verify-icon">
                    <MailCheck />
                  </div>
                  <StepHeading eyebrow="Step 4 of 4" title="One last step!">
                    We sent a verification link to <a href={`mailto:${form.email}`}>{form.email}</a>
                    .
                  </StepHeading>
                  <p className="mx-auto mt-5 max-w-lg text-center text-sm leading-6 text-muted-foreground">
                    Open the link within 30 minutes. After verification, an Umbrella administrator
                    will review your application.
                  </p>
                  {notice && <p className="mt-4 text-center text-sm text-primary">{notice}</p>}
                  {error && (
                    <p role="alert" className="auth-error mx-auto mt-4 max-w-md">
                      {error}
                    </p>
                  )}
                  <div className="mt-8 flex flex-wrap justify-center gap-3">
                    <Button
                      size="lg"
                      className="auth-primary-button"
                      disabled={resending}
                      onClick={async () => {
                        setResending(true);
                        setError("");
                        setNotice("");
                        try {
                          await apiRequest("/api/v1/auth/email-verification/resend", {
                            method: "POST",
                            body: JSON.stringify({ email: form.email }),
                          });
                          setNotice("Verification email sent.");
                        } catch (value) {
                          setError(
                            value instanceof Error ? value.message : "Email could not be resent.",
                          );
                        } finally {
                          setResending(false);
                        }
                      }}
                    >
                      {resending ? <Loader2 className="animate-spin" /> : <RotateCw />} Resend
                      verification email
                    </Button>
                    <Button variant="outline" size="lg" onClick={() => setStep(1)}>
                      Change email address
                    </Button>
                  </div>
                  <div className="auth-mini-trust">
                    <span>
                      <ShieldCheck /> Secure
                    </span>
                    <span>
                      <Sparkles /> Fast
                    </span>
                    <span>
                      <CheckCircle2 /> Easy
                    </span>
                  </div>
                  <p className="mt-8 text-center text-sm text-muted-foreground">
                    Already verified?{" "}
                    <Link to="/signin" className="font-semibold text-primary">
                      Sign in
                    </Link>
                  </p>
                </div>
              )}
            </motion.div>
          </AnimatePresence>
        </div>
        <AuthTrustNote />
      </main>
      <footer className="auth-footer">© 2026 Umbrella Animal BioHub. All rights reserved.</footer>
    </div>
  );
}

function RegistrationStepper({ current }: { current: number }) {
  return (
    <nav className="auth-stepper" aria-label="Registration progress">
      {STEPS.map((label, index) => {
        const number = index + 1;
        return (
          <div
            key={label}
            className={cn(
              "auth-step",
              current === number && "is-current",
              current > number && "is-complete",
            )}
          >
            <span>{current > number ? <Check /> : number}</span>
            <strong>{label}</strong>
          </div>
        );
      })}
    </nav>
  );
}

function StepHeading({
  eyebrow,
  title,
  children,
}: {
  eyebrow: string;
  title: string;
  children: ReactNode;
}) {
  return (
    <div className="auth-step-heading">
      <span>{eyebrow}</span>
      <h1>{title}</h1>
      <p>{children}</p>
    </div>
  );
}

function Field({
  label,
  htmlFor,
  hint,
  children,
}: {
  label: string;
  htmlFor: string;
  hint?: string;
  children: ReactNode;
}) {
  return (
    <div className="space-y-2">
      <div className="flex items-center gap-1.5">
        <Label htmlFor={htmlFor}>{label}</Label>
        {hint && <FieldHint label={hint} />}
      </div>
      {children}
    </div>
  );
}

function StepActions({ children }: { children: ReactNode }) {
  return <div className="auth-step-actions">{children}</div>;
}
