import { AnimatePresence, motion, useReducedMotion } from "framer-motion";
import {
  Briefcase,
  Building2,
  Check,
  Clock3,
  Fingerprint,
  Globe,
  KeyRound,
  Loader2,
  Mail,
  MailPlus,
  Send,
  ShieldCheck,
  Sparkles,
  UserRound,
  X,
} from "lucide-react";
import { useState, type FormEvent } from "react";

import { Avatar, AvatarFallback } from "@/components/ui/avatar";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { FieldHint } from "@/components/umbrella/field-hint";
import { AI_CAPABILITIES } from "@/lib/umbrella-types";
import { cn } from "@/lib/utils";

const EASE = [0.22, 1, 0.36, 1] as const;

/** Mirrors `InvitationRequest.orcid` in backend/app/schemas.py. */
const ORCID_PATTERN = /^\d{4}-\d{4}-\d{4}-\d{3}[\dX]$/;
const EMAIL_PATTERN = /^\S+@\S+\.\S+$/;

interface InviteForm {
  email: string;
  full_name: string;
  institution: string;
  professional_title: string;
  country: string;
  orcid: string;
  specialties: string[];
}

const EMPTY_FORM: InviteForm = {
  email: "",
  full_name: "",
  institution: "",
  professional_title: "",
  country: "",
  orcid: "",
  specialties: [],
};

const FIELDS = [
  {
    name: "email",
    label: "Email address",
    icon: Mail,
    type: "email",
    placeholder: "you@lab.edu",
    hint: "The collaborator's work email. The invitation and temporary password are delivered here.",
    span: "full",
  },
  {
    name: "full_name",
    label: "Full name",
    icon: UserRound,
    type: "text",
    placeholder: "Dr. Alex Morgan",
    hint: "Full name as it should appear on their Umbrella profile. At least 2 characters.",
    span: "half",
  },
  {
    name: "professional_title",
    label: "Professional title",
    icon: Briefcase,
    type: "text",
    placeholder: "Research Scientist",
    hint: "Their role, e.g. Research Scientist, PhD Candidate, Lab Director.",
    span: "half",
  },
  {
    name: "institution",
    label: "Institution",
    icon: Building2,
    type: "text",
    placeholder: "University or institute",
    hint: "The university, laboratory, or organization they belong to.",
    span: "half",
  },
  {
    name: "country",
    label: "Country code",
    icon: Globe,
    type: "text",
    placeholder: "TN",
    hint: "Two-letter ISO country code, e.g. TN, US, FR.",
    span: "half",
  },
  {
    name: "orcid",
    label: "ORCID",
    icon: Fingerprint,
    type: "text",
    placeholder: "0000-0000-0000-0000",
    hint: "Optional. Must be 16 digits in 0000-0000-0000-0000 form (a trailing X is allowed).",
    span: "full",
  },
] as const satisfies ReadonlyArray<{
  name: keyof Omit<InviteForm, "specialties">;
  label: string;
  icon: typeof Mail;
  type: string;
  placeholder: string;
  hint: string;
  span: "full" | "half";
}>;

const TIMELINE = [
  {
    icon: MailPlus,
    title: "Invitation delivered",
    body: "An email with a temporary password is sent immediately.",
  },
  {
    icon: Clock3,
    title: "72-hour window",
    body: "The temporary password expires if it is not used in time.",
  },
  {
    icon: KeyRound,
    title: "Permanent password",
    body: "The biologist must set their own password at first login.",
  },
];

function initialsOf(name: string) {
  return name
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .slice(0, 2)
    .map((part) => part[0]?.toUpperCase() ?? "")
    .join("");
}

export function AdminInvitation({
  onSubmit,
}: {
  onSubmit: (payload: Record<string, unknown>) => Promise<void>;
}) {
  const reduceMotion = useReducedMotion();
  const [form, setForm] = useState<InviteForm>(EMPTY_FORM);
  const [message, setMessage] = useState("");
  const [isError, setIsError] = useState(false);
  const [submitting, setSubmitting] = useState(false);

  function update(name: keyof Omit<InviteForm, "specialties">, value: string) {
    setForm((current) => ({ ...current, [name]: value }));
  }

  function toggleSpecialty(value: string) {
    setForm((current) => ({
      ...current,
      specialties: current.specialties.includes(value)
        ? current.specialties.filter((item) => item !== value)
        : [...current.specialties, value],
    }));
  }

  const valid = {
    email: EMAIL_PATTERN.test(form.email.trim()),
    full_name: form.full_name.trim().length >= 2,
    institution: form.institution.trim().length >= 2,
    professional_title: form.professional_title.trim().length >= 2,
    country: form.country.trim().length === 2,
  };
  // ORCID is optional, but a non-empty value must satisfy the server pattern
  // or the API rejects the whole request with a 422.
  const orcidValid = form.orcid.trim() === "" || ORCID_PATTERN.test(form.orcid.trim());
  const orcidInvalid = form.orcid.trim() !== "" && !orcidValid;

  const requiredKeys = Object.keys(valid) as Array<keyof typeof valid>;
  const completedCount = requiredKeys.filter((key) => valid[key]).length;
  const completion = Math.round((completedCount / requiredKeys.length) * 100);
  const canSubmit = completedCount === requiredKeys.length && orcidValid && !submitting;

  async function handleSubmit(event: FormEvent) {
    event.preventDefault();
    if (!canSubmit) return;

    setMessage("");
    setIsError(false);
    setSubmitting(true);
    try {
      await onSubmit({
        email: form.email.trim(),
        full_name: form.full_name.trim(),
        institution: form.institution.trim(),
        professional_title: form.professional_title.trim(),
        country: form.country.trim().toUpperCase(),
        orcid: form.orcid.trim() || undefined,
        specialties: form.specialties,
      });
      setMessage(`Invitation sent to ${form.email.trim()}.`);
      setForm(EMPTY_FORM);
    } catch (value) {
      setIsError(true);
      setMessage(value instanceof Error ? value.message : "Invitation failed.");
    } finally {
      setSubmitting(false);
    }
  }

  const initials = initialsOf(form.full_name);

  return (
    <div className="admin-invite-page">
      <motion.form
        className="admin-invite-form-panel"
        onSubmit={handleSubmit}
        initial={reduceMotion ? false : { opacity: 0, scale: 0.96, y: 12 }}
        animate={{ opacity: 1, scale: 1, y: 0 }}
        transition={{ duration: 0.45, ease: EASE }}
      >
        <div className="admin-panel-heading">
          <div>
            <p className="admin-section-kicker">Secure onboarding</p>
            <h2>Invitation details</h2>
          </div>
          <Badge variant="secondary" className="gap-1.5">
            <ShieldCheck className="size-3.5" />
            Admin only
          </Badge>
        </div>

        <motion.div
          className="admin-invite-fields"
          initial="hidden"
          animate="visible"
          variants={{
            hidden: {},
            visible: { transition: { staggerChildren: 0.05, delayChildren: 0.12 } },
          }}
        >
          {FIELDS.map((field) => {
            const isInvalidOrcid = field.name === "orcid" && orcidInvalid;
            return (
              <motion.div
                key={field.name}
                className={cn("admin-invite-field", field.span === "full" && "is-full")}
                variants={{
                  hidden: reduceMotion ? {} : { opacity: 0, y: 12 },
                  visible: { opacity: 1, y: 0 },
                }}
                transition={{ duration: 0.35, ease: EASE }}
              >
                <div className="admin-invite-label-row">
                  <Label htmlFor={`invite-${field.name}`}>{field.label}</Label>
                  <FieldHint label={field.hint} />
                  {field.name === "orcid" && (
                    <span className="admin-invite-optional">Optional</span>
                  )}
                </div>
                <div className="admin-field-wrap">
                  <span className="admin-field-icon" aria-hidden="true">
                    <field.icon />
                  </span>
                  <Input
                    id={`invite-${field.name}`}
                    type={field.type}
                    value={form[field.name]}
                    onChange={(event) =>
                      update(
                        field.name,
                        field.name === "country"
                          ? event.target.value.toUpperCase().slice(0, 2)
                          : event.target.value,
                      )
                    }
                    placeholder={field.placeholder}
                    required={field.name !== "orcid"}
                    aria-invalid={isInvalidOrcid || undefined}
                    autoComplete={field.name === "email" ? "email" : "off"}
                    className={cn(isInvalidOrcid && "border-destructive")}
                  />
                </div>
                <AnimatePresence>
                  {isInvalidOrcid && (
                    <motion.p
                      className="admin-invite-field-error"
                      initial={reduceMotion ? false : { opacity: 0, height: 0 }}
                      animate={{ opacity: 1, height: "auto" }}
                      exit={reduceMotion ? undefined : { opacity: 0, height: 0 }}
                      transition={{ duration: 0.2, ease: EASE }}
                    >
                      Use the 0000-0000-0000-0000 format.
                    </motion.p>
                  )}
                </AnimatePresence>
              </motion.div>
            );
          })}
        </motion.div>

        <fieldset className="admin-invite-specialties">
          <legend className="admin-invite-label-row">
            <span className="admin-invite-legend-text">Research specialties</span>
            <FieldHint label="Optional. Pre-assigns the specialist domains this biologist works with. They can change these later." />
            <span className="admin-invite-optional">Optional</span>
          </legend>
          <div className="admin-specialty-grid">
            {AI_CAPABILITIES.map((capability) => {
              const selected = form.specialties.includes(capability);
              return (
                <motion.button
                  key={capability}
                  type="button"
                  aria-pressed={selected}
                  onClick={() => toggleSpecialty(capability)}
                  className={cn("admin-specialty-chip", selected && "is-selected")}
                  whileHover={reduceMotion ? undefined : { y: -2 }}
                  whileTap={reduceMotion ? undefined : { scale: 0.97 }}
                  transition={{ duration: 0.15, ease: EASE }}
                >
                  <AnimatePresence initial={false} mode="wait">
                    {selected ? (
                      <motion.span
                        key="on"
                        className="grid place-items-center"
                        initial={reduceMotion ? false : { scale: 0, rotate: -90 }}
                        animate={{ scale: 1, rotate: 0 }}
                        exit={reduceMotion ? undefined : { scale: 0, rotate: 90 }}
                        transition={{ duration: 0.18, ease: EASE }}
                      >
                        <Check />
                      </motion.span>
                    ) : (
                      <motion.span
                        key="off"
                        className="grid place-items-center"
                        initial={reduceMotion ? false : { scale: 0 }}
                        animate={{ scale: 1 }}
                        exit={reduceMotion ? undefined : { scale: 0 }}
                        transition={{ duration: 0.18, ease: EASE }}
                      >
                        <Sparkles />
                      </motion.span>
                    )}
                  </AnimatePresence>
                  {capability}
                </motion.button>
              );
            })}
          </div>
        </fieldset>

        <div className="admin-invite-footer">
          <AnimatePresence mode="wait">
            {message ? (
              <motion.p
                key={message}
                role="status"
                className={cn("admin-invite-message", isError && "is-error")}
                initial={reduceMotion ? false : { opacity: 0, y: -6 }}
                animate={{ opacity: 1, y: 0 }}
                exit={reduceMotion ? undefined : { opacity: 0 }}
                transition={{ duration: 0.22, ease: EASE }}
              >
                {isError ? <X aria-hidden="true" /> : <Check aria-hidden="true" />}
                {message}
              </motion.p>
            ) : (
              <motion.p
                key="hint"
                className="admin-invite-footer-hint"
                initial={reduceMotion ? false : { opacity: 0 }}
                animate={{ opacity: 1 }}
                exit={reduceMotion ? undefined : { opacity: 0 }}
              >
                {canSubmit
                  ? "Ready to send."
                  : `Complete the required fields to continue (${completedCount}/${requiredKeys.length}).`}
              </motion.p>
            )}
          </AnimatePresence>

          <motion.div
            whileHover={reduceMotion || !canSubmit ? undefined : { scale: 1.02 }}
            whileTap={reduceMotion || !canSubmit ? undefined : { scale: 0.98 }}
            transition={{ duration: 0.15, ease: EASE }}
          >
            <Button type="submit" size="lg" disabled={!canSubmit}>
              {submitting ? (
                <Loader2 className="size-4 animate-spin" />
              ) : (
                <Send className="size-4" />
              )}
              Send invitation
            </Button>
          </motion.div>
        </div>
      </motion.form>

      <motion.aside
        className="admin-invite-profile"
        aria-label="Live invitation preview"
        initial={reduceMotion ? false : { opacity: 0, x: 24 }}
        animate={{ opacity: 1, x: 0 }}
        transition={{ duration: 0.5, delay: 0.1, ease: EASE }}
      >
        <div className="admin-invite-profile-card">
          <div className="admin-invite-profile-cover" aria-hidden="true" />

          <div className="admin-invite-profile-identity">
            <Avatar size="lg" className="admin-invite-avatar">
              <AvatarFallback>
                <AnimatePresence mode="wait" initial={false}>
                  <motion.span
                    key={initials || "placeholder"}
                    initial={reduceMotion ? false : { opacity: 0, scale: 0.7 }}
                    animate={{ opacity: 1, scale: 1 }}
                    exit={reduceMotion ? undefined : { opacity: 0, scale: 0.7 }}
                    transition={{ duration: 0.22, ease: EASE }}
                  >
                    {initials || <UserRound className="size-5" />}
                  </motion.span>
                </AnimatePresence>
              </AvatarFallback>
            </Avatar>

            <LiveText
              as="strong"
              className="admin-invite-profile-name"
              value={form.full_name.trim()}
              placeholder="New collaborator"
              reduceMotion={Boolean(reduceMotion)}
            />
            <LiveText
              as="span"
              className="admin-invite-profile-role"
              value={form.professional_title.trim()}
              placeholder="Role pending"
              reduceMotion={Boolean(reduceMotion)}
            />

            <Badge variant={canSubmit ? "default" : "secondary"} className="mt-3 gap-1.5">
              {canSubmit ? <Check className="size-3" /> : <Clock3 className="size-3" />}
              {canSubmit ? "Ready to invite" : "Draft"}
            </Badge>
          </div>

          <dl className="admin-invite-profile-details">
            <ProfileRow
              icon={<Mail />}
              label="Email"
              value={form.email.trim()}
              reduceMotion={Boolean(reduceMotion)}
            />
            <ProfileRow
              icon={<Building2 />}
              label="Institution"
              value={form.institution.trim()}
              reduceMotion={Boolean(reduceMotion)}
            />
            <ProfileRow
              icon={<Globe />}
              label="Country"
              value={form.country.trim()}
              reduceMotion={Boolean(reduceMotion)}
            />
            <ProfileRow
              icon={<Fingerprint />}
              label="ORCID"
              value={orcidValid ? form.orcid.trim() : ""}
              reduceMotion={Boolean(reduceMotion)}
            />
          </dl>

          <div className="admin-invite-profile-specialties">
            <span className="admin-invite-profile-subtitle">Specialties</span>
            {form.specialties.length ? (
              <div className="admin-invite-chip-row">
                <AnimatePresence mode="popLayout">
                  {form.specialties.map((capability) => (
                    <motion.span
                      key={capability}
                      layout={!reduceMotion}
                      className="admin-invite-chip"
                      initial={reduceMotion ? false : { opacity: 0, scale: 0.7 }}
                      animate={{ opacity: 1, scale: 1 }}
                      exit={reduceMotion ? undefined : { opacity: 0, scale: 0.7 }}
                      transition={{ duration: 0.2, ease: EASE }}
                    >
                      {capability}
                    </motion.span>
                  ))}
                </AnimatePresence>
              </div>
            ) : (
              <p className="admin-invite-profile-empty">No specialties selected yet.</p>
            )}
          </div>

          <div className="admin-invite-progress">
            <div className="admin-invite-progress-head">
              <span>Profile completion</span>
              <motion.strong
                key={completion}
                initial={reduceMotion ? false : { opacity: 0, y: -4 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.2 }}
              >
                {completion}%
              </motion.strong>
            </div>
            <div className="admin-invite-progress-track">
              <motion.span
                animate={{ width: `${completion}%` }}
                transition={reduceMotion ? { duration: 0 } : { duration: 0.45, ease: EASE }}
              />
            </div>
          </div>
        </div>

        <ol className="admin-invite-timeline">
          {TIMELINE.map((item) => (
            <li key={item.title}>
              <span>
                <item.icon />
              </span>
              <div>
                <strong>{item.title}</strong>
                <p>{item.body}</p>
              </div>
            </li>
          ))}
        </ol>
      </motion.aside>
    </div>
  );
}

function ProfileRow({
  icon,
  label,
  value,
  reduceMotion,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  reduceMotion: boolean;
}) {
  return (
    <div className={cn("admin-invite-profile-row", value && "is-filled")}>
      <dt>
        <span aria-hidden="true">{icon}</span>
        {label}
      </dt>
      <dd>
        <AnimatePresence mode="wait" initial={false}>
          <motion.span
            key={value || "empty"}
            initial={reduceMotion ? false : { opacity: 0, y: 4 }}
            animate={{ opacity: 1, y: 0 }}
            exit={reduceMotion ? undefined : { opacity: 0, y: -4 }}
            transition={{ duration: 0.2, ease: EASE }}
          >
            {value || "—"}
          </motion.span>
        </AnimatePresence>
      </dd>
    </div>
  );
}

function LiveText({
  as: Tag,
  className,
  value,
  placeholder,
  reduceMotion,
}: {
  as: "strong" | "span";
  className?: string;
  value: string;
  placeholder: string;
  reduceMotion: boolean;
}) {
  return (
    <Tag className={cn(className, !value && "is-placeholder")}>
      <AnimatePresence mode="wait" initial={false}>
        <motion.span
          key={value || "placeholder"}
          initial={reduceMotion ? false : { opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={reduceMotion ? undefined : { opacity: 0, y: -6 }}
          transition={{ duration: 0.22, ease: EASE }}
        >
          {value || placeholder}
        </motion.span>
      </AnimatePresence>
    </Tag>
  );
}
