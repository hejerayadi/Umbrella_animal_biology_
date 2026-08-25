import { useState } from "react";
import { AlertTriangle, Check, Copy, FileText, Quote } from "lucide-react";
import { toast } from "sonner";

import { Button } from "@/components/ui/button";
import { Markdown } from "@/components/umbrella/markdown";
import type { WritingDraftSpec } from "@/lib/umbrella-types";
import { cn } from "@/lib/utils";

/** Roughly what a journal's word limit counts, which is what the user cares about. */
function wordCount(text: string): number {
  return text.trim().split(/\s+/).filter(Boolean).length;
}

function CopyButton({ text, label }: { text: string; label: string }) {
  const [copied, setCopied] = useState(false);

  return (
    <Button
      variant="ghost"
      size="sm"
      className="h-7 shrink-0 gap-1.5 px-2 text-xs"
      onClick={() => {
        // `writeText` rejects on an insecure origin and when the document is
        // not focused. Copying is the main thing this panel is for, so a
        // silent no-op would look like a broken button.
        navigator.clipboard
          .writeText(text)
          .then(() => {
            setCopied(true);
            setTimeout(() => setCopied(false), 1600);
          })
          .catch(() => toast.error("Could not copy — select the text and copy manually."));
      }}
    >
      {copied ? (
        <>
          <Check className="size-3.5" aria-hidden="true" /> Copied
        </>
      ) : (
        <>
          <Copy className="size-3.5" aria-hidden="true" /> {label}
        </>
      )}
    </Button>
  );
}

/**
 * The text the Literature Agent wrote, under an assistant message.
 *
 * Set apart from the answer rather than run into it. The draft is not a
 * finding *about* the question the way a genome size is - it is the thing the
 * user asked to be written, meant to be read as a unit and pasted into a
 * manuscript. Inline it had no boundary: nothing showed where the assistant's
 * framing stopped and the generated abstract began, and the Responder tended
 * to reprint the whole thing under its own summary of it.
 *
 * The draft renders as plain text in a serif face, not as markdown. It is
 * manuscript prose, and the system prompt asks for no headings - so the only
 * thing markdown could do here is misread an author's underscore or asterisk
 * as formatting. The journal list is the opposite case: the agent is asked for
 * a markdown list, so that half does go through `Markdown`.
 */
export function WritingPanel({ spec }: { spec: WritingDraftSpec }) {
  const { draft, recommendedJournals } = spec;
  const hasDraft = Boolean(draft && draft.trim());
  const hasJournals = Boolean(recommendedJournals && recommendedJournals.trim());

  return (
    <section className="mt-4 overflow-hidden rounded-xl border border-border bg-card">
      <header className="flex flex-wrap items-start justify-between gap-2 border-b border-border px-4 py-3">
        <div className="min-w-0">
          <div className="flex items-center gap-2">
            <FileText className="size-4 shrink-0 text-primary" aria-hidden="true" />
            <h3 className="truncate text-sm font-semibold text-card-foreground">
              {hasDraft ? spec.section : "Suggested journals"}
            </h3>
          </div>
          <p className="mt-0.5 pl-6 text-xs text-muted-foreground">
            {hasDraft
              ? `Generated text — ${wordCount(draft ?? "")} words. Review before use.`
              : "Candidate venues for this manuscript."}
          </p>
        </div>

        <span className="shrink-0 rounded-full bg-accent px-2.5 py-1 text-[0.7rem] font-semibold text-accent-foreground">
          Draft
        </span>
      </header>

      {/* The one caveat worth interrupting for. A draft that reads as though it
          is grounded in literature, when the references behind it are stand-ins,
          is the failure this platform is built to avoid - so it is stated above
          the text rather than in a footnote under it. */}
      {spec.referencesArePlaceholder && (
        <p className="flex items-start gap-2 border-b border-border bg-accent/60 px-4 py-2 text-xs text-accent-foreground">
          <AlertTriangle className="mt-0.5 size-3.5 shrink-0" aria-hidden="true" />
          <span>
            <span className="font-semibold">Placeholder references.</span> Literature search is not
            yet connected to PubMed, so nothing cited here is a real publication.
          </span>
        </p>
      )}

      {hasDraft && (
        <div className="max-h-[32rem] overflow-y-auto px-5 py-4">
          <p className="whitespace-pre-wrap font-serif text-[0.95rem] leading-7 text-foreground/90">
            {draft}
          </p>
        </div>
      )}

      {hasJournals && (
        <div className={cn("px-5 py-4", hasDraft && "border-t border-border")}>
          {hasDraft && (
            <h4 className="mb-2 flex items-center gap-2 text-xs font-semibold uppercase tracking-wide text-muted-foreground">
              <Quote className="size-3.5" aria-hidden="true" /> Suggested journals
            </h4>
          )}
          <Markdown content={recommendedJournals ?? ""} />
        </div>
      )}

      {spec.referencesUsed.length > 0 && (
        <details className="border-t border-border px-5 py-2.5">
          <summary className="cursor-pointer text-xs font-medium text-muted-foreground">
            References available to this draft ({spec.referencesUsed.length})
          </summary>
          <ul className="mt-2 space-y-1 text-xs text-muted-foreground">
            {spec.referencesUsed.map((reference, index) => (
              <li key={`${reference}-${index}`} className="wrap-anywhere">
                {reference}
              </li>
            ))}
          </ul>
        </details>
      )}

      <footer className="flex flex-wrap items-center justify-between gap-2 border-t border-border px-4 py-2.5">
        <p className="min-w-0 text-[0.7rem] text-muted-foreground">
          Literature Agent
          {hasDraft && spec.referencesUsed.length === 0 && !spec.referencesArePlaceholder && (
            <> &middot; no literature retrieved, so the text cites nothing</>
          )}
          {hasDraft && spec.styleCorrected && <> &middot; grammar checked</>}
        </p>

        <div className="flex shrink-0 items-center gap-1">
          {hasDraft && <CopyButton text={draft ?? ""} label="Copy text" />}
          {hasJournals && !hasDraft && (
            <CopyButton text={recommendedJournals ?? ""} label="Copy list" />
          )}
        </div>
      </footer>
    </section>
  );
}
