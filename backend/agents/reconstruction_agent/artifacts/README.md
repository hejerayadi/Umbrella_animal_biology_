# Handoff audit evidence — captured 29 August 2026

Raw output from four end-to-end runs against live NCBI, taken while auditing the
Grand Orchestrator → Genome → Gap Finder → Reconstruction handoff. Nothing here
was hand-written: instrumentation wrapped the real functions by monkeypatching,
so the behaviour recorded is the behaviour that shipped.

> **These files record the behaviour *before* the audit's findings were fixed.**
> They are kept as evidence, not as examples. Read them as "what the handoff did
> on 29 August 2026", never as the current contract — six defects they document
> have since been repaired, and the payload shape changed in the process. The
> live contract is `card.json` plus `to_agent_output` in
> `src/reconstruction_agent/api/v1/mappers/reconstruction_mapper.py`.

| File | What it shows |
| --- | --- |
| `reconstruction_handoff_payload.json` | The exact payload crossing the agent boundary: five context keys, ten gaps, real flanking sequence. |
| `polarbear_trace.json` | Stage-by-stage counts — record census, ground-truth N profile, raw gaps, selection policy. |
| `polarbear_reconstruction_response.json` | What Reconstruction returned: per-gap status, evidence, and the top-level best fill. |
| `tiger_trace.json` | Control run — a chromosome-level assembly completes without escalating. |

Subject: *Ursus maritimus*, assembly `GCF_017311325.1`, scaffold
`NW_024426341.1` (1,153,480 bp).

## What has changed since

Reading these against today's code, four differences will stand out. All four
are the fixes, not drift:

- **`reconstruction_sequence` is gone.** It held `"ACCTTAC"` — seven bases — under
  a name that read as the repaired 1.15 Mb scaffold. It is now
  `reconstruction_best_fill`, an object carrying `start`, `end`, `length_bp` and
  `confidence` so the scope cannot be misread.
- **The handoff context is larger.** `gaps_found`, `gaps_over_floor`,
  `gaps_selected` and `selection_policy` now travel with `target_gaps`, so a
  consumer can tell ten-of-thirty from ten-of-ten.
- **Errors are no longer duplicated.** `polarbear_trace.json` shows the same
  annotation error twice; that was a LangGraph reducer being handed an already
  accumulated list, and each node now returns only its own new errors.
- **`tiger_trace.json` no longer represents a skipped case.** Escalation used to
  depend on `assembly_level` alone, so a chromosome-level assembly was never
  offered for reconstruction however many unresolved bases it held. It now also
  escalates on NCBI's own gap count — measured, *Mus musculus* GCF_000001635.27
  is chromosome-level with 73,600,614 unresolved bases.
