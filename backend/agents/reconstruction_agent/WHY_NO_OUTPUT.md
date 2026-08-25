# Why the Reconstruction Agent produced no output

*Diagnosed against the live services on 2026-08-25. The earlier version of this
document blamed NUMTs and e-value ranking. **That was wrong**, and the section
at the bottom records why, so nobody re-derives it.*

## The real problem

**The agent searched a database that could not contain the answer.**

`BlastSearchInput.database` defaulted to `em_std_vrt`, and
`_RELAXED_DATABASE` widened to `em_vrt`. EMBL/ENA taxonomic divisions are
**mutually exclusive**: `vrt` is *"Other Vertebrates"* — the vertebrates that
are **not** mammal, human, mouse or rodent. Those live in `mam`, `hum`, `mus`
and `rod`.

So for a polar bear — or any mammal, which is most of what this agent is asked
about — the correct homologues were not in the searched database at all. No
e-value, no retry, no ranking change could ever have found them. The relaxed
retry widened `vrt` to `vrt`, so it could not help either.

## The evidence

A 45-base hole punched at position 6000 of the polar bear mitochondrion
(`NC_003428.1`), query = the two 500-base flanks joined, junction at 500 — the
exact parameters the agent sends:

| database | latency | hits | carrying the gap | top hit | fill vs truth |
|---|---|---|---|---|---|
| `em_std_vrt` *(the old default)* | 165 s | 50 | 7 | **fish** — *Corydoras*, *Puntius*, *Brochis*, ~67 % identity | consensus **33 % correct**, no exact match |
| `em_vrt` *(the old relaxed retry)* | 245 s | 50 | 8 | the same fish | wrong |
| **`em_std_mam`** | **213 s** | 50 | **50** | *Ursus maritimus* `AF303111.1`, **95.7 %**, single HSP q1–1000 | **exact — all 45 bases** |
| `em_mam` | 309 s | 50 | 50 | the same | exact |
| `em_std` *(all divisions)* | **>813 s, never finished** | — | — | — | unusable |
| `em_all` | 749 s | 50 | 50 | correct, but past the 600 s slice | too slow |

Truth: `TTTGAAAGCATAAAAAAAATAATCTTCTTGCCCCCTCTAATCGTA`.
From `em_std_mam`, 30 of 50 hits agreed on exactly that.

**The database was the first and largest break, but not the only one.** With
`em_std_mam` the search returns fifty references all carrying the answer - and
the run still produced nothing, because a second defect in the MAFFT mapper was
hiding behind the first (item 4 below). Both had to be fixed before a single
base came back. The consensus, the reasoner and the critic needed no changes.

## What else was wrong

1. **`hit_os` is the literal string `"NA"`** on all 200 hits across four real
   searches. `Reference.organism` was read from it, so no BLAST-derived
   reference ever had an organism: `ReferenceRanker`'s relatedness weight (0.3
   of the score) silently multiplied zero, `_with_organism_affinity` was inert,
   and `evolutionary_context` spent a real NCBI lookup per run resolving the
   string `"NA"`. The organism is in `hit_desc` throughout.
2. **`gap_bases` was dropped by two field-by-field `Reference` rebuilds**
   (`_with_residues`, `apply_relatedness`), resetting `carries_gap` to False —
   and `_with_residues` fires on exactly the multi-HSP bracketing carriers,
   because the mapper clears their residues so they can be fetched.
3. **`accumulate_references._prefer` discarded information.** It picked a winner
   on `has_sequence` then `quality`, so a relatedness-only rewrite — which
   changes neither — never survived the merge at all.
4. **The MAFFT mapper demanded an exact indel placement.** This one hid
   *behind* the database bug and only surfaced once the database was fixed.
   `_locate_gap_columns` walked to the flank junction and required the inserted
   run to begin at exactly that column. When the last base of the left flank
   equals the last base of the missing segment, two placements of the indel
   describe the identical sequence and an aligner may pick either - and MAFFT
   picked the earlier one. Measured: the polar bear left flank ends in `A` and
   the true fill `TTTG...CGTA` also ends in `A`, so the 45 columns landed at
   column 499 rather than 500. The mapper found no run at all and the run
   reported *"no reference sequence aligned across gap_1"* while eight
   references sat in that alignment carrying every base of the answer.

   The fix tolerates the slide - the same allowance `tools/blast/mapper` already
   made for the same reason - and then **re-anchors the fill**, because a fill
   read from columns placed one base early is correct *there* and not between
   the flanks the caller holds: the columns yielded `ATTTG...CGT` where the
   removed bases were `TTTG...CGTA`.

5. **The orchestrator's species was read wrongly.** `_organism_from` checked
   `species` before `species_record.scientific_name`; the real payload carries
   both, so the agent took `"polar bear"` and never saw `"Ursus maritimus"`.
   `organism_affinity` compares binomials, so it failed on every orchestrated
   run.

## What the agent does now

```
Target + context
      ↓
LLM planner  ──  Database Discovery Tool (EBI's live catalogue, 284 nucleotide codes)
      ↓
planner picks candidates, seeded by an NCBI-taxonomy prior
      ↓
BLAST in parallel  (asyncio.gather — wall clock is max(), not the sum)
      ↓
measure blast_hits_carrying_gap per database
      ↓
keep the winner, reuse it for every later gap
      ↓
MAFFT → candidates → Evo2 / reasoner → critic
```

No database is named anywhere in the code. Codes are composed from the target's
division and **validated against EBI's catalogue before submission** — which is
the `em_rel_vrt` class of bug (an invented code, HTTP 400 on every retry) made
structurally impossible rather than fixed once.

## Scale, on a real scaffold

`NW_007907101` — the polar bear scaffold the Genome Agent hands over — has **34
N-runs of 10+ bases in its first megabase**, so roughly **500** across its
15.9 Mb, against a budget of 8 BLAST calls. Attempting them all does not give a
worse answer, it gives none: the planner emits a step per gap and the run spends
its slices refusing them for budget. Gaps are now ranked (both flanks usable
first, then shortest) and the run commits to `RECONSTRUCTION_MAX_GAPS_PER_RUN`
of them, reporting the rest as not attempted with the reason.

## Latency, measured

| | measured | fits a 480 s slice |
|---|---|---|
| `em_std_mam`, mitochondrial query | 213 s | yes |
| `em_mam`, mitochondrial query | 309 s | yes |
| `em_mam`, **nuclear scaffold** query | **546 s**, high variance | no — resumes across slices |
| `em_std` / `em_all` | >813 s / 749 s | no — marked `too_slow`, never proposed |

A nuclear scaffold search does not fit one slice. That is what the parked-job
resume path is for: the job id is written onto the payload before the poll, the
payload is parked in `pending_jobs`, and the next slice collects the same job
instead of paying for it twice.

## Superseded — do not re-chase

The previous diagnosis said the hits were **NUMTs** (nuclear copies of
mitochondrial DNA) and prescribed *"rank on span, not just e-value"*, calling it
*"a scientific tuning problem, not a broken pipeline."* All of that was wrong:

- The `OZ…` contigs it pointed at were not NUMTs outranking real mitochondrial
  records. The real mitochondrial records were **in a different division** and
  were never candidates at any rank.
- Ranking on span would not have helped. `em_std_vrt` contains 7 gap-crossing
  hits for this query and all 7 are fish; ranking them better still yields a
  33 %-correct fill.
- It was a broken pipeline, and a one-line default was the break.

One earlier correction still stands: `execute_tools` dispatches a round with
`asyncio.gather`, so searches in one round run **concurrently**. That is what
makes probing several databases at once cost the slowest search rather than
their sum.

Also disproven while investigating: low-complexity flanks were suspected of
causing the nuclear timeouts. They do not explain it — on that scaffold a flank
whose search never finished scores 0.85 linguistic complexity against 0.87 for
one that finished normally. A DUST filter is now sent because nothing was being
sent before, not because it fixes a timeout.

## How to check any of this

```bash
cd backend/agents/reconstruction_agent

# 453 unit/integration tests, and the 32 agentic scenarios.
./.venv/Scripts/python.exe -m pytest tests -q
./.venv/Scripts/python.exe scripts/qa_audit.py     # flaky by design: 29-31/32

# Graph works with no network at all.
./.venv/Scripts/python.exe scripts/smoke_test.py

# Every credential and service.
./.venv/Scripts/python.exe scripts/test_external_services.py

# The one that matters: mask known bases, reconstruct, score the result
# against what was removed - through the real services.
./.venv/Scripts/python.exe scripts/ground_truth.py
./.venv/Scripts/python.exe scripts/ground_truth.py --control em_std_vrt
```

`qa_audit.py` is **timing-flaky by construction** — its continuation scenarios
run with `yield_after_seconds=0.01`, so whether a slice gets cut before or after
a tool call is a race. Five consecutive runs scored 29, 30, 30, 30 and 31 of 32
on identical code. Only *REVISE → re-plan with critique attached* fails every
time, and it fails on `main` too.

Note what the mocked suites cannot tell you: **they all fake the network edge**,
so BLAST returns instantly and always with usable hits. The database bug and the
timing bug both lived entirely in the gap between those fakes and the real
service. `ground_truth.py` is the only test that closes it.
