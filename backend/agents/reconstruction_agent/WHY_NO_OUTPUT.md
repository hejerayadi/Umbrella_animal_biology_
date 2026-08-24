# Why the Reconstruction Agent produces no output

*Measured against the real services on 2026-08-24.*

## The real problem

**BLAST never returns a reference that reaches across the gap, so there is
nothing to read a fill out of.**

Everything else works. Every credential is valid, every service answers, the
graph runs its full six-iteration loop, and the agent returns a well-formed
result on time. That result is simply empty:

```
gap_1: unresolved (confidence 0.0)
  why: No reference sequence aligned across gap_1, so no reconstruction
       could be supported by evidence.
```

That sentence is **true**. It is not a crash dressed up as an answer — the
agent really was handed no usable evidence. The failure is upstream of the
reasoning, in what the homology search brings back.

## The evidence

Real test: a 2 kb region of the polar bear mitochondrion (`NC_003428.1`) with a
known 45-base hole punched in it, so the answer could be scored. The query sent
to BLAST is the two flanks joined — 500 bases each — so the fill has to cross
the junction at position **500**. Here is what came back:

```
OZ078384.2 | 17 HSP(s)
   q 1-368  h 101407-100976    ident=82.4
   q 1-391  h 101431-100983    ident=80.7
OZ110968.1 | 14 HSP(s)
   q 1-391  h 3881886-3882352  ident=79.7
OZ412278.1 |  6 HSP(s)
   q 1-374  h 29403252-29402879 ident=81.9
```

Two things are wrong here, and both are about *which sequences BLAST found*:

1. **Every HSP stops around query position 368–391.** The junction is at 500.
   Not one hit even reaches the gap, let alone carries bases across it.
2. **These are not mitochondrial sequences.** They are 78–83% identity matches
   against multi-megabase nuclear contigs (`OZ…`, at coordinates like
   3,881,886). Those are NUMTs — old nuclear copies of mitochondrial DNA.
   The actual mitochondrial homologues, which would be near-identical and would
   span the region easily, never appear in the top 50 at all.

So the search is returning distant nuclear pseudo-copies instead of close real
relatives, and even those peter out before the part that matters.

## Why that ends the run

The chain is short and each link is behaving correctly:

1. No hit carries bases across the gap → `carries_gap` is false for all of them.
2. `ToolSelector._mafft` finds no carriers, and aligns the non-carriers instead
   (logging *"Every reference stops short of the gap; alignment cannot fill it"*).
3. MAFFT opens no columns at the junction, because no row has anything to put
   there → `spans_gap=False`.
4. The reasoner has no candidate, the critic abstains, the gap is reported
   `unresolved` at confidence 0.

Nothing here is a bug to fix. Step 4 is the honest consequence of step 1.

## Why the database is the suspect

The first pass searches `em_std_vrt` — ENA's *standard* vertebrate division —
ranked by e-value. A long, repetitive nuclear contig that matches a flank in
seventeen fragments can outrank a short, near-identical mitochondrial record,
and if the right record is not in that division at all, no ranking helps.

This is the open question, and it is a **scientific tuning problem, not a broken
pipeline**: which ENA division suits which kind of target, and how to rank a
short near-perfect match above a long approximate one.

## How this looks in the two ways the agent is called

**Called on its own.** It repairs *one named sequence* and refuses to pick a
target for itself — ask it to "reconstruct the mammoth genome" with no sequence
and no accession and it correctly refuses, which reads like a crash from
outside. Give it a real sequence and it runs properly, then hits the wall above.
Note that `POST /api/v1/reconstructions` also has no trace id, so it gets one
slice and cannot resume; `POST /execute` with a **stable `X-Trace-Id`** across
calls is the one that can.

**Called from the Genome Agent.** The hand-off works well: the Genome Agent sees
a Scaffold/Contig assembly, resolves the species' largest RefSeq record under
2 Mb, and publishes it as `sequence_accession`. That scaffold carries ~17 gaps.
Each one meets the same evidence wall, so the agent returns `COMPLETED` with
everything unresolved, the Genome Agent resumes and answers from the draft
assembly, and the user gets a fluent paragraph in which nothing was
reconstructed — with no error logged anywhere.

## Already fixed — don't re-chase these

| Was | Now |
| --- | --- |
| `AGENT_YIELD_AFTER_SECONDS=75`, sized for a 120 s orchestrator timeout raised to 600 s. One BLAST job takes **193 s**, so no run ever finished a single search. | **480 s.** Six full iterations now complete in one slice. |
| `_RELAXED_DATABASE = "em_rel_vrt"` — not one of EBI's 409 valid codes, so every relaxed retry died on **HTTP 400**. | `em_vrt`, valid and genuinely broader. |
| A parked BLAST job resumed only if a freshly planned payload *hashed* the same — hostage to the LLM re-planning byte for byte. | The whole payload is stored and re-issued verbatim. |
| "Ran out of time" and "no evidence found" produced the same sentence. | Reported separately; a timed-out region says so. |
| `ScriptedBlast` never set `job_id`, so the QA scenarios guarding resumption could not pass. | Models submit-then-poll; audit 29/32 → 31/32. |

Also corrected: an earlier draft claimed N BLAST searches cost N × 193 s.
`execute_tools` dispatches a round with `asyncio.gather`, so they run
**concurrently** — which is why the call budgets were raised (4 → 8 BLAST,
12 → 24 tool calls) rather than trimmed.

## What would fix the real problem

1. **Pick the database by target type.** A mitochondrial query and a nuclear
   scaffold gap need different ENA divisions. One default cannot serve both.
2. **Rank on span, not just e-value.** The only hits worth aligning are the ones
   that reach the junction. That is measurable before MAFFT is ever called —
   `blast_hits_carrying_gap` is already in the diagnostics and is currently 0
   on every run.
3. **Re-test with ground truth.** Mask a known stretch, reconstruct it, score
   the proposal against the bases removed. That is the only way to tell a
   working evidence path from a plausible-looking one.

## How to check any of this

```bash
cd backend/agents/reconstruction_agent

# Graph works with no network at all. Passes; always has.
./.venv/Scripts/python.exe scripts/smoke_test.py

# Every credential and service. All OK - and prints the 193 s BLAST latency
# next to the word "OK", which is how it hid for so long. Takes ~4 minutes.
./.venv/Scripts/python.exe scripts/test_external_services.py

# 364 unit/integration tests, and the 32 agentic scenarios.
./.venv/Scripts/python.exe -m pytest tests -q
./.venv/Scripts/python.exe scripts/qa_audit.py
```

Note what the test suites cannot tell you: **every one of them mocks the network
edge**, so BLAST returns instantly and always with usable hits. Both the timing
bug and the evidence problem live entirely in the gap between those fakes and
the real service.
