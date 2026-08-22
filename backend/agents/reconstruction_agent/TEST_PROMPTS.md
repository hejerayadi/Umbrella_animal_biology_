# Frontend test prompts — Reconstruction Agent

Paste these into the Umbrella chat. Every prompt was checked against the real
planner (it routes to Reconstruction), the real extractor (the sequence or
accession is picked up), and the agent's own admission rules (the gaps are
attempted, not skipped). Outcomes below are measured against the live agent, not
predicted.

**All nine agents must be running** (`python -m backend.run_agents`). The
orchestrator reaches this one over HTTP on port 8006; if it is not up you get
"agent unreachable", which is a transport failure rather than a routing one.

---

## A. Start here — a gap that reconstructs correctly

```
Fill in the missing bases in this human mitochondrial sequence: ATGTTCGCCGACCGTTGACTATTCTCTACAAACCACAAAGACATTGGAACACTATACCTATTATTCGGCGCATGAGCTGGAGTCCTAGGCACAGCTCTAAGCCTCCTTATTCGAGCCGAGCTGGGCCAGCCAGGCAACCTTCTAGGTAACGACCACATCTACAACGTTATCGTCACAGCCCATGCATTTGNNNNNNNNNNNNNNNNNNNNNTACCCATCATAATCGGAGGCTTTGGCAACTGACTAGTTCCCCTAATAATCGGTGCCCCCGATATGGCGTTTCCCCGCATAAACAACATAAGCTTCTGACTCTTACCTCCCTCTCTCCTACTCCTGCTCGCATCTGCTATAGTGGAGGCCGGAGCAGGAACAGGTTGAACAGTCTACCCTCCCTTAGCAG
```

A real human mtDNA COI region (`NC_012920.1:5904-6303`) with 21 bases masked.
**The correct fill is `TAATAATCTTCTTCATAGTAA`.**

Measured: the agent returns exactly that sequence, backed by references at 0.997
and 1.000 identity. This is the prompt to use when you want to see the agent
actually work.

One caveat you will see: it is labelled `low_confidence`, scoring **0.1482**
against a reporting threshold of **0.15**. The bases are right; the score lands
just under the bar. See *Confidence calibration* at the bottom.

## B. Real woolly mammoth — nothing to paste

```
Reconstruct the unresolved regions of PZ244058.1
```

`PZ244058.1` is a real *Mammuthus primigenius* mitochondrial genome from NCBI —
ancient DNA, 16,771 bases, 99.80% complete, with **two genuine sequencing gaps**:

| Gap | Position | Length | Flanks |
| --- | --- | --- | --- |
| gap_1 | 16292–16294 | 2 b | 500 / 80 |
| gap_2 | 16374–16405 | 31 b | 80 / 366 |

Both clear the 50-base flank minimum, so both are attempted, and two gaps costs
two BLAST calls out of a budget of four.

Measured outcome: this one still comes back **unresolved** (`abstained`). Both
gaps sit in the mitochondrial control region, which is repeat-rich, and gap_2 has
only 80 bases of left flank to anchor a search. That is a genuinely hard case
rather than a plumbing failure — it is a good test of the abstain path, but use
prompt A if you want to see a successful reconstruction.

Variations that also route correctly:

```
The mammoth mitogenome PZ244058.1 has gaps. Fill them in.
```

```
Fill in the missing bases in accession PZ244058.1
```

## C. More gaps — pushes the budget and the slicing

```
Reconstruct the unresolved regions of PZ244055.1
```

Same species, **6 gaps** (longest 68 b) — more than the BLAST budget allows, so
this exercises CONTINUE/resume and ends with partial results, which is correct
behaviour rather than a failure. For a deliberately hopeless case, `PZ244052.1`
has 57 gaps.

## D. Two gaps in one pasted sequence

Truths: gap_1 = `CTGGGCCAGCCAGGCAACCT`, gap_2 = `AGCTTCTGACTCTTACCT`.

```
This assembly has two gaps marked with N. Reconstruct the missing regions: ATGTTCGCCGACCGTTGACTATTCTCTACAAACCACAAAGACATTGGAACACTATACCTATTATTCGGCGCATGAGCTGGAGTCCTAGGCACAGCTCTAAGCCTCCTTATTCGAGCCGAGNNNNNNNNNNNNNNNNNNNNTCTAGGTAACGACCACATCTACAACGTTATCGTCACAGCCCATGCATTTGTAATAATCTTCTTCATAGTAATACCCATCATAATCGGAGGCTTTGGCAACTGACTAGTTCCCCTAATAATCGGTGCCCCCGATATGGCGTTTCCCCGCATAAACAACATANNNNNNNNNNNNNNNNNNCCCTCTCTCCTACTCCTGCTCGCATCTGCTATAGTGGAGGCCGGAGCAGGAACAGGTTGAACAGTCTACCCTCCCTTAGCAG
```

## E. FASTA format — tests the wrapped-record path

Paste the whole block including the `>` line.

```
Please reconstruct this fragment:
>mtDNA_COI_partial
ATGTTCGCCGACCGTTGACTATTCTCTACAAACCACAAAGACATTGGAACACTATACCTATTATTCGGCG
CATGAGCTGGAGTCCTAGGCACAGCTCTAAGCCTCCTTATTCGAGCCGAGCTGGGCCAGCCAGGCAACCT
TCTAGGTAACGACCACATCTACAACGTTATCGTCACAGCCCATGCATTTGNNNNNNNNNNNNNNNNNNNN
NTACCCATCATAATCGGAGGCTTTGGCAACTGACTAGTTCCCCTAATAATCGGTGCCCCCGATATGGCGT
TTCCCCGCATAAACAACATAAGCTTCTGACTCTTACCTCCCTCTCTCCTACTCCTGCTCGCATCTGCTAT
AGTGGAGGCCGGAGCAGGAACAGGTTGAACAGTCTACCCTCCCTTAGCAG
```

## F. Nothing to work on — should ask, not crash

```
Reconstruct the woolly mammoth genome
```

Expect a reply asking for a sequence or an accession, not a raw error. This is a
dead end by design: no other agent holds the caller's own incomplete assembly,
and choosing an arbitrary published record would be inventing the target rather
than repairing one.

## G. Already complete — the "no gaps" answer

```
Reconstruct the unresolved regions of NC_012920.1
```

The human reference mitogenome has no `N` runs at all. Expect a completed answer
saying there is nothing to reconstruct — a real result, not a failure.

## H. Control — must NOT reach this agent

```
What is the genome size of the Arctic fox?
```

Verified to route to **Genome**. If it lands on Reconstruction, `card.json` is
over-claiming and the description needs tightening.

---

## What a successful run looks like

In the agent's own log, for prompt A:

- `blast_hits_carrying_gap` well above zero — the count that predicts whether a
  fill is possible at all, as opposed to the raw hit count, which can be 50 and
  still yield nothing
- `references_sent_to_mafft` with `carrying_gap` alongside it, showing only
  gap-carrying references reaching the aligner
- the gap returned with `reconstructed_sequence` set and evidence attached

## Known problem: alignment sometimes times out

Tested through the whole system, the same prompt can go two ways, because the
agent's planner is an LLM and picks its own first move:

- picks `blast_search` first -> short reference fragments -> alignment finishes
  in about 5 seconds -> **correct fill**
- picks `ncbi_search` first -> whole genomes, ~16,000 letters each -> alignment
  runs past the 75-second slice budget and is killed -> **no result**, with
  `mafft_align: aborted ... ran out of wall clock` in the warnings

So a run can fail for timing reasons even though the evidence was fine. If you
see "ran out of wall clock" in the answer, that is this problem and not a bad
sequence. Re-running often picks the other path.

## Confidence calibration — read before judging a result

Prompt A returns the **exactly correct** 21 bases and still reports
`low_confidence`, because it scores 0.1482 against a 0.15 threshold.

The suppressor is the ambiguity penalty. Real cross-species references disagree
at some columns — three carried the exact human sequence, others carried
`TAATGATCTTCTTTATAGTTA` — so the per-column margin is modest, support lands near
0.1, and `ConfidencePolicy._decisiveness_factor` multiplies the score by 0.25.
Gap length and reference depth are not the cause; both were effectively 1.0.

That threshold was derived by `scripts/tune_settings.py` from **synthetic**
sequences with controlled divergence. Real homologues diverge less tidily, so the
score lands right on the boundary. Re-deriving it against real evidence is a
calibration exercise in its own right; lowering it to make a demo pass would be
the tuning-to-the-demo that the policy's own comments warn against.
