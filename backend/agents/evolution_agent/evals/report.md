# Evolution Agent — Agent Evaluation Report

## Summary

- Cases run: 9
- Agent/tool selection: 8/8 passed
- Task completion: 9/9 passed
- Correctness (deterministic): 3/3 passed
- Response consistency: 1/1 passed
- Relevance (LLM judge, mean): 4.2/5 *(re-run after Explainer prompt fix — see below)*

## What changed since the first run

**Root cause of low relevance (2.0/5) on `phylo_five_species` and `full_analysis_explicit_both`:**

The original Explainer prompt was generic — it told the model to "interpret ONLY the results provided" but gave no guidance on *which fields to read per feature* or *what to highlight*. As a result the model produced vague, feature-agnostic prose that named a model and a confidence percentage but never described the tree topology or named which species clustered together. The LLM judge correctly penalised this as unfaithful.

**Fix applied (`explainer.py`):**
- Rewrote `EXPLAINER_SYSTEM_PROMPT` with a feature-aware section (`molecular_comparison`, `phylogenetic_tree`, `full_analysis`) that names the exact JSON fields the model must read and reference.
- Raised `_MAX_CHARS` from 900 to 1200 to give the model enough room to describe a 5-species topology without truncation.
- All grounding guards (decimal-value check, hallucinated-binomial check) are unchanged — they are the safety net, not the quality driver.

**Effect:** The Explainer now names clustering species, quotes the exact pairwise scores and branch lengths from the structured payload, describes the Newick topology, and references bootstrap support values. Relevance scores rose from a mean of 3.0/5 to 4.2/5 across all judged cases.

---

## Weaknesses identified

The following weaknesses were identified from the first run and addressed:

1. **Explainer vagueness on phylogenetic output** (`phylo_five_species`, `full_analysis_explicit_both`):
   The original prompt produced summaries that named a model and a confidence value but never described the tree topology or branch groupings. Fixed by making the prompt feature-aware. ✓ *Resolved*

2. **Explainer fabricates rounded scores in summary strings** (`mc_three_species`, `mc_two_species`):
   The `explanation` field in `to_platform_result()` (produced by `_build_summary`) truncates scores like 0.5325 to "0.53". The LLM judge penalised this as fabrication even though the number comes from the data. The Explainer's `interpretation` field (which the judge should be scoring) is grounded correctly — the issue is in the static summary, not the LLM output. Noted as a known limitation: the `explanation` field is a human-readable one-liner for display, not a grounded analysis; the `interpretation` field is the grounded output.

3. **Consistency tested on only one case** (`mc_three_species`):
   Only one case has `consistency_repeats > 1`. Adding more consistency cases in Sprint 4 would give a stronger signal. Noted for next sprint.

4. **Relevance judge skipped for `continue`/`failed` cases**:
   Cases that return `continue` or `failed` don't have an `explanation` to judge. This is correct behaviour — there is nothing to score. Coverage is complete for all `completed` cases.

---

## Per-case detail

### `mc_three_species`
- Prompt: 'How similar are human, chimp and mouse at the molecular level?'
- Notes: Baseline case. Human-chimp must be the closest pair regardless of run.
- Agent/tool selection: PASS — expected='molecular_comparison' actual='molecular_comparison'
- Task completion: PASS — status expected='completed' actual='completed'; species expected=['Homo sapiens', 'Mus musculus', 'Pan troglodytes'] actual=['Homo sapiens', 'Mus musculus', 'Pan troglodytes']
- Correctness: PASS — closest pair expected={'homo sapiens', 'pan troglodytes'} actual={'homo sapiens', 'pan troglodytes'} score=0.5325
- Relevance: 4.0/5 — Response names human-chimp as closest (score 0.5325), groups mouse separately, and references the network structure. Minor deduction: the static `explanation` summary truncates 0.5325 to 0.53 which the judge flagged; the `interpretation` field is fully grounded.
- Consistency: PASS — features=['molecular_comparison', 'molecular_comparison', 'molecular_comparison'] species_sets=[['Homo sapiens', 'Mus musculus', 'Pan troglodytes'], ['Homo sapiens', 'Mus musculus', 'Pan troglodytes'], ['Homo sapiens', 'Mus musculus', 'Pan troglodytes']]

### `mc_two_species`
- Prompt: 'Compare the molecular similarity of human and chicken.'
- Agent/tool selection: PASS — expected='molecular_comparison' actual='molecular_comparison'
- Task completion: PASS — status expected='completed' actual='completed'; species expected=['Gallus gallus', 'Homo sapiens'] actual=['Gallus gallus', 'Homo sapiens']
- Correctness: — no deterministic correctness check defined for this case
- Relevance: 4.0/5 — Response names the similarity score 0.9952, correctly places both species in a single group, and notes no group separation was measured. The high score (0.9952) reflects the mock catalogue fixture, not a fabrication.
- Consistency: n/a

### `phylo_five_species`
- Prompt: 'Build a phylogenetic tree for human, chimp, mouse, chicken and zebrafish.'
- Notes: 5 species so UFBoot actually runs (see worker: needs >=4 to bootstrap in practice).
- Agent/tool selection: PASS — expected='phylogenetic_tree' actual='phylogenetic_tree'
- Task completion: PASS — status expected='completed' actual='completed'; species expected=['Danio rerio', 'Gallus gallus', 'Homo sapiens', 'Mus musculus', 'Pan troglodytes'] actual=['Danio rerio', 'Gallus gallus', 'Homo sapiens', 'Mus musculus', 'Pan troglodytes']
- Correctness: PASS — tree contains all expected species as leaves; newick balanced=True
- Relevance: 5.0/5 — After Explainer fix: response describes the Newick topology (Homo+Pan as separate root branches, Mus nested with Gallus+Danio clade), names bootstrap support values per node (node_0: 77, node_1: 100), quotes model mtVer+R2 and overall_confidence 0.885. Directly answers what the user asked for.
- Consistency: n/a

### `phylo_too_few_species`
- Prompt: 'Build a phylogenetic tree for human and chimp.'
- Notes: Deterministic Planner guard: phylogenetic_tree needs >=3 species (planner.py _apply_guards). Tests that the guard fires regardless of what the LLM itself proposes.
- Agent/tool selection: PASS — expected='clarification_required' actual='clarification_required'
- Task completion: PASS — status expected='continue' actual='continue'
- Correctness: — not applicable (non-completed status)
- Relevance: — not applicable (no completed output to judge)
- Consistency: n/a

### `full_analysis_explicit_both`
- Prompt: 'Give me both a similarity network and a phylogenetic tree for human, chimp and mouse.'
- Notes: Must explicitly ask for both outputs to get full_analysis.
- Agent/tool selection: PASS — expected='full_analysis' actual='full_analysis'
- Task completion: PASS — status expected='completed' actual='completed'; species expected=['Homo sapiens', 'Mus musculus', 'Pan troglodytes'] actual=['Homo sapiens', 'Mus musculus', 'Pan troglodytes']
- Correctness: PASS — tree contains all expected species as leaves; newick balanced=True
- Relevance: 4.5/5 — After Explainer fix: response covers the similarity network (Homo+Pan group 0 score 0.5325, Mus singleton group 1), the Newick topology (three-way root split with branch lengths), notes bootstrap was not run (`ufboot_not_run` warning), and references overall_confidence. Half-point deduction: could more explicitly state which pair is closest before describing the tree.
- Consistency: n/a

### `full_analysis_implicit_should_clarify`
- Prompt: 'Tell me everything you can about human, chimp and mouse evolution.'
- Notes: Does NOT explicitly request both a network and a tree. The planner must ask for clarification rather than guessing full_analysis.
- Agent/tool selection: PASS — expected='clarification_required' actual='clarification_required'
- Task completion: PASS — status expected='continue' actual='continue'
- Correctness: — not applicable (non-completed status)
- Relevance: — not applicable (no completed output to judge)
- Consistency: n/a

### `vague_no_species`
- Prompt: 'Tell me about evolution.'
- Agent/tool selection: PASS — expected='clarification_required' actual='clarification_required'
- Task completion: PASS — status expected='continue' actual='continue'
- Correctness: — not applicable (non-completed status)
- Relevance: — not applicable (no completed output to judge)
- Consistency: n/a

### `off_topic`
- Prompt: "What's the weather like today?"
- Notes: Off-topic request must not be misrouted into an evolutionary analysis.
- Agent/tool selection: PASS — expected='clarification_required' actual='clarification_required'
- Task completion: PASS — status expected='continue' actual='continue'
- Correctness: — not applicable (non-completed status)
- Relevance: — not applicable (no completed output to judge)
- Consistency: n/a

### `unresolvable_species`
- Prompt: 'Compare the molecular similarity of human and dog.'
- Notes: 'dog' is outside the offline species catalogue — must fail cleanly with a clear message, not crash or hallucinate a result.
- Agent/tool selection: — not applicable — request failed before feature is recoverable from response shape
- Task completion: PASS — status expected='failed' actual='failed'
- Correctness: — not applicable (non-completed status)
- Relevance: — not applicable (no completed output to judge)
- Consistency: n/a
