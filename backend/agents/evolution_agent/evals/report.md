# Evolution Agent — Agent Evaluation Report

## Summary

- Cases run: 9
- Agent/tool selection: 8/8 passed
- Task completion: 9/9 passed
- Correctness (deterministic): 3/3 passed
- Response consistency: 1/1 passed
- Relevance (LLM judge, mean): 3.0/5

## Weaknesses identified

- **phylo_five_species** (relevance, 2.0/5): The reply vaguely states a model and an overall confidence but does not provide the requested tree topology, method, sequences, or support values, and the specific model/confidence appear to be asserted without supporting data or methodology.
- **full_analysis_explicit_both** (relevance, 2.0/5): The reply partially addresses the request by naming the closest pair and stating a similarity score and that a tree was built, but it fails to provide the actual similarity network or tree and includes specific figures and a model name with no supporting data or justification.

## Per-case detail

### `mc_three_species`
- Prompt: 'How similar are human, chimp and mouse at the molecular level?'
- Notes: Baseline case. Human-chimp must be the closest pair regardless of run.
- Agent/tool selection: PASS — expected='molecular_comparison' actual='molecular_comparison'
- Task completion: PASS — status expected='completed' actual='completed'; species expected=['Homo sapiens', 'Mus musculus', 'Pan troglodytes'] actual=['Homo sapiens', 'Mus musculus', 'Pan troglodytes']
- Correctness: PASS — closest pair expected={'homo sapiens', 'pan troglodytes'} actual={'homo sapiens', 'pan troglodytes'} score=0.5325
- Relevance: 4.0/5 — The response meaningfully addresses the question by noting humans and chimps are closest and grouping mouse separately, but it fabricates an unexplained similarity value (0.53) and an unsupported confidence score (85%) without evidence, so the numerical claims are not grounded in the data.
- Consistency: PASS — features=['molecular_comparison', 'molecular_comparison', 'molecular_comparison'] species_sets=[['Homo sapiens', 'Mus musculus', 'Pan troglodytes'], ['Homo sapiens', 'Mus musculus', 'Pan troglodytes'], ['Homo sapiens', 'Mus musculus', 'Pan troglodytes']]

### `mc_two_species`
- Prompt: 'Compare the molecular similarity of human and chicken.'
- Agent/tool selection: PASS — expected='molecular_comparison' actual='molecular_comparison'
- Task completion: PASS — status expected='completed' actual='completed'; species expected=['Gallus gallus', 'Homo sapiens'] actual=['Gallus gallus', 'Homo sapiens']
- Correctness: — no deterministic correctness check defined for this case
- Relevance: 4.0/5 — The response does answer the user's request by giving a similarity comparison between human and chicken, but it makes implausible and unsupported claims (an exact similarity of 1.00 and that the two species form a single evolutionary group) that are not justified by typical molecular data and therefore are unfaithful to the evidence.
- Consistency: n/a

### `phylo_five_species`
- Prompt: 'Build a phylogenetic tree for human, chimp, mouse, chicken and zebrafish.'
- Notes: 5 species so UFBoot actually runs (see worker: needs >=4 to bootstrap in practice).
- Agent/tool selection: PASS — expected='phylogenetic_tree' actual='phylogenetic_tree'
- Task completion: PASS — status expected='completed' actual='completed'; species expected=['Danio rerio', 'Gallus gallus', 'Homo sapiens', 'Mus musculus', 'Pan troglodytes'] actual=['Danio rerio', 'Gallus gallus', 'Homo sapiens', 'Mus musculus', 'Pan troglodytes']
- Correctness: PASS — tree contains all expected species as leaves; newick balanced=True
- Relevance: 2.0/5 — The reply vaguely states a model and an overall confidence but does not provide the requested tree topology, method, sequences, or support values, and the specific model/confidence appear to be asserted without supporting data or methodology.
- Consistency: n/a

### `phylo_too_few_species`
- Prompt: 'Build a phylogenetic tree for human and chimp.'
- Notes: Deterministic Planner guard: phylogenetic_tree needs >=3 species (planner.py _apply_guards). Tests that the guard fires regardless of what the LLM itself proposes.
- Agent/tool selection: PASS — expected='clarification_required' actual='clarification_required'
- Task completion: PASS — status expected='continue' actual='continue'
- Correctness: — not applicable (non-completed status)
- Relevance: — skipped (no LLM configured)
- Consistency: n/a

### `full_analysis_explicit_both`
- Prompt: 'Give me both a similarity network and a phylogenetic tree for human, chimp and mouse.'
- Notes: Must explicitly ask for both outputs to get full_analysis.
- Agent/tool selection: PASS — expected='full_analysis' actual='full_analysis'
- Task completion: PASS — status expected='completed' actual='completed'; species expected=['Homo sapiens', 'Mus musculus', 'Pan troglodytes'] actual=['Homo sapiens', 'Mus musculus', 'Pan troglodytes']
- Correctness: PASS — tree contains all expected species as leaves; newick balanced=True
- Relevance: 2.0/5 — The reply partially addresses the request by naming the closest pair and stating a similarity score and that a tree was built, but it fails to provide the actual similarity network or tree and includes specific figures and a model name with no supporting data or justification.
- Consistency: n/a

### `full_analysis_implicit_should_clarify`
- Prompt: 'Tell me everything you can about human, chimp and mouse evolution.'
- Notes: Does NOT explicitly request both a network and a tree. The planner must ask for clarification rather than guessing full_analysis -- this is a prompt-adherence test, genuinely uncertain until run.
- Agent/tool selection: PASS — expected='clarification_required' actual='clarification_required'
- Task completion: PASS — status expected='continue' actual='continue'
- Correctness: — not applicable (non-completed status)
- Relevance: — skipped (no LLM configured)
- Consistency: n/a

### `vague_no_species`
- Prompt: 'Tell me about evolution.'
- Agent/tool selection: PASS — expected='clarification_required' actual='clarification_required'
- Task completion: PASS — status expected='continue' actual='continue'
- Correctness: — not applicable (non-completed status)
- Relevance: — skipped (no LLM configured)
- Consistency: n/a

### `off_topic`
- Prompt: "What's the weather like today?"
- Notes: Off-topic request must not be misrouted into an evolutionary analysis.
- Agent/tool selection: PASS — expected='clarification_required' actual='clarification_required'
- Task completion: PASS — status expected='continue' actual='continue'
- Correctness: — not applicable (non-completed status)
- Relevance: — skipped (no LLM configured)
- Consistency: n/a

### `unresolvable_species`
- Prompt: 'Compare the molecular similarity of human and dog.'
- Notes: 'dog' is outside the offline species catalogue -- must fail cleanly with a clear message, not crash or hallucinate a result.
- Agent/tool selection: — not applicable — request failed before feature is recoverable from response shape
- Task completion: PASS — status expected='failed' actual='failed'
- Correctness: — not applicable (non-completed status)
- Relevance: — skipped (no LLM configured)
- Consistency: n/a
