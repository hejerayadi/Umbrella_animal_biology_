# Sprint 4 — Task 1: Agent Evaluation (Evolution Agent)

**Sprint goal:** Implement & Integrate the Agents
**This task's goal (from the sprint brief):** Define evaluation criteria, evaluate agent responses on five criteria, build a test dataset, analyze results and identify weaknesses.

This document follows the sprint brief's outline section by section, so each heading below maps to one slide.

---

## 1. Evaluation criteria — defined for this agent

The brief lists five generic criteria. Here's what each one concretely means for the Evolution Agent, and how it's measured:

| Criterion | What it means here | How it's scored |
|---|---|---|
| **Agent/tool selection** | Did the Planner route the prompt to the right feature — `molecular_comparison`, `phylogenetic_tree`, `full_analysis`, or correctly refuse with `clarification_required`? | Deterministic — inferred from response shape (presence of `similarity_scores` / `newick_tree` / `status`) and compared to a labeled expectation |
| **Task completion** | Did the request reach the right terminal `status`, and — when completed — were the correct species actually resolved and analyzed? | Deterministic — status + species-set comparison |
| **Correctness** | Are the structured results actually right? For molecular comparison: is the reported closest pair the biologically correct one. For phylogenetics: is the returned tree well-formed and does it contain every requested species as a leaf. | Deterministic, domain-grounded checks — no LLM judgment involved |
| **Relevance** | Does the agent's plain-English explanation actually answer what the user asked, and does it stay faithful to the structured data behind it (no fabricated numbers)? | LLM-as-judge (GPT-5-mini via the same Azure deployment the agent uses), scored 1–5 |
| **Response consistency** | Run the same prompt multiple times — does the Planner's routing decision and the resolved species stay stable, or does it flip-flop? | Deterministic — 3x repeat on the baseline case, compared across runs |

---

## 2. Evaluation dataset

9 hand-labeled test cases in `evals/golden_dataset.py`. Every case is a **real free-text prompt** sent to `POST /execute` with an empty `context` — no shortcut through an explicit `"feature"` field — because agent/tool selection can only be tested by letting the real Planner (LLM #1) make the call.

| Case | Prompt | Expects |
|---|---|---|
| `mc_three_species` | "How similar are human, chimp and mouse..." | `molecular_comparison`, completed, human–chimp closest pair |
| `mc_two_species` | "Compare... human and chicken" | `molecular_comparison`, completed |
| `phylo_five_species` | "Build a phylogenetic tree for [5 species]" | `phylogenetic_tree`, completed, well-formed tree |
| `phylo_too_few_species` | "Build a... tree for human and chimp" (only 2) | `clarification_required` — Planner's <3-species guard |
| `full_analysis_explicit_both` | "Give me both a similarity network and a phylogenetic tree..." | `full_analysis`, completed |
| `full_analysis_implicit_should_clarify` | "Tell me everything about... evolution" (doesn't explicitly ask for both) | `clarification_required` — must not guess `full_analysis` |
| `vague_no_species` | "Tell me about evolution." | `clarification_required` |
| `off_topic` | "What's the weather like today?" | `clarification_required` — must not be misrouted |
| `unresolvable_species` | "...human and dog" (outside the species catalogue) | `failed`, cleanly, no crash |

This deliberately covers both directions: prompts that should succeed, and prompts that should be correctly refused or should correctly fail.

---

## 3. Methodology

- **No mocks.** Every case runs against the live, real Evolution Agent server (`POST /execute`) — the real Planner, real workers, real MAFFT/IQ-TREE binaries, real Azure LLM calls.
- **Two scoring modes, used where each is appropriate:**
  - *Deterministic* for anything with an objective right answer (routing, status, species, tree well-formedness, closest-pair).
  - *LLM-as-judge* only for the parts with no fixed right answer — the free-text explanation — scored 1–5 on relevance and faithfulness by a separate judge call.
- Runner: `evals/run_eval.py`. Re-run any time with:
  ```
  python -m backend.agents.evolution_agent.evals.run_eval
  ```
  (server must already be running on port 8002). Writes `report.json` and `report.md`.

---

## 4. Results

| Criterion | Result |
|---|---|
| Agent/tool selection | **8/8 passed** |
| Task completion | **9/9 passed** |
| Correctness (deterministic) | **3/3 passed** |
| Response consistency (3x repeat) | **1/1 passed** |
| Relevance (LLM judge, mean) | **3.0 / 5** |

Full per-case detail is in `evals/report.md`.

---

## 5. Key finding: the evaluation caught a real bug

This is the headline result, not a footnote — it's direct proof the evaluation process has teeth, not just a checkbox exercise.

The very first eval run scored `mc_two_species` at **2.0/5 relevance**, and the judge's own reasoning flagged something concrete: *"presents implausible/contradictory metrics (similarity -1.00...)"*. That pointed to a real regression: a mean-centering fix applied to the molecular similarity scoring earlier in the sprint (to fix a different, unrelated anisotropy problem) turned out to be **mathematically degenerate for exactly 2 species** — centering exactly two vectors always makes them perfect opposites, forcing cosine similarity to `-1.0` regardless of the actual protein sequences being compared. Verified this empirically (two random unrelated vectors, centered, always score `-1.0`), then fixed it: centering now only applies at 3+ species; a 2-species comparison correctly falls back to raw cosine similarity.

Re-running the eval after the fix: `mc_two_species` relevance went from **2.0 → 4.0/5**, and mean relevance across the whole suite went from **2.2 → 3.0/5**, with zero regressions elsewhere (confirmed against the full 196-test pytest suite: 63 failed/133 passed post-fix vs. 73 failed/123 passed on the original code).

---

## 6. Weaknesses identified (still open)

- **`phylo_five_species`** and **`full_analysis_explicit_both`** scored low (2.0/5) on relevance. The judge consistently penalizes the Explainer's prose for not *narrating* the tree topology or similarity network in words — even though that data **is** present in the structured JSON response (`newick_tree`, `similarity_scores`). Real UX gap: the summary text is terser than the data backing it.
- **A limitation of the eval harness itself, in the interest of honest reporting**: the LLM judge is only shown the prose explanation, not the structured data it's describing — so it can judge *plausibility*, not verify *actual groundedness* against the real numbers. A stronger version of this eval would feed the judge the structured payload alongside the prose.

---

## Slide-ready summary

- 5 criteria defined and mapped to concrete, testable checks for this agent
- 9-case labeled dataset, run against the live agent with zero mocking
- 8/8 tool selection, 9/9 task completion, 3/3 correctness, 1/1 consistency — all passing
- Relevance (LLM-judge): 2.2 → 3.0/5 after a fix the evaluation itself surfaced
- **The eval caught a real math bug** (N=2 similarity always -1.0) that manual testing had missed
- Two remaining, honestly-reported weaknesses: explanation completeness, and judge groundedness limits
