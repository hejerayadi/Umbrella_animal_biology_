# Test prompts — Evolution Agent

Prompts to type into the frontend (or `POST /execute` on the agent directly) to
exercise this agent end to end. Every prompt below was run against the live
agent — real Planner, real UniProt fetch, real MAFFT + IQ-TREE, real Explainer.
The routing, timings and values are measured, not assumed.

Start the agents from the repository root with `python -m backend.run_agents --reload`.

## Before you start — read this first

**Only five species resolve.** The species resolver is an offline catalogue
(`orchestrator/services/species_resolver.py`). Anything else is rejected before a
worker runs, no matter how well known it is.

| Scientific name | Accepted aliases |
|---|---|
| Homo sapiens | human, humans, human being |
| Pan troglodytes | chimpanzee, chimp, common chimpanzee |
| Mus musculus | mouse, house mouse, laboratory mouse |
| Gallus gallus | chicken, red junglefowl |
| Danio rerio | zebrafish, zebra fish, zebra danio |

**The first molecular request downloads the ESM-2 model** (`esm2_t12_35M_UR50D`,
~150 MB) into the torch hub cache. That one call is much slower than the ~15 s
below; every later call reuses the loaded model.

**Three deterministic guards fire after the Planner**, regardless of what the LLM
proposes: molecular comparison needs ≥ 2 species, a tree needs ≥ 3, and
`full_analysis` runs only when the user explicitly asks for *both* outputs.

---

## 1. Molecular comparison — the good demo

> How similar are human, chimp and mouse at the molecular level?

**Expect** `completed` in ~20 s, 2 LLM calls, no warnings. Human–chimp scores
`0.5325`; both human–mouse (`-0.8295`) and chimp–mouse (`-0.9144`) are negative.
Groups: `{human, chimp}` and `{mouse}` alone. Overall confidence `0.8511`.

Negative scores are correct and expected — embeddings are mean-centred per
request, so a score is a *relative* position within this species set, not an
absolute identity percentage. **If your frontend renders these as a percentage
bar, -0.91 will look broken.** Render them as a −1…1 scale.

## 2. Phylogenetic tree — the good demo

> Build a phylogenetic tree for humans, the common chimpanzee, house mouse, red junglefowl and zebra danio.

**Expect** `completed` in ~33 s, model `MTREV+G4`, overall confidence `0.925`,
no warnings. This prompt also exercises every alias in the catalogue at once —
the returned `species_list` should come back fully normalised to the five
scientific names.

Newick leaves are quoted because the names contain spaces:
`('Homo sapiens':0.024,'Pan troglodytes':0.049,(...))`. A viewer that splits on
whitespace will mangle them.

## 3. Both branches together

> Give me both a similarity network and a phylogenetic tree for human, chimp, mouse and chicken.

**Expect** `completed` in ~30 s, with `similarity_scores`, `species_groups`,
`similarity_network` *and* `newick_tree` populated. Overall confidence `0.8849`.
The word "both" is what unlocks this path.

## 4. Clarification — the agent should refuse to guess

> Tell me about evolution.

**Expect** `continue`, 1 LLM call only, ~5 s, and
`"Which species would you like to analyse?"`. The Explainer is never called on
this path, so the budget stays at 1.

> Build a phylogenetic tree for human and chimp.

**Expect** `continue` in ~4 s: *"Phylogenetic tree requires at least 3 species.
Would you like to add more, or switch to a similarity comparison?"* This is the
deterministic guard, not the LLM — it fires even if the model proposed a tree.

> Compare haemoglobin between human, mouse and chicken.

**Expect** `continue` — the agent asks whether you want a sequence comparison or
a tree. Note it does **not** yet act on "haemoglobin": naming a gene in prose
does not set `target_gene_or_protein`, which only arrives as a structured field.
Everything defaults to CYTB, then COX1.

## 5. Unsupported species — the honest failure

> Compare Homo sapiens and Gorilla gorilla at the molecular level.

**Expect** `failed` in ~6 s: *"Could not resolve species to scientific names:
'Gorilla gorilla'. Please use scientific names … or common names from the
supported catalogue."* Worth showing a reviewer: the agent refuses rather than
silently analysing four species and pretending the fifth was included.

> Compare the molecular similarity of polar bear and wombat.

**Expect** `continue`, not `failed` — the Planner asks which wombat species you
mean and never reaches the resolver. Both outcomes are acceptable; which one you
get depends on how confidently the LLM names a species.

## 6. Ambiguous phrasing — routes to the tree

> Analyse the evolutionary relationships between human, chimp, mouse and chicken.

> How are human, chimp and mouse related evolutionarily?

**Expect** `completed` with a tree and **no** `similarity_scores`. "Relationships"
reads as topology, so these take the phylogenetic branch only. If you want the
network too, you have to ask for both (§3).

---

## Known rough edges — avoid these in a live demo

**Two-species molecular comparison is misleading.**

> Compare the molecular similarity of human and chicken.

returns a similarity of **0.9952** and puts both in one group. Human and chicken
cytochrome b are not 99.5 % similar. Centring needs ≥ 3 points to estimate a
batch direction, so a 2-species request falls back to raw ESM-2 cosine, which is
anisotropic and near 1.0 for everything. Confidence is correctly reported as
unavailable, but the score on its own reads as a confident wrong answer. **Use
three or more species.**

**A 3-species tree has no branch support.**

> Build a phylogenetic tree for human, chimp and mouse.

returns `completed` with `warnings: ["ufboot_not_run"]`, `bootstrap_support: {}`
and `overall_confidence: null` — UFBoot needs ≥ 4 taxa. This is the agent being
honest rather than inventing a number, but the frontend must render a null
confidence as *"not available"* and not as `0 %`.

**Casing is inconsistent between fields.** `species_list` is title-cased
(`"Homo sapiens"`), while the names inside `similarity_scores` and
`species_groups` are lower-cased (`"homo sapiens"`). Match case-insensitively if
you join these for display.

## What to watch in the response

| Field | What it tells you |
|---|---|
| `status` | `completed` / `continue` (clarification) / `failed` |
| `interpretation` | the Explainer's prose (LLM #2), grounded and verified against the data |
| `explanation` | a deterministic one-line template — *not* the LLM's answer |
| `llm_calls` | 2 for a completed analysis, 1 for clarification. Never more. |
| `warnings` | `ufboot_not_run`, `offline_sequence_fallback`, `interpretation_unavailable`, `*_failed` |
| `score_is_mock` | must be `false` now that the real workers are wired in |
| `providers_are_mocked` | per-branch, e.g. `{"phylogeny": false}` |

If `warnings` contains `offline_sequence_fallback`, UniProt was unreachable and
the tree was built from the 60-aa offline fragments — treat that result as a
smoke test, not a finding.
