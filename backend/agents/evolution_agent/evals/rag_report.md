# Evolution Agent — RAG Evaluation Report (Species Resolver)

## Overview

The Species Resolver is the retrieval component of the Evolution Agent.
It maps free-text species names (common names, aliases, scientific names)
to canonical scientific names that the downstream workers require.

This evaluation uses RAGAS-style metrics measured deterministically
(no LLM required). The offline backend is evaluated here; the Qdrant
vector backend (Sprint 3+) will be evaluated when the cluster is live.

## RAGAS Metrics

| Metric | Definition | Score |
|---|---|---|
| Retrieval Relevance   | Returned the correct canonical name | 1.000 |
| Context Quality       | Returned name is well-formed (Genus epithet) | 1.000 |
| Answer Faithfulness   | Returned name exists in the knowledge base | 1.000 |
| Answer Relevance      | Returned name answers the user's query intent | 1.000 |

**Overall: 36/36 cases passed**

## Pass Rate by Category

| Category | Passed | Total |
|---|---|---|
| alias | 8 | 8 |
| batch | 3 | 3 |
| case_normalisation | 4 | 4 |
| common_name | 6 | 6 |
| edge_case | 3 | 3 |
| exact_scientific | 5 | 5 |
| fuzzy | 2 | 2 |
| unknown | 5 | 5 |

## Failures and Weaknesses

No failures — all cases passed.

## Known Limitations

1. **Catalogue coverage**: only 5 species are in the offline catalogue.
   Any species outside this set fails at the resolver, not the worker.
   Sprint 3 will replace the offline dict with a Qdrant vector search
   over ESMC embeddings, extending coverage to any species with a
   GenBank entry.

2. **Substring fallback ambiguity**: the fuzzy fallback uses substring
   matching (`key in query or query in key`). A query like `'pan'`
   would match `'pan troglodytes'` with score 0.6, which is correct,
   but `'mouse'` inside `'house mouse'` also matches, producing the
   right answer for the wrong reason. The Qdrant backend will resolve
   this with semantic similarity instead.

3. **No typo tolerance**: `'chimpanzea'` (one character off) returns
   None because neither exact nor substring matching handles typos.
   The Qdrant backend with cosine similarity will handle this.

4. **Qdrant backend not evaluated**: `_QdrantBackend` requires a live
   cluster and is excluded from this run. A separate evaluation run
   with `QDRANT_URL` and `QDRANT_API_KEY` set will exercise it.

## Per-Case Detail

### `exact_homo_sapiens` — PASS
- Query: `Homo sapiens`
- Expected: `Homo sapiens`  Actual: `Homo sapiens`  Score: 1.0
- Detail: expected='Homo sapiens' actual='Homo sapiens'; score=1.00 (min=1.0) ✓
- Notes: Title-cased scientific name — exact dict hit, score must be 1.0.
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `exact_pan_troglodytes` — PASS
- Query: `Pan troglodytes`
- Expected: `Pan troglodytes`  Actual: `Pan troglodytes`  Score: 1.0
- Detail: expected='Pan troglodytes' actual='Pan troglodytes'; score=1.00 (min=1.0) ✓
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `exact_mus_musculus` — PASS
- Query: `Mus musculus`
- Expected: `Mus musculus`  Actual: `Mus musculus`  Score: 1.0
- Detail: expected='Mus musculus' actual='Mus musculus'; score=1.00 (min=1.0) ✓
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `exact_gallus_gallus` — PASS
- Query: `Gallus gallus`
- Expected: `Gallus gallus`  Actual: `Gallus gallus`  Score: 1.0
- Detail: expected='Gallus gallus' actual='Gallus gallus'; score=1.00 (min=1.0) ✓
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `exact_danio_rerio` — PASS
- Query: `Danio rerio`
- Expected: `Danio rerio`  Actual: `Danio rerio`  Score: 1.0
- Detail: expected='Danio rerio' actual='Danio rerio'; score=1.00 (min=1.0) ✓
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `lower_homo_sapiens` — PASS
- Query: `homo sapiens`
- Expected: `Homo sapiens`  Actual: `Homo sapiens`  Score: 1.0
- Detail: expected='Homo sapiens' actual='Homo sapiens'; score=1.00 (min=1.0) ✓
- Notes: Lowercase scientific name must resolve to title-cased canonical.
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `lower_mus_musculus` — PASS
- Query: `mus musculus`
- Expected: `Mus musculus`  Actual: `Mus musculus`  Score: 1.0
- Detail: expected='Mus musculus' actual='Mus musculus'; score=1.00 (min=1.0) ✓
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `common_human` — PASS
- Query: `human`
- Expected: `Homo sapiens`  Actual: `Homo sapiens`  Score: 1.0
- Detail: expected='Homo sapiens' actual='Homo sapiens'; score=1.00 (min=1.0) ✓
- Notes: Most common alias — must resolve exactly.
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `common_chimp` — PASS
- Query: `chimp`
- Expected: `Pan troglodytes`  Actual: `Pan troglodytes`  Score: 1.0
- Detail: expected='Pan troglodytes' actual='Pan troglodytes'; score=1.00 (min=1.0) ✓
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `common_chimpanzee` — PASS
- Query: `chimpanzee`
- Expected: `Pan troglodytes`  Actual: `Pan troglodytes`  Score: 1.0
- Detail: expected='Pan troglodytes' actual='Pan troglodytes'; score=1.00 (min=1.0) ✓
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `common_mouse` — PASS
- Query: `mouse`
- Expected: `Mus musculus`  Actual: `Mus musculus`  Score: 1.0
- Detail: expected='Mus musculus' actual='Mus musculus'; score=1.00 (min=1.0) ✓
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `common_chicken` — PASS
- Query: `chicken`
- Expected: `Gallus gallus`  Actual: `Gallus gallus`  Score: 1.0
- Detail: expected='Gallus gallus' actual='Gallus gallus'; score=1.00 (min=1.0) ✓
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `common_zebrafish` — PASS
- Query: `zebrafish`
- Expected: `Danio rerio`  Actual: `Danio rerio`  Score: 1.0
- Detail: expected='Danio rerio' actual='Danio rerio'; score=1.00 (min=1.0) ✓
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `alias_humans` — PASS
- Query: `humans`
- Expected: `Homo sapiens`  Actual: `Homo sapiens`  Score: 1.0
- Detail: expected='Homo sapiens' actual='Homo sapiens'; score=1.00 (min=1.0) ✓
- Notes: Plural form — separate entry in catalogue.
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `alias_human_being` — PASS
- Query: `human being`
- Expected: `Homo sapiens`  Actual: `Homo sapiens`  Score: 1.0
- Detail: expected='Homo sapiens' actual='Homo sapiens'; score=1.00 (min=1.0) ✓
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `alias_common_chimpanzee` — PASS
- Query: `common chimpanzee`
- Expected: `Pan troglodytes`  Actual: `Pan troglodytes`  Score: 1.0
- Detail: expected='Pan troglodytes' actual='Pan troglodytes'; score=1.00 (min=1.0) ✓
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `alias_house_mouse` — PASS
- Query: `house mouse`
- Expected: `Mus musculus`  Actual: `Mus musculus`  Score: 1.0
- Detail: expected='Mus musculus' actual='Mus musculus'; score=1.00 (min=1.0) ✓
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `alias_laboratory_mouse` — PASS
- Query: `laboratory mouse`
- Expected: `Mus musculus`  Actual: `Mus musculus`  Score: 1.0
- Detail: expected='Mus musculus' actual='Mus musculus'; score=1.00 (min=1.0) ✓
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `alias_red_junglefowl` — PASS
- Query: `red junglefowl`
- Expected: `Gallus gallus`  Actual: `Gallus gallus`  Score: 1.0
- Detail: expected='Gallus gallus' actual='Gallus gallus'; score=1.00 (min=1.0) ✓
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `alias_zebra_fish` — PASS
- Query: `zebra fish`
- Expected: `Danio rerio`  Actual: `Danio rerio`  Score: 1.0
- Detail: expected='Danio rerio' actual='Danio rerio'; score=1.00 (min=1.0) ✓
- Notes: Two-word variant of zebrafish.
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `alias_zebra_danio` — PASS
- Query: `zebra danio`
- Expected: `Danio rerio`  Actual: `Danio rerio`  Score: 1.0
- Detail: expected='Danio rerio' actual='Danio rerio'; score=1.00 (min=1.0) ✓
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `fuzzy_the_common_chimpanzee` — PASS
- Query: `the common chimpanzee`
- Expected: `Pan troglodytes`  Actual: `Pan troglodytes`  Score: 0.6
- Detail: expected='Pan troglodytes' actual='Pan troglodytes'; score=0.60 (min=0.6) ✓
- Notes: Substring fallback: 'common chimpanzee' is a substring of the query so the resolver should still find Pan troglodytes at score 0.6.
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `fuzzy_laboratory_mouse_strain` — PASS
- Query: `laboratory mouse strain`
- Expected: `Mus musculus`  Actual: `Mus musculus`  Score: 0.6
- Detail: expected='Mus musculus' actual='Mus musculus'; score=0.60 (min=0.6) ✓
- Notes: 'laboratory mouse' appears inside the query string.
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `mixed_case_HUMAN` — PASS
- Query: `HUMAN`
- Expected: `Homo sapiens`  Actual: `Homo sapiens`  Score: 1.0
- Detail: expected='Homo sapiens' actual='Homo sapiens'; score=1.00 (min=1.0) ✓
- Notes: All-caps input normalised to lowercase before lookup.
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `mixed_case_ZebraFish` — PASS
- Query: `ZebraFish`
- Expected: `Danio rerio`  Actual: `Danio rerio`  Score: 1.0
- Detail: expected='Danio rerio' actual='Danio rerio'; score=1.00 (min=1.0) ✓
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `whitespace_leading_trailing` — PASS
- Query: `  homo sapiens  `
- Expected: `Homo sapiens`  Actual: `Homo sapiens`  Score: 1.0
- Detail: expected='Homo sapiens' actual='Homo sapiens'; score=1.00 (min=1.0) ✓
- Notes: Leading/trailing whitespace is stripped by resolve().
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `empty_string` — PASS
- Query: ``
- Expected: `None`  Actual: `None`  Score: None
- Detail: correctly returned None
- Notes: Empty string must return None — not crash.
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `whitespace_only` — PASS
- Query: `   `
- Expected: `None`  Actual: `None`  Score: None
- Detail: correctly returned None
- Notes: Whitespace-only string must return None — not crash.
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `unknown_dog` — PASS
- Query: `dog`
- Expected: `None`  Actual: `None`  Score: None
- Detail: correctly returned None
- Notes: Dog (Canis lupus familiaris) is not in the offline catalogue. Must return None cleanly — not hallucinate a species.
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `unknown_cat` — PASS
- Query: `cat`
- Expected: `None`  Actual: `None`  Score: None
- Detail: correctly returned None
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `unknown_elephant` — PASS
- Query: `elephant`
- Expected: `None`  Actual: `None`  Score: None
- Detail: correctly returned None
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `unknown_scientific_gorilla` — PASS
- Query: `Gorilla gorilla`
- Expected: `None`  Actual: `None`  Score: None
- Detail: correctly returned None
- Notes: Gorilla is not in the catalogue even though it shares a genus with common primates. Must not be confused with Pan troglodytes.
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `unknown_gibberish` — PASS
- Query: `xyzzy blorp`
- Expected: `None`  Actual: `None`  Score: None
- Detail: correctly returned None
- Notes: Complete gibberish — must return None without crashing.
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `batch_all_known` — PASS
- Query: `human|chimp|mouse`
- Expected: `Homo sapiens|Pan troglodytes|Mus musculus`  Actual: `Homo sapiens|Pan troglodytes|Mus musculus`  Score: None
- Detail: canonical=['Homo sapiens', 'Pan troglodytes', 'Mus musculus'] (expected=['Homo sapiens', 'Pan troglodytes', 'Mus musculus']); unresolved=[] (expected=[])
- Notes: All known species — unresolved list must be empty.
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `batch_mixed_known_unknown` — PASS
- Query: `human|dog`
- Expected: `Homo sapiens|<unresolved:dog>`  Actual: `Homo sapiens`  Score: None
- Detail: canonical=['Homo sapiens'] (expected=['Homo sapiens']); unresolved=['dog'] (expected=['dog'])
- Notes: Mixed batch: 'human' resolves, 'dog' does not. canonical=['Homo sapiens'], unresolved=['dog'].
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0

### `batch_all_unknown` — PASS
- Query: `dog|cat`
- Expected: `<unresolved:dog>|<unresolved:cat>`  Actual: ``  Score: None
- Detail: canonical=[] (expected=[]); unresolved=['dog', 'cat'] (expected=['dog', 'cat'])
- Notes: All unknown — canonical list must be empty.
- RR=1.0  CQ=1.0  AF=1.0  AR=1.0
