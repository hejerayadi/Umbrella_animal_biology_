# Phase 5 — Certification of the Complete Real Standalone Recognition Agent

**Status: PASS**

The complete real flow — `AgentRequest` → validation → Azure GPT-5 mini planning →
remote BioCLIP-2 classification → deterministic confidence gate → live GBIF/NCBI
enrichment → grounded explanation → delegation decision → `AgentResult` — runs end to
end against all real providers, and now describes its own provenance truthfully.

The first Phase 5 pass returned **PASS WITH LIMITATIONS**: three defects (F1, F2, F3)
caused false runtime provenance and false user-facing explanations, and the protection
boundary forbade fixing them. Under the Phase 5 Corrective Authorization they have been
corrected, the three strict xfails that recorded them are now ordinary passing regression
tests, and the corrected behaviour is verified live.

---

## 1. Baseline

| Item | Value |
| --- | --- |
| Branch | `group-d-recognition-sprint3` |
| HEAD SHA | `059277c4dbbfdffe2e80bf39e0b06e624550c444` |
| HEAD subject | `implementation ncbi gbif valide` |
| Phase 4 in HEAD | Yes |
| Working tree at start | Clean |
| Offline suite at baseline | **810 passed, 0 failed, 0 skipped** |

---

## 2. Environment checkpoint

Presence only. No value, endpoint, key or address appears in this report or in
`phase5_results.json`.

| Variable | State |
| --- | --- |
| `RECOGNITION_LLM_PROVIDER_MODE` | PRESENT |
| `AZURE_OPENAI_BASE_URL` | PRESENT |
| `AZURE_OPENAI_API_KEY` | PRESENT |
| `AZURE_OPENAI_DEPLOYMENT` | PRESENT |
| `BIOCLIP_PROVIDER_MODE` | PRESENT |
| `TAXONOMY_PROVIDER_MODE` | PRESENT |
| `NCBI_TOOL` | PRESENT |
| `NCBI_EMAIL` | PRESENT |
| `NCBI_API_KEY` | ABSENT (optional) |

At the start of Phase 5 both provider modes were `mock` and the NCBI variables were
absent. With explicit user approval they were switched to `remote`/`real`, the NCBI
identification was added, and the live values were left in place. `.env` is git-ignored
and untracked.

---

## 3. Files created or changed

**Implementation (corrective pass, authorized scope only):**

- `workflows/graph.py` — F1
- `workflows/nodes.py` — F2, F3

`git diff --stat HEAD` shows **exactly these two files**.

**Created (all untracked, none committed):**

- `tests/test_phase5_certification.py`
- `docs/PHASE5_CERTIFICATION.md` — this report
- `docs/phase5_results.json`

**Modified outside the repo index:** `.env` (git-ignored).

**Untouched, as required:** BioCLIP provider, taxonomy provider and rate limiter,
reasoning LLM provider, configuration, workflow state, agent construction, confidence
thresholds, ranking, schemas and the seven-key contract, Phase 3 and Phase 4 tests,
orchestrator, registry, frontend and other agents. No refactoring or unrelated cleanup.
Nothing was committed or pushed.

---

## 4. Test counts

| Execution | Before correction | After correction |
| --- | --- | --- |
| Complete offline suite | 892 passed, **3 xfailed** | **923 passed, 0 failed, 0 skipped, 0 xfailed** |
| Phase 5 module | 82 passed, 3 xfailed | **113 passed, 0 xfailed** |
| Phase 3 regression | 141 passed | **141 passed** |
| Phase 4 regression | 172 passed | **172 passed** |
| Architecture + no-leak gates | 63 passed | **63 passed** |
| Live corrective verification | — | **2 runs, ALL_PASS** |

Baseline was 810. The suite is fully green with **zero xfails**: the three strict xfails
were converted into positive regression tests, not deleted or weakened. Every Phase 1, 3
and 4 assertion about mock-mode provenance and mock-mode disclosure still passes
unchanged — the mock wording was preserved verbatim, because it was never the defect.

---

## 5. The corrections

### F1 — top-level taxonomy provenance now follows what actually ran

`graph.py` derives `gbif_mode` and `ncbi_mode` through a new
`nodes.taxonomy_source_modes(state)`, whose only input is runtime evidence: the
per-species `taxonomy_report` that the provider which actually ran produced. Configuration
is never consulted, so an injected or swapped provider reports as what it really is.

Edge cases are explicit rather than invented:

- **no lookup performed** (the classifier named nothing, so there was no candidate to
  validate) — the wired provider's own declared mode is reported, and a new additive
  provenance key `taxonomy_executed: false` says plainly that it was never asked anything;
- **a provider declaring no mode** — `"unknown"`, never a guess;
- **sources disagreeing** — `"mixed"`, never averaged away.

Because the top level is derived from the nested report, the two **cannot** contradict
each other by construction.

| | Before | After |
| --- | --- | --- |
| Live run | `gbif_mode: "mock"`, `ncbi_mode: "mock"` | `gbif_mode: "real"`, `ncbi_mode: "real"`, `taxonomy_executed: true` |
| Nested report | `"real"` — contradicting the above | `"real"` — agreeing |

### F2 — disclosure describes the providers that actually ran

`_safety_footer` now takes the runtime evidence and chooses each half independently: the
classification sentence from the mode the classifier that ran reports about itself, and
the taxonomy sentence from the derived source modes. Mixed and non-executed cases are
described only by what is proven. The taxonomy degradation warning names the source that
actually ran. The footer is still assembled **outside** model-generated text, so the model
cannot author provenance wording — asserted by a test that feeds the model its own false
disclaimer and checks the real one still appears.

**Before**, appended verbatim to every live answer:

> Species classification is produced by a deterministic Sprint 2 mock of BioCLIP-2, not by
> real BioCLIP-2 inference; the classification score is a test value, not a probability.
> GBIF and NCBI validation are mocked and were not checked against the live databases.

**After**, on a live run:

> Species classification is produced by real remote BioCLIP-2 inference; the classification
> score is a ranking score over the model's label set, not a calibrated probability. GBIF
> and NCBI identifiers come from live lookups against the public GBIF and NCBI services; a
> source that was unavailable or did not match leaves its identifier null rather than
> filled in.

Mock execution keeps its explicit mock disclosure, byte for byte.

### F3 — the explanation agrees with the structured evidence

The taxonomy sentence moved into `_taxonomy_sentence`, driven by the identifiers actually
on the candidate rather than by the status label alone, and naming each source with the
mode that ran. All four statuses — `verified`, `mock_verified`, `partial`, `unverified` —
plus the no-lookup case are handled explicitly. The classifier phrase is mode-aware too.

**Before**, in a single live response:

- `gbif_id: 2435350`, `ncbi_taxid: 9785`, `taxonomy_status: "verified"`
- explanation: *"Neither mocked taxonomy source supplied an identifier for it."*

**After:**

> Both taxonomy sources supplied an identifier for it: the live GBIF lookup returned
> 2435350 and the live NCBI lookup returned taxid 9785.

A test asserts every identifier-shaped number in the explanation appears in the structured
evidence, so an identifier can never be invented.

### F4 — unchanged, as instructed

Azure rate limiting remains informational. No retry was added and nothing was changed.

### F5 — introduced and fixed inside this pass

The first draft used `str.capitalize()` to start a sentence with a source name, which
lowercases the remainder and produced *"The live ncbi lookup"*. Caught by the degraded
live run, fixed with a `sentence_start` flag that capitalises only the article, and
covered by a regression test.

---

## 6. Live corrective verification

Two runs against Azure GPT-5 mini + remote BioCLIP-2 + real GBIF + real NCBI, each bounded
by a 240 s in-process deadline and an external timeout, with no Azure retry and deliberate
pacing so the deployment rate limit was not provoked.

**Run 1 — complete real agent** (*Loxodonta africana*, 18.6 s, 2 LLM calls, model-authored
explanation):

| Check | Result |
| --- | --- |
| top-level `gbif_mode == "real"` | ✅ |
| top-level `ncbi_mode == "real"` | ✅ |
| `taxonomy_executed == true` | ✅ |
| nested taxonomy modes all `real` | ✅ |
| nested and top-level agree | ✅ |
| explanation contains no mock or Sprint 2 claim | ✅ |
| explanation discloses remote inference and live lookups | ✅ |
| verified identifiers described accurately (`gbif 2435350`, `ncbi 9785`) | ✅ |
| seven output keys exact | ✅ |
| candidates, order and scores unchanged vs. the pre-correction run | ✅ |
| decision unchanged (`identified`) | ✅ |

**Run 2 — controlled taxonomy degradation** (*Acinonyx jubatus*, 8.1 s). GBIF was failed
by injecting a dead transport into **our own** client; no external service was attacked or
overloaded.

| Check | Result |
| --- | --- |
| still reports `gbif_mode`/`ncbi_mode` as `real` (an outage is not a change of provider) | ✅ |
| `taxonomy_degraded == true`, candidate `partial` | ✅ |
| `gbif_id` null and never inferred; `ncbi_taxid` 32536 survives | ✅ |
| explanation names the source that supplied and the one that did not | ✅ |
| warning says *"A live taxonomy source was unavailable"*, not *"mocked"* | ✅ |
| seven keys intact, request still completed | ✅ |

**Result: ALL_PASS.** The corrected explanation for the degraded run:

> Remote BioCLIP-2 inference supports Acinonyx jubatus as the highest-ranked taxonomic
> label (classification score 0.907, margin over the next label 0.834). The live NCBI
> lookup supplied taxid 32536; the live GBIF lookup supplied none, and its identifier is
> reported as null rather than filled in. […]

---

## 7. Offline coverage added

All twelve cases the authorization enumerates, deterministic and offline — no test calls
Azure, BioCLIP, GBIF or NCBI:

1. remote BioCLIP + real GBIF/NCBI ✅
2. mock BioCLIP + mock taxonomy ✅
3. verified real taxonomy ✅
4. mock-verified taxonomy ✅
5. partial taxonomy (each source down in turn) ✅
6. unavailable/degraded real taxonomy ✅
7. no taxonomy execution ✅
8. dependency-injected/swapped provider modes ✅
9. LLM-authored explanation followed by the correct deterministic footer ✅
10. deterministic fallback explanation ✅
11. absence of contradictory mock and real claims ✅
12. seven-key contract unchanged ✅

One correction worth recording: the earlier plan fixture used graph **node names**
(`classify_with_bioclip2`) rather than the plan **vocabulary** (`classify_image`,
`score_confidence`, `validate_taxonomy`, `explain`). `sanitize_plan` rejected it wholesale
— correct behaviour, but it meant the model-authored path was never actually exercised.
Fixed, so requirements 9 and 12 are genuinely tested rather than silently skipped.

---

## 8. Everything from the first pass that still stands

The twelve required scenarios, the invariance results, the seven-key contract, the
evaluation matrix and the calibration recommendation are unchanged by the correction and
remain as certified. In particular:

- BioCLIP is the only source of candidates; text, GPT and taxonomy cannot add, promote,
  reorder, rename or rescore one.
- Maximum 2 GPT calls; `store=False` on every Azure request; `SDK_MAX_RETRIES == 0`.
- Top-1 correct 5/7, expected species in Top-K 6/7, decisions 4 `identified` /
  1 `uncertain` / 2 `not_identified` — an integration observation on seven images, **not**
  an accuracy claim.
- **`CALIBRATION_RECOMMENDATION`: change nothing.** `IDENTIFIED_MIN_SCORE` (0.75),
  `IDENTIFIED_MIN_MARGIN` (0.08) and `UNCERTAIN_MIN_SCORE` (0.45) are untouched. The
  observed score distribution is strongly bimodal (0.91–0.95 vs 0.21–0.46) and the
  existing pair separates it cleanly.

The corrective live run re-confirmed that candidates, order, scores and decision are
byte-identical to the pre-correction run: provenance wording is downstream of the science
and did not touch it.

---

## 9. Remaining limitations

1. Seven images across six groups, three re-fetched for the corrective verification.
   Enough to certify integration; far too small for an accuracy claim.
2. Per-stage latency is still not instrumented — only end-to-end wall clock.
3. **F4 stands.** The Azure deployment rate-limits under back-to-back load, so the
   model-authored path is exercised opportunistically rather than on every live request.
   Both corrective runs did reach it.
4. Outage scenarios are injected into our own transports; no external service was
   deliberately failed.
5. Text alignment is inert in real mode by design (no name catalogue), so agreement and
   conflict are certified against the mock-catalogue provider.
6. The `"mixed"` and `"unknown"` taxonomy modes are covered offline only — no real
   deployment currently produces a per-source mode disagreement.

---

## 10. Confirmations

- **Only `workflows/graph.py` and `workflows/nodes.py` changed.** No threshold, ranking
  rule, schema, seven-key contract, provider, config, workflow state or agent construction
  was touched. Phases 3 and 4 remain unchanged and fully green.
- **Nothing was committed or pushed.** Three untracked Phase-5 artifacts plus the two
  modified implementation files.
- **No leaks.** No secret, endpoint, email, image byte, model weight, cache or temporary
  artifact is untracked-and-unignored. Evaluation images were deleted after validation. No
  stray temp file from the remote classifier and no lingering process remained.

---

## Acceptance gate

| Requirement | Result |
| --- | --- |
| F1, F2 and F3 fixed | ✅ PASS |
| Complete suite has zero failures | ✅ PASS — 923 passed |
| Three strict xfails converted into normal passes | ✅ PASS |
| Zero xfails related to these findings | ✅ PASS — zero xfails in the suite |
| Real and mock provenance both accurate | ✅ PASS |
| Explanation never contradicts structured evidence | ✅ PASS |
| Phases 3 and 4 unchanged | ✅ PASS — 141 and 172, untouched |
| No secret, image, cache or temporary artifact tracked | ✅ PASS |
| Complete real flow on previously unregistered images | ✅ PASS |
| Taxonomy only annotates | ✅ PASS |
| GPT grounded and within two calls | ✅ PASS |
| Output schema and seven keys unchanged | ✅ PASS |
| Provenance accurately reports the providers | ✅ **PASS** |

**Phase 5: PASS.**

*Stopping here. Phase 6 not begun. Nothing committed or pushed.*
