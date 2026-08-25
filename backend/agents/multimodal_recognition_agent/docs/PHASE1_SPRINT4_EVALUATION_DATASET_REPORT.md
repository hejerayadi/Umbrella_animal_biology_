# Phase 1 Report — Sprint 4 Evaluation Dataset

**Sprint:** 4 — Multimodal Species Recognition Agent
**Phase:** 1 — evaluation criteria and dataset specification
**Owner:** Leith
**Branch:** `group-d-recognition-sprint4`
**Date:** 2026-08-25

No `.env` value, API key, Azure endpoint, image byte, Base64 payload, data URL
or private path appears anywhere in this report or in anything it added.

---

## Status

**PASS**

36 documented cases across 12 independently verified species, every real image
carrying a free licence and full attribution, every taxonomy identifier checked
against the official GBIF and NCBI sources, 80 new validation tests passing, and
the complete Recognition offline suite green at **1276 passed** with the 1196
baseline entirely intact.

---

## 1. Preconditions verified before implementation

| Check | Result |
| --- | --- |
| Branch | `group-d-recognition-sprint4` |
| HEAD at start | `132dac43e8ea0e5558517d5372c8a3e5074f782f` — *"Document validated Recognition Sprint 4 baseline"* |
| Working tree at start | **clean** (`git status --porcelain=v1` empty) |
| Phase 0 report committed | **Yes** — `git ls-files --error-unmatch docs/PHASE0_SPRINT4_BASELINE_REPORT.md` resolves |
| Phase 0 gate | PASS |

Runtime contracts were read from source, not from documentation, and every
accepted value in the dataset is derived from them:

| Contract | Source | Values |
| --- | --- | --- |
| Statuses | `schema.py::AgentStatus` | `completed`, `needs_agent`, `failed` (`continue` excluded — no code path constructs it) |
| Decisions | `domain/models.py::Decision` | `identified`, `uncertain`, `not_identified` |
| Error codes | `domain/errors.py::ErrorCode` | 17 members, plus the `INTERNAL_ERROR` literal from `api.py` |
| Delegation capabilities | `workflows/state.py::HELPER_OUTPUT_KEYS` | `Evolution`, `Genome`, `Biodiversity`, `Trait`, `Literature`, `Protein` |
| Resume keys | same map's values | `evolution_analysis`, `genome`, `biodiversity_report`, `traits`, `papers`, `protein_structure` |
| Delegation preconditions | `workflows/nodes.py::make_delegation_node` | intent `scientific_follow_up` **and** decision `identified` **and** a primary species **and** a recognised capability |
| Media types / bounds | `config.py` | `image/jpeg`, `image/png`, `image/webp`; ≤10 MiB; ≤25 MP; ≥64×64 |

Repository conventions were followed: JSON data files carry a leading
`_fixture_note`-style disclosure (here `_benchmark_note`) exactly as
`fixtures/mock_bioclip_predictions.json` and `fixtures/mock_taxonomy.json` do,
generated assets are excluded by a local `.gitignore` exactly as
`fixtures/demo_images/` is, and the tests live in the agent's own `tests/`
directory so they run with the complete suite.

**No Recognition runtime file was changed.** See §8.

---

## 2. Files created

| Path | Lines | Role |
| --- | ---: | --- |
| `evaluation/__init__.py` | 14 | Package docstring; states that this package contains no runner and cannot execute the agent |
| `evaluation/.gitignore` | 8 | Excludes `assets/` — image bytes are never committed |
| `evaluation/README.md` | 89 | Bounded-benchmark declaration, ground-truth policy, permitted and prohibited use |
| `evaluation/manifest.json` | 2 098 | **The dataset** — 36 cases, 12 species, full provenance per case |
| `evaluation/manifest_schema.py` | 188 | Accepted vocabularies **derived from runtime contracts**, plus the deterministic loader |
| `evaluation/fetch_assets.py` | 118 | Minimal opt-in, deterministic image downloader. Not a runner |
| `tests/test_sprint4_phase1_evaluation_dataset.py` | 658 | The 80 validation tests |
| `docs/PHASE1_SPRINT4_EVALUATION_DATASET_REPORT.md` | 476 | This report |

All eight are **untracked**; nothing was committed or pushed.

---

## 3. Cases by category

| Category | Required | **Delivered** |
| --- | ---: | ---: |
| Real animal-image recognition | ≥ 20 | **24** |
| Ambiguous, poor-quality or non-animal | ≥ 5 | **5** |
| Invalid-input or dependency-failure | ≥ 3 | **4** (3 invalid input + 1 dependency failure) |
| Delegation and resume | ≥ 2 | **3** (2 escalations + 1 resume) |
| **Total** | **≥ 30** | **36** |

### The 24 real cases

12 species × 2 distinct images each. **11 clear, 13 challenging.**

| Species | GBIF usageKey | NCBI taxid | Case ids |
| --- | ---: | ---: | --- |
| *Panthera leo* | 5219404 | 9689 | `REC-PLEO-01/02` |
| *Panthera tigris* | 5219416 | 9694 | `REC-PTIG-01/02` |
| *Panthera pardus* | 5219436 | 9691 | `REC-PPAR-01/02` |
| *Ursus maritimus* | 2433451 | 29073 | `REC-UMAR-01/02` |
| *Loxodonta africana* | 2435350 | 9785 | `REC-LAFR-01/02` |
| *Giraffa camelopardalis* | 2441205 | 9894 | `REC-GCAM-01/02` |
| *Equus quagga* | 2440892 | 89248 | `REC-EQUA-01/02` |
| *Ailuropoda melanoleuca* | 2433399 | 9646 | `REC-AMEL-01/02` |
| *Vulpes lagopus* | 5219303 | 494514 | `REC-VLAG-01/02` |
| *Phoenicopterus roseus* | 4352332 | 435638 | `REC-PROS-01/02` |
| *Aptenodytes forsteri* | 2481661 | 9233 | `REC-AFOR-01/02` |
| *Bubo scandiacus* | 5959143 | 371907 | `REC-BSCA-01/02` |

**Species represented: 12** (requirement: ≥ 10). Every species has two *distinct*
source files — asserted by test, so the same image cannot be counted twice.

The challenging half is challenging for stated, inspectable reasons: a melanistic
leopard whose rosettes are nearly invisible; an Arctic fox in grey-brown summer
coat rather than the familiar white morph; a flamingo reduced to a backlit
silhouette; an extreme crop of zebra stripes with no body context; distant
colonies, herds and cubs; monochrome camera-trap frames; and animals occluded by
fences and bamboo.

### The 5 ambiguous / non-animal cases

Each was drawn from a real species category and then **rejected as a recognition
case on inspection**, which is what makes them realistic hard negatives rather
than synthetic ones.

| Case | Kind | What it actually shows |
| --- | --- | --- |
| `AMB-01` | `non_animal` | A zoo walkway with visitors and fencing. Filed under *Panthera leo*; contains no animal at all. |
| `AMB-02` | `poor_quality` | Sea ice and open water; any bear is far below the resolution at which a species could be called. |
| `AMB-03` | `non_animal_depiction` | A white giraffe **sculpture** on a city street — right outline, not an organism. |
| `AMB-04` | `non_animal` | A painted zoo mural of a world map and tiger subspecies names. |
| `AMB-05` | `ambiguous_specimen` | A taxidermy leopard in a glass display case — the right species, but not a live animal. |

None carries species-level ground truth, and all four species metrics are
declared **not applicable** on each. Accepted decisions are restricted to
`uncertain` and `not_identified`, with `expected_status` `completed`, because an
inconclusive answer is a completed scientific outcome in this agent, not a
failure.

### The 4 failure cases

| Case | Expectation | Basis |
| --- | --- | --- |
| `INV-01` | `failed` / `EMPTY_INSTRUCTION` | Valid image, whitespace-only instruction |
| `INV-02` | `failed` / `MISSING_IMAGE` | Valid instruction, no image entry |
| `INV-03` | `failed` / `UNSUPPORTED_IMAGE_SOURCE` | `https://` URL instead of an inline data URL; the agent never fetches |
| `DEP-01` | `failed` / `CLASSIFICATION_UNAVAILABLE` | Injected failing classification transport — offline only, no real provider is broken |

### The 3 delegation cases

| Case | Expectation |
| --- | --- |
| `DEL-01` | `needs_agent`, capability `Genome`, decision `identified` |
| `DEL-02` | `needs_agent`, capability `Evolution`, decision `identified` |
| `DEL-03` | `completed` with the resume key `genome` already in context — the agent must finish, not escalate twice |

`DEL-01` and `DEL-02` record that delegation is *conditional* on the classifier
reaching `identified`. If it does not, that is a finding about the classifier to
report in Phase 3, **not** a dataset defect to be edited away.

---

## 4. Ground-truth verification method

### Species identity — source page **plus direct visual inspection**

Wikimedia Commons category membership alone proved insufficient, and that is the
single most important methodological finding of this phase. Candidates were
harvested from each species' own Commons category and filtered
programmatically, then **every one of the 48 shortlisted images was rendered and
inspected by eye** before any of it entered the manifest.

That inspection rejected material that filename and category metadata had
happily endorsed:

- a zoo walkway with visitors, filed under *Panthera leo*;
- a painted information mural, filed under *Panthera tigris*;
- a taxidermy specimen in a display case, filed under *Panthera pardus*;
- a **street sculpture**, filed under *Giraffa camelopardalis*;
- two sea-ice frames in which no bear is resolvable;
- a historic photograph of a hunter with dead tigers;
- book-plate scans and an anatomical diagram;
- and, critically, **two zebra hybrids** (`Donkey × Plains Zebra` and
  `Grévy's × Plains Zebra`) which are *not* *Equus quagga* and would have been
  silently wrong ground truth.

Five of the rejects were good enough as negatives that they were deliberately
re-purposed into the ambiguous category rather than discarded.

Each case records what the inspection actually saw, in its `notes` and
`ground_truth_basis` fields, so a reviewer can audit the judgement without
re-downloading anything.

### Taxonomy identifiers — official sources, checked once, written down

| Source | Endpoint | Acceptance rule |
| --- | --- | --- |
| GBIF | `https://api.gbif.org/v1/species/match` | accepted **only** on `matchType == "EXACT"`, `rank == "SPECIES"` and `status == "ACCEPTED"` |
| NCBI | official NCBI Taxonomy Browser web pages | accepted **only** when an entry of rank *species* carried the exact binomial |

All 12 species resolved cleanly at both sources; **nothing was left unresolved
and nothing was guessed.**

Two deliberate choices worth recording:

1. **NCBI E-utilities were not used.** NCBI's guidelines ask API callers to
   identify themselves with a tool and an email, and the agent's own
   `build_taxonomy_provider` refuses to start rather than "calling NCBI
   unidentified". Ordinary low-volume Taxonomy Browser page reads honour that
   principle and required no email; **no user email was transmitted anywhere.**
2. **This verification was research, performed once by the dataset author and
   recorded in the manifest.** It is not a code path. No test and no module in
   this repository contacts GBIF, NCBI or any other external service — asserted
   by the offline test suite, which passes with no network available to it.

> **Scope note.** The Phase 1 instructions require identifiers "verified against
> official sources" and also state that no live GBIF or NCBI call is permitted.
> These were read as addressing different things: the prohibition governs the
> **code and the test run**, which contact nothing; the verification requirement
> governs **how the author established ground truth**, which is exactly what the
> plan's ground-truth policy demands ("Taxonomy identifiers must be checked
> against official sources"). Had the prohibition been read as absolute, the
> identifiers would have had to be left `null` and the taxonomy-correctness
> metric would have been unmeasurable in Phase 3. Flagging the reading
> explicitly so it can be overruled if that was not the intent.

### Nothing came from the agent

No expected value was produced by, copied from, or confirmed against a
Recognition Agent output. **The agent has not been executed against this dataset
at all.** `derived_from_agent_output: false` is recorded in the manifest and
asserted by test.

This mattered concretely. Comparing the officially verified identifiers with the
agent's own development fixture `fixtures/mock_taxonomy.json`:

| Species | Fixture GBIF | **Official GBIF** | Verdict |
| --- | ---: | ---: | --- |
| *Panthera leo* | 5219404 | 5219404 | agrees |
| *Panthera pardus* | 5219436 | 5219436 | agrees |
| *Panthera tigris* | 5219426 | **5219416** | **differs** |
| *Loxodonta africana* | 2433451 | **2435350** | **differs** |
| *Ursus maritimus* | `null` | 2433451 | fixture correctly left it null |
| *Vulpes lagopus* | `null` | 5219303 | fixture correctly left it null |

The fixture's key for *Loxodonta africana*, `2433451`, is in fact **the GBIF key
for *Ursus maritimus***. This is not a defect in the agent: the fixture states in
its own `_fixture_note` that its identifiers "must be verified against the real
databases before any scientific claim is made", and it is used only in mock mode
where the status is reported as `mock_verified` rather than `verified`. But it is
a precise demonstration of why the plan forbids deriving ground truth from the
agent — anyone who had copied those values into the benchmark would have shipped
two wrong expected answers.

**No runtime file was changed in response to this finding.** It is reported, as
Phase 1 requires, not fixed.

---

## 5. Licensing and source coverage

**34 of 36 cases carry an image** (`INV-02` has no image by design, `INV-03`
carries a synthetic remote-URL context). Those 34 draw on **29 distinct source
files**.

| Coverage | Count |
| --- | --- |
| Public `https://` source page recorded | **34 / 34 (100 %)** |
| Public `https://` asset URL recorded | **34 / 34 (100 %)** |
| Attribution recorded | **34 / 34 (100 %)** |
| Free licence recorded | **34 / 34 (100 %)** |
| Creative Commons deed URL recorded | 31 / 31 applicable (100 %) |
| Public-domain works (no deed exists) | 3 |
| Private photographs used | **0** |

| Licence | Cases |
| --- | ---: |
| CC BY-SA 2.0 | 10 |
| CC BY-SA 4.0 | 9 |
| CC BY 2.0 | 4 |
| CC BY 4.0 | 3 |
| CC0 | 3 |
| Public domain | 3 |
| CC BY-SA 3.0 | 1 |
| CC BY-SA 3.0 de | 1 |

Every image was additionally screened against the agent's **own validation
bounds** before selection — accepted MIME type, ≤ 10 MiB, ≤ 25 MP, ≥ 64×64 — so
no case can fail on the size guard instead of on recognition. Asserted by test.

One licence rule was tightened during implementation rather than relaxed. An
initial test demanded a `licence_url` on every asset and failed on eight
public-domain works. Public domain is a *status*, not a licence with a deed, so
demanding a URL would have invited an invented one. The rule now requires a
canonical `creativecommons.org` deed URL for every CC licence, and requires any
asset without one to be **explicitly declared public domain** — which is
stricter, not weaker, because a genuinely missing licence can no longer pass as
an unremarkable gap.

---

## 6. Tests — exact commands and results

Interpreter: `backend/agents/multimodal_recognition_agent/.venv/Scripts/python.exe`
— Python 3.11.0, pytest 8.3.2. Run from the repository root.

### Command 1 — the new Phase 1 dataset tests

```bash
backend/agents/multimodal_recognition_agent/.venv/Scripts/python.exe -m pytest backend/agents/multimodal_recognition_agent/tests/test_sprint4_phase1_evaluation_dataset.py -q
```

```
80 passed in 0.20s
```

| Metric | Value |
| --- | --- |
| Collected | **80** |
| Passed | **80** |
| Failed / Errors / Skipped / xfail | **0 / 0 / 0 / 0** |
| Exit code | **0** |

### Command 2 — the complete Recognition offline suite

```bash
backend/agents/multimodal_recognition_agent/.venv/Scripts/python.exe -m pytest backend/agents/multimodal_recognition_agent/tests -q
```

```
1276 passed, 1 warning in 137.84s (0:02:17)
```

| Metric | Phase 0 baseline | **Phase 1** | Δ |
| --- | ---: | ---: | :--- |
| Collected | 1196 | **1276** | +80 (the new file) |
| Passed | 1196 | **1276** | +80 |
| Failed | 0 | **0** | none |
| Errors | 0 | **0** | none |
| Skipped | 0 | **0** | none |
| xfailed / xpassed | 0 | **0** | none |
| Warnings | 1 | **1** | none (same `StarletteDeprecationWarning`) |
| Exit code | 0 | **0** | none |

**The 1196-test baseline is intact.** The only change to the suite total is the
80 new dataset tests; no pre-existing test changed behaviour, and the single
warning is still FastAPI's test-client deprecation, not Recognition code.

### Requirement-to-test mapping

| # | Required validation | Covering tests |
| ---: | --- | --- |
| 1 | Manifest schema | `test_the_manifest_is_a_json_object_with_the_expected_top_level_shape`, `test_every_case_carries_every_common_field`, `test_every_case_declares_a_known_category`, `test_every_declared_category_is_documented_in_the_manifest` |
| 2 | Minimum 30 total cases | `test_the_benchmark_has_at_least_thirty_cases` |
| 3 | Minimum category counts | `test_each_category_meets_its_minimum` (4 parametrised), `test_every_case_belongs_to_one_of_the_four_counted_categories` |
| 4 | At least 20 real cases | `test_each_category_meets_its_minimum[real_recognition-20]` |
| 5 | At least 10 species | `test_the_real_cases_cover_at_least_ten_distinct_species`, `test_every_species_has_at_least_two_distinct_images`, `test_the_real_cases_include_both_clear_and_challenging_images` |
| 6 | Stable, unique case ids | `test_case_ids_are_unique`, `test_case_ids_match_the_stable_identifier_format`, `test_the_exact_set_of_case_ids_is_unchanged` |
| 7 | Required fields by category | `test_each_category_carries_its_own_required_fields`, `test_real_and_ambiguous_and_delegation_cases_all_reference_an_image`, `test_failure_cases_expect_a_controlled_error_code_and_no_species`, `test_delegation_cases_name_a_capability_and_a_resume_key`, `test_the_resume_case_completes_rather_than_escalating_again` |
| 8 | Source and licence for every real image | `test_every_image_case_carries_complete_source_and_licence_metadata`, `test_every_image_licence_is_a_free_licence`, `test_every_creative_commons_licence_carries_its_deed_url`, `test_an_asset_without_a_deed_url_is_declared_public_domain`, `test_every_image_records_an_attribution`, `test_every_source_page_and_asset_url_is_a_public_https_url` |
| 9 | Scientific-name format | `test_every_expected_species_is_a_well_formed_binomial`, `test_every_accepted_top_k_label_is_a_well_formed_binomial`, `test_the_species_table_records_verified_identifiers_for_every_species`, `test_case_identifiers_agree_with_the_species_table` |
| 10 | Deterministic loading order | `test_loading_the_manifest_twice_yields_the_same_order`, `test_the_loader_does_not_reorder_or_mutate_the_committed_file`, `test_the_categories_appear_in_contiguous_committed_order` |
| 11 | Valid status / decision / capability | `test_every_expected_status_is_one_recognition_can_actually_return`, `test_no_case_expects_the_continue_status`, `test_every_expected_and_accepted_decision_is_a_real_decision_value`, `test_an_expected_decision_is_always_inside_its_own_accepted_set`, `test_every_expected_capability_is_one_the_agent_can_emit`, `test_every_expected_error_code_is_a_real_controlled_code`, `test_completed_cases_never_expect_an_error_code`, `test_every_difficulty_is_a_known_value_or_null` |
| 12 | No Base64, data URLs, private paths | `test_the_manifest_contains_no_inline_image_payload` (4 parametrised), `test_the_manifest_contains_no_absolute_or_private_path` (4 parametrised), `test_no_long_base64_like_blob_hides_anywhere_in_the_manifest`, `test_every_local_asset_path_is_relative_and_inside_the_ignored_directory`, `test_the_assets_directory_is_git_ignored` |
| 13 | No silently skipped or disabled case | `test_no_case_carries_a_skip_or_disable_flag` (6 parametrised), `test_every_case_is_runnable_in_at_least_one_mode`, `test_every_inapplicable_mode_states_a_reason`, `test_every_case_records_how_its_ground_truth_was_established` |
| 14 | Applicability for ambiguous cases | `test_ambiguous_cases_declare_applicable_and_inapplicable_metrics`, `test_ambiguous_cases_never_carry_species_level_ground_truth`, `test_ambiguous_cases_mark_species_metrics_as_not_applicable`, `test_ambiguous_cases_accept_only_inconclusive_decisions`, `test_ambiguous_cases_record_what_kind_of_ambiguity_they_are`, `test_no_metric_is_both_applicable_and_not_applicable` |
| 15 | Bounded, non-training declaration | `test_the_benchmark_is_explicitly_marked_bounded_and_not_training_data`, `test_the_benchmark_note_says_what_it_is_and_is_not`, `test_the_prohibited_uses_are_recorded_explicitly` (4 parametrised), `test_the_ground_truth_sources_are_recorded_and_are_not_the_agent`, `test_the_thresholds_are_named_so_they_cannot_be_quietly_moved`, `test_the_evaluation_package_contains_no_runner` |

**No live Azure, BioCLIP-2, GBIF or NCBI call occurred during either test run.**
Both suites are offline by construction.

---

## 7. Applicability declarations

Every case declares whether it is runnable offline, live, or both — and any
"no" must state a reason, asserted by test, so a case cannot be quietly excused
from a run.

| Mode | Cases | Note |
| --- | ---: | --- |
| Offline-applicable | 4 | The four failure cases; they need no classifier verdict |
| Live-applicable | 35 | Everything except `DEP-01` |
| Both | 3 | `INV-01`, `INV-02`, `INV-03` |

The 32 image-recognition, ambiguous and delegation cases are **live-only**, and
the recorded reason is concrete: the offline mock classifier is a SHA-256 oracle
holding exactly **three** fixture digests, so it returns no candidates for any of
these images. Running them offline would produce `not_identified` for every one
and score them as passes *for the wrong reason* — a false green that would
misrepresent the agent.

`DEP-01` is **offline-only** for the mirror-image reason: exercising it live
would mean deliberately breaking a real provider call, and its behaviour is
already fully determined by an injected transport.

---

## 8. Runtime behaviour untouched

| Check | Evidence |
| --- | --- |
| Recognition runtime files changed | **0** — `git status` lists only the new `evaluation/` package, the new test file and this report |
| `requirements.txt` changed | **No** |
| Thresholds, prompts, providers, workflow changed | **No** — no file under `adapters/`, `domain/`, `workflows/`, `config.py`, `validation.py`, `text_analysis.py`, `agent.py`, `api.py` or `schema.py` was touched |
| Orchestrator changed | **No** |
| Agent executed against the benchmark | **No** |
| RAG / RAGAS / Qdrant / embeddings added | **No** |
| XAI added | **No** |
| LangSmith added | **No** |
| Phase 2 runner implemented | **No** — asserted by `test_the_evaluation_package_contains_no_runner`, which fails if any module in `evaluation/` references `RecognitionAgent` or builds an `AgentRequest` |
| Baseline suite intact | 1196 pre-existing tests still pass, unchanged |

The evaluation package imports from the agent's contracts (read-only) and has no
side effects on import.

---

## 9. No image, private path or secret is tracked

| Check | Result |
| --- | --- |
| `git add -A --dry-run` | Lists exactly **7 files**: `.gitignore`, `README.md`, `__init__.py`, `fetch_assets.py`, `manifest.json`, `manifest_schema.py`, and the test module. **No image, no `assets/`.** |
| `assets/` ignore rule | Proven live: a real file written to `evaluation/assets/probe.jpg` was invisible to `git status`, and `git check-ignore -v` attributed it to `evaluation/.gitignore:8`. Probe deleted. |
| Image files anywhere under the agent | **None** untracked or tracked (`git status --porcelain --ignored` filtered on image extensions returns nothing) |
| Base64 / data URLs in the manifest | **None** — asserted four ways, plus a scan for any unprefixed 200-character base64-like run |
| Absolute or private paths in the manifest | **None** — asserted against Windows drive, `/home/`, `/Users/` and UNC patterns |
| Credentials / `.env` content | **None** — a `grep` for key, token, secret, password and PEM markers across every new file returns nothing |
| Private photographs | **None** — every image is a public Commons file under a free licence |

---

## 10. Phase 1 gate

| Gate criterion | Verdict | Basis |
| --- | --- | --- |
| At least 30 cases satisfying the schema | **PASS** | 36 cases; 80 schema tests pass |
| Minimum per-category counts | **PASS** | 24 / 5 / 4 / 3 against 20 / 5 / 3 / 2 |
| At least 10 species, ≥2 distinct images each | **PASS** | 12 species, 2 distinct source files each, asserted by test |
| Clear and challenging images both present | **PASS** | 11 clear, 13 challenging |
| Every real image has legal source metadata | **PASS** | 34/34 with source page, asset URL, free licence and attribution |
| Ground truth reviewed independently of agent output | **PASS** | Commons source pages + visual inspection of all 48 candidates; GBIF and NCBI checked at the official sources; `derived_from_agent_output: false` asserted |
| Ambiguous cases not forced into species scoring | **PASS** | No species, no Top-K, no identifiers; all four species metrics declared not applicable |
| Invalid/dependency cases are behavioural | **PASS** | Status + controlled `error_code` only; no species expected |
| All dataset tests pass | **PASS** | 80/80 |
| Complete Recognition offline suite passes | **PASS** | 1276/1276, baseline 1196 intact |
| Labelled a bounded Sprint 4 benchmark, not an accuracy dataset | **PASS** | `bounded: true`, `is_training_data: false`, `may_tune_thresholds: false`, `may_tune_prompts: false`, plus the `_benchmark_note` naming the thresholds it must never move |
| No runtime file changed | **PASS** | §8 |
| No image, private path or secret tracked | **PASS** | §9 |

### Decision

**PHASE 1 — PASS.**

No stop condition was triggered: every licence was verified, every image identity
was confirmed by direct inspection, every official taxonomy identifier resolved
cleanly, and no runtime contract contradicted the plan.

Two items are handed forward rather than acted on:

1. **The mock fixture's GBIF keys for *Panthera tigris* and *Loxodonta
   africana* do not match GBIF** (§4), and the *Loxodonta* value is in fact
   *Ursus maritimus*'s key. Disclosed by the fixture's own note and confined to
   mock mode, where the status is `mock_verified` rather than `verified`. Left
   untouched — Phase 1 may not change runtime data.
2. **32 of 36 cases are live-only** (§7), because the three-digest mock oracle
   cannot produce a meaningful verdict for a real photograph. Phase 2's offline
   dry run must therefore exercise the runner and the evaluators with injected
   providers, and must not be mistaken for a measurement of recognition quality.

No evaluation runner was implemented. No evaluator, scorer, metric or report
generator was written. The agent was not executed against the benchmark. Nothing
was committed or pushed. **Phase 2 has not been started.**

---

*Compiled 2026-08-25 on branch `group-d-recognition-sprint4` at
`132dac43e8ea0e5558517d5372c8a3e5074f782f`. No Recognition runtime file was
modified. No API key, Azure endpoint, `.env` value, image payload, Base64 data or
private path appears anywhere in this document or in the dataset it describes.*
