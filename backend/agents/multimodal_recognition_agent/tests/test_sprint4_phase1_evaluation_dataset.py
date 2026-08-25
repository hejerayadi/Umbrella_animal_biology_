"""Sprint 4 Phase 1: validation of the bounded evaluation benchmark.

These tests validate the *dataset*, not the agent. They never construct
`RecognitionAgent`, never execute the workflow and never touch a network: every
assertion below is made against `evaluation/manifest.json` and against
vocabularies imported from the agent's own runtime contracts.

Two properties are worth stating up front, because they are the ones that make
the benchmark trustworthy rather than merely present:

1. **The accepted values are derived, not restated.** `manifest_schema` reads
   the statuses out of `schema.AgentStatus`, the decisions out of the `Decision`
   Literal, the error codes out of `ErrorCode`, and the delegation capabilities
   out of `HELPER_OUTPUT_KEYS`. A manifest cannot encode an expectation the
   agent is incapable of producing.

2. **No image payload can enter the repository.** The manifest carries a public
   source page and a public asset URL per case; the bytes live in a git-ignored
   directory. A test below scans the raw manifest text for Base64, data URLs and
   absolute paths, so a well-meaning future edit that pastes an image in fails
   loudly.
"""
from __future__ import annotations

import json
import re

import pytest

from ..evaluation import manifest_schema as ms

MANIFEST = ms.load_manifest()
CASES = ms.cases(MANIFEST)

# The complete, frozen set of case ids. Pinned as a literal so that renaming,
# renumbering or quietly dropping a case is a test failure rather than a silent
# change to what Sprint 4 measured. Phase 3 results are reported against these
# ids, so they have to mean the same thing later that they mean today.
EXPECTED_CASE_IDS = (
    "REC-PLEO-01", "REC-PLEO-02", "REC-PTIG-01", "REC-PTIG-02",
    "REC-PPAR-01", "REC-PPAR-02", "REC-UMAR-01", "REC-UMAR-02",
    "REC-LAFR-01", "REC-LAFR-02", "REC-GCAM-01", "REC-GCAM-02",
    "REC-EQUA-01", "REC-EQUA-02", "REC-AMEL-01", "REC-AMEL-02",
    "REC-VLAG-01", "REC-VLAG-02", "REC-PROS-01", "REC-PROS-02",
    "REC-AFOR-01", "REC-AFOR-02", "REC-BSCA-01", "REC-BSCA-02",
    "AMB-01", "AMB-02", "AMB-03", "AMB-04", "AMB-05",
    "INV-01", "INV-02", "INV-03", "DEP-01",
    "DEL-01", "DEL-02", "DEL-03",
)

SCIENTIFIC_NAME = re.compile(r"^[A-Z][a-z]+ [a-z][a-z-]+$")
CASE_ID = re.compile(r"^(REC-[A-Z]{4}-\d{2}|AMB-\d{2}|INV-\d{2}|DEP-\d{2}|DEL-\d{2})$")


def real_cases():
    return [c for c in CASES if c["category"] == "real_recognition"]


# --- 1. manifest schema -----------------------------------------------------

def test_the_manifest_is_a_json_object_with_the_expected_top_level_shape():
    for key in ("_benchmark_note", "benchmark_id", "species", "categories", "cases",
                "ground_truth_sources", "permitted_use", "prohibited_use"):
        assert key in MANIFEST, f"manifest is missing top-level key {key!r}"
    assert isinstance(MANIFEST["cases"], list)
    assert isinstance(MANIFEST["species"], dict)


def test_every_case_carries_every_common_field():
    """Missing is different from null. A field that is absent means nobody
    decided; a field that is present and null means somebody decided it does not
    apply. Only the second is acceptable in a benchmark."""
    for case in CASES:
        missing = [f for f in ms.COMMON_FIELDS if f not in case]
        assert not missing, f"{case.get('case_id')} is missing fields: {missing}"


def test_every_case_declares_a_known_category():
    for case in CASES:
        assert case["category"] in ms.VALID_CATEGORIES, (
            f"{case['case_id']} has unknown category {case['category']!r}"
        )


def test_every_declared_category_is_documented_in_the_manifest():
    for category in ms.VALID_CATEGORIES:
        assert category in MANIFEST["categories"], f"{category} has no description"


# --- 2. minimum 30 total cases ---------------------------------------------

def test_the_benchmark_has_at_least_thirty_cases():
    assert len(CASES) >= ms.MIN_TOTAL_CASES, (
        f"Sprint 4 requires at least {ms.MIN_TOTAL_CASES} cases, found {len(CASES)}"
    )


# --- 3. minimum category counts / 4. at least 20 real cases ----------------

@pytest.mark.parametrize(
    "category,minimum",
    [
        ("real_recognition", ms.MIN_REAL_CASES),
        ("ambiguous_or_non_animal", ms.MIN_AMBIGUOUS_CASES),
        ("invalid_input_or_dependency_failure", ms.MIN_FAILURE_CASES),
        ("delegation_resume", ms.MIN_DELEGATION_CASES),
    ],
)
def test_each_category_meets_its_minimum(category, minimum):
    found = len(ms.cases_by_category(category, MANIFEST))
    assert found >= minimum, f"{category}: need >= {minimum}, found {found}"


def test_every_case_belongs_to_one_of_the_four_counted_categories():
    """No case may sit outside the counted categories, which is how a case could
    otherwise exist in the file while contributing to no minimum at all."""
    counted = sum(
        len(ms.cases_by_category(c, MANIFEST)) for c in ms.VALID_CATEGORIES
    )
    assert counted == len(CASES)


# --- 5. at least 10 species -------------------------------------------------

def test_the_real_cases_cover_at_least_ten_distinct_species():
    species = {c["expected_species"] for c in real_cases()}
    assert None not in species, "a real recognition case has no expected species"
    assert len(species) >= ms.MIN_DISTINCT_SPECIES, (
        f"need >= {ms.MIN_DISTINCT_SPECIES} species, found {len(species)}: {sorted(species)}"
    )


def test_every_species_has_at_least_two_distinct_images():
    """The plan asks for two images per species where feasible. Two *distinct*
    ones: the same file twice would measure nothing new."""
    by_species: dict[str, set[str]] = {}
    for case in real_cases():
        by_species.setdefault(case["expected_species"], set()).add(case["asset"]["asset_url"])
    thin = {sp: len(urls) for sp, urls in by_species.items() if len(urls) < 2}
    assert not thin, f"species with fewer than two distinct images: {thin}"


def test_the_real_cases_include_both_clear_and_challenging_images():
    difficulties = {c["difficulty"] for c in real_cases()}
    assert "clear" in difficulties and "challenging" in difficulties, (
        f"expected both clear and challenging real images, found {difficulties}"
    )


def test_every_species_used_by_a_case_is_declared_in_the_species_table():
    declared = set(MANIFEST["species"])
    for case in CASES:
        if case["expected_species"]:
            assert case["expected_species"] in declared, (
                f"{case['case_id']} uses undeclared species {case['expected_species']!r}"
            )


# --- 6. stable and unique case ids -----------------------------------------

def test_case_ids_are_unique():
    ids = [c["case_id"] for c in CASES]
    duplicates = {i for i in ids if ids.count(i) > 1}
    assert not duplicates, f"duplicate case ids: {sorted(duplicates)}"


def test_case_ids_match_the_stable_identifier_format():
    for case in CASES:
        assert CASE_ID.match(case["case_id"]), f"unstable case id {case['case_id']!r}"


def test_the_exact_set_of_case_ids_is_unchanged():
    assert tuple(c["case_id"] for c in CASES) == EXPECTED_CASE_IDS


# --- 7. required fields by category ----------------------------------------

def test_each_category_carries_its_own_required_fields():
    for case in CASES:
        for field in ms.REQUIRED_BY_CATEGORY[case["category"]]:
            assert field in case, (
                f"{case['case_id']} ({case['category']}) is missing {field!r}"
            )


def test_real_and_ambiguous_and_delegation_cases_all_reference_an_image():
    for case in CASES:
        if case["category"] in ("real_recognition", "ambiguous_or_non_animal",
                                "delegation_resume"):
            assert case["asset"], f"{case['case_id']} needs an image and has none"


def test_failure_cases_expect_a_controlled_error_code_and_no_species():
    for case in ms.cases_by_category("invalid_input_or_dependency_failure", MANIFEST):
        assert case["expected_status"] == "failed"
        assert case["expected_error_code"] in ms.VALID_ERROR_CODES
        assert case["expected_species"] is None, (
            f"{case['case_id']} is a failure case and must not expect a species"
        )
        assert case["expected_decision"] is None


def test_delegation_cases_name_a_capability_and_a_resume_key():
    escalating = [
        c for c in ms.cases_by_category("delegation_resume", MANIFEST)
        if c["expected_status"] == "needs_agent"
    ]
    assert escalating, "no escalating delegation case exists"
    for case in escalating:
        assert case["expected_delegation_capability"] in ms.VALID_CAPABILITIES
        assert case["resume_context_key"] in ms.VALID_RESUME_KEYS
        # The delegation node only fires on an established identification.
        assert case["expected_decision"] == "identified", (
            f"{case['case_id']}: make_delegation_node cannot delegate unless the "
            "decision is 'identified'"
        )


def test_the_resume_case_completes_rather_than_escalating_again():
    resumes = [
        c for c in ms.cases_by_category("delegation_resume", MANIFEST)
        if c.get("resume_context_present")
    ]
    assert resumes, "the resume half of the delegation contract is not covered"
    for case in resumes:
        assert case["expected_status"] == "completed"
        assert case["expected_delegation_capability"] is None


# --- 8. source and licence metadata for every real image -------------------

def test_every_image_case_carries_complete_source_and_licence_metadata():
    for case in CASES:
        asset = case["asset"]
        if not asset:
            continue
        for field in ms.ASSET_FIELDS:
            assert asset.get(field) not in (None, ""), (
                f"{case['case_id']} asset is missing {field!r}"
            )


def test_every_image_licence_is_a_free_licence():
    for case in CASES:
        asset = case["asset"]
        if not asset:
            continue
        licence = asset["licence"].lower()
        assert licence.startswith(ms.ACCEPTED_LICENCE_PREFIXES), (
            f"{case['case_id']} has unverified licence {asset['licence']!r}"
        )


def test_every_creative_commons_licence_carries_its_deed_url():
    """A CC licence has a deed, so the manifest must link it. Public-domain works
    are exempt because there is no deed to link - demanding one would invite an
    invented URL, which is worse than an absent one."""
    for case in CASES:
        asset = case["asset"]
        if not asset:
            continue
        licence = asset["licence"].lower()
        if licence.startswith(ms.LICENCES_REQUIRING_A_DEED_URL):
            # Canonical CC deed URLs are published as http://creativecommons.org/...
            # and are identifiers rather than fetch targets, so both schemes count.
            # Nothing in this repository ever requests them.
            deed = asset.get("licence_url") or ""
            assert deed.startswith(("https://creativecommons.org/",
                                    "http://creativecommons.org/")), (
                f"{case['case_id']} is {asset['licence']} but records no CC deed URL "
                f"(found {deed!r})"
            )


def test_an_asset_without_a_deed_url_is_declared_public_domain():
    """The only acceptable reason to have no licence URL. This is what stops a
    missing licence from passing as an unremarkable gap."""
    for case in CASES:
        asset = case["asset"]
        if not asset or asset.get("licence_url"):
            continue
        assert asset["licence"].lower().startswith(ms.PUBLIC_DOMAIN_LICENCES), (
            f"{case['case_id']} has neither a deed URL nor a public-domain "
            f"declaration - its licence is {asset['licence']!r} and is unverified"
        )


def test_every_image_records_an_attribution():
    """Required for CC BY and CC BY-SA, and recorded for everything else so the
    creator is credited regardless of whether the licence compels it."""
    for case in CASES:
        asset = case["asset"]
        if not asset:
            continue
        assert asset["attribution"].strip(), f"{case['case_id']} credits nobody"


def test_every_source_page_and_asset_url_is_a_public_https_url():
    for case in CASES:
        asset = case["asset"]
        if not asset:
            continue
        for field in ("source_page", "asset_url"):
            assert asset[field].startswith("https://"), (
                f"{case['case_id']}.{field} is not a public https URL"
            )


def test_every_image_media_type_is_one_the_agent_accepts():
    for case in CASES:
        asset = case["asset"]
        if not asset:
            continue
        assert asset["media_type"] in ms.VALID_MEDIA_TYPES, (
            f"{case['case_id']} declares media type {asset['media_type']!r}, which "
            "validation.py would reject outright"
        )


def test_every_image_fits_inside_the_agents_own_validation_bounds():
    """A case the agent would refuse on size alone measures the size guard, not
    recognition. These bounds are the defaults in config.from_env."""
    for case in CASES:
        asset = case["asset"]
        if not asset:
            continue
        assert asset["bytes"] <= 10_485_760, f"{case['case_id']} exceeds MAX_IMAGE_BYTES"
        assert asset["width"] * asset["height"] <= 25_000_000, (
            f"{case['case_id']} exceeds MAX_IMAGE_PIXELS"
        )
        assert asset["width"] >= 64 and asset["height"] >= 64, (
            f"{case['case_id']} is below the minimum accepted dimensions"
        )


# --- 9. scientific-name format ---------------------------------------------

def test_every_expected_species_is_a_well_formed_binomial():
    for case in CASES:
        name = case["expected_species"]
        if name is None:
            continue
        assert SCIENTIFIC_NAME.match(name), f"{case['case_id']}: {name!r} is not a binomial"


def test_every_accepted_top_k_label_is_a_well_formed_binomial():
    for case in CASES:
        for label in case["accepted_top_k_labels"] or ():
            assert SCIENTIFIC_NAME.match(label), (
                f"{case['case_id']}: accepted label {label!r} is not a binomial"
            )


def test_the_species_table_records_verified_identifiers_for_every_species():
    for name, record in MANIFEST["species"].items():
        assert SCIENTIFIC_NAME.match(name), f"{name!r} is not a binomial"
        assert isinstance(record["gbif_id"], int) and record["gbif_id"] > 0
        assert isinstance(record["ncbi_taxid"], int) and record["ncbi_taxid"] > 0
        assert record["gbif_match_type"] == "EXACT"
        assert record["gbif_status"] == "ACCEPTED"
        assert record["verified_on"]


def test_case_identifiers_agree_with_the_species_table():
    """A case may omit an identifier, but it may never contradict the verified
    one - that is how a typo becomes a wrong 'expected' answer."""
    table = MANIFEST["species"]
    for case in CASES:
        name = case["expected_species"]
        if not name:
            continue
        if case["expected_gbif_id"] is not None:
            assert case["expected_gbif_id"] == table[name]["gbif_id"], case["case_id"]
        if case["expected_ncbi_taxid"] is not None:
            assert case["expected_ncbi_taxid"] == table[name]["ncbi_taxid"], case["case_id"]


# --- 10. deterministic loading order ---------------------------------------

def test_loading_the_manifest_twice_yields_the_same_order():
    first = [c["case_id"] for c in ms.cases(ms.load_manifest())]
    second = [c["case_id"] for c in ms.cases(ms.load_manifest())]
    assert first == second


def test_the_loader_does_not_reorder_or_mutate_the_committed_file():
    raw = json.load(open(ms.MANIFEST_PATH, encoding="utf-8"))
    assert [c["case_id"] for c in raw["cases"]] == [c["case_id"] for c in CASES]


def test_the_categories_appear_in_contiguous_committed_order():
    """Order is part of the contract: Phase 3 reports results positionally as
    well as by id, so cases must not drift between runs."""
    order = [c["category"] for c in CASES]
    assert order == sorted(order, key=lambda c: order.index(c)), (
        "cases of the same category are not contiguous"
    )


# --- 11. valid status, decision and capability values ----------------------

def test_every_expected_status_is_one_recognition_can_actually_return():
    for case in CASES:
        assert case["expected_status"] in ms.VALID_STATUSES, (
            f"{case['case_id']} expects status {case['expected_status']!r}"
        )


def test_no_case_expects_the_continue_status():
    """Recognition never constructs CONTINUE. A case expecting it could never pass."""
    assert all(c["expected_status"] != "continue" for c in CASES)


def test_every_expected_and_accepted_decision_is_a_real_decision_value():
    for case in CASES:
        if case["expected_decision"] is not None:
            assert case["expected_decision"] in ms.VALID_DECISIONS, case["case_id"]
        for decision in case["accepted_decisions"]:
            assert decision in ms.VALID_DECISIONS, (
                f"{case['case_id']} accepts unknown decision {decision!r}"
            )


def test_an_expected_decision_is_always_inside_its_own_accepted_set():
    for case in CASES:
        if case["expected_decision"] is not None:
            assert case["expected_decision"] in case["accepted_decisions"], (
                f"{case['case_id']} expects a decision it does not accept"
            )


def test_every_expected_capability_is_one_the_agent_can_emit():
    for case in CASES:
        capability = case["expected_delegation_capability"]
        if capability is not None:
            assert capability in ms.VALID_CAPABILITIES, (
                f"{case['case_id']} expects capability {capability!r}, which is not a "
                "key of HELPER_OUTPUT_KEYS"
            )


def test_every_expected_error_code_is_a_real_controlled_code():
    for case in CASES:
        code = case["expected_error_code"]
        if code is not None:
            assert code in ms.VALID_ERROR_CODES, (
                f"{case['case_id']} expects unknown error code {code!r}"
            )


def test_completed_cases_never_expect_an_error_code():
    for case in CASES:
        if case["expected_status"] == "completed":
            assert case["expected_error_code"] is None, case["case_id"]


def test_every_difficulty_is_a_known_value_or_null():
    for case in CASES:
        if case["difficulty"] is not None:
            assert case["difficulty"] in ms.VALID_DIFFICULTIES, case["case_id"]


# --- 12. no Base64, data URLs or private absolute paths --------------------

RAW_MANIFEST_TEXT = ms.MANIFEST_PATH.read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "forbidden",
    ["data:image", ";base64,", "data:application", "-----BEGIN"],
)
def test_the_manifest_contains_no_inline_image_payload(forbidden):
    assert forbidden not in RAW_MANIFEST_TEXT, (
        f"the manifest contains {forbidden!r}: image payloads must never be committed"
    )


@pytest.mark.parametrize(
    "pattern",
    [r"[A-Za-z]:\\\\", r"/home/[a-z]", r"/Users/[A-Za-z]", r"\\\\\\\\[A-Za-z]"],
)
def test_the_manifest_contains_no_absolute_or_private_path(pattern):
    hits = re.findall(pattern, RAW_MANIFEST_TEXT)
    assert not hits, f"the manifest contains absolute/private paths: {hits[:3]}"


def test_every_local_asset_path_is_relative_and_inside_the_ignored_directory():
    for case in CASES:
        asset = case["asset"]
        if not asset:
            continue
        path = asset["local_path"]
        assert path.startswith(ms.ASSETS_DIRNAME + "/"), (
            f"{case['case_id']} local_path {path!r} is outside the ignored assets directory"
        )
        assert ".." not in path, f"{case['case_id']} local_path escapes with '..'"


def test_no_long_base64_like_blob_hides_anywhere_in_the_manifest():
    """A payload could be pasted without the `data:` prefix. Nothing legitimate
    in this file is a 200-character run of base64 characters."""
    blobs = re.findall(r"[A-Za-z0-9+/]{200,}={0,2}", RAW_MANIFEST_TEXT)
    assert not blobs, f"{len(blobs)} base64-like blob(s) found in the manifest"


def test_the_assets_directory_is_git_ignored():
    gitignore = (ms.MANIFEST_PATH.parent / ".gitignore").read_text(encoding="utf-8")
    assert "assets/" in gitignore


# --- 13. no silently skipped or disabled case ------------------------------

@pytest.mark.parametrize("flag", ["skip", "skipped", "disabled", "xfail", "ignore", "todo"])
def test_no_case_carries_a_skip_or_disable_flag(flag):
    for case in CASES:
        assert flag not in case, (
            f"{case['case_id']} carries {flag!r}: a case is either in the benchmark "
            "and reported, or removed and explained - never quietly switched off"
        )


def test_every_case_is_runnable_in_at_least_one_mode():
    for case in CASES:
        applicability = case["applicability"]
        assert applicability["offline"] or applicability["live"], (
            f"{case['case_id']} is applicable in neither offline nor live mode, so it "
            "would never run and never be reported"
        )


def test_every_inapplicable_mode_states_a_reason():
    """'Not applicable' has to be argued, not asserted, or a case can be quietly
    excused from the run without anybody noticing why."""
    for case in CASES:
        applicability = case["applicability"]
        if not applicability["offline"]:
            assert applicability.get("offline_reason"), (
                f"{case['case_id']} is offline-inapplicable with no reason given"
            )
        if not applicability["live"]:
            assert applicability.get("live_reason"), (
                f"{case['case_id']} is live-inapplicable with no reason given"
            )


def test_every_case_records_how_its_ground_truth_was_established():
    for case in CASES:
        assert case["ground_truth_basis"].strip(), (
            f"{case['case_id']} has no ground-truth basis recorded"
        )
        assert len(case["ground_truth_basis"]) > 40, (
            f"{case['case_id']} ground-truth basis is too thin to audit"
        )


# --- 14. applicability declarations for ambiguous cases --------------------

def test_ambiguous_cases_declare_applicable_and_inapplicable_metrics():
    for case in ms.cases_by_category("ambiguous_or_non_animal", MANIFEST):
        applicability = case["applicability"]
        assert applicability["metrics_applicable"], case["case_id"]
        assert applicability["metrics_not_applicable"], case["case_id"]


def test_ambiguous_cases_never_carry_species_level_ground_truth():
    """The plan is explicit: ambiguous and non-animal images must not be forced
    into exact-species scoring."""
    for case in ms.cases_by_category("ambiguous_or_non_animal", MANIFEST):
        assert case["expected_species"] is None, case["case_id"]
        assert case["accepted_top_k_labels"] is None, case["case_id"]
        assert case["expected_gbif_id"] is None, case["case_id"]
        assert case["expected_ncbi_taxid"] is None, case["case_id"]


def test_ambiguous_cases_mark_species_metrics_as_not_applicable():
    for case in ms.cases_by_category("ambiguous_or_non_animal", MANIFEST):
        not_applicable = case["applicability"]["metrics_not_applicable"]
        for metric in ("top1_species", "top5_species", "gbif_identifier", "ncbi_taxid"):
            assert metric in not_applicable, (
                f"{case['case_id']} does not declare {metric} inapplicable"
            )


def test_ambiguous_cases_accept_only_inconclusive_decisions():
    for case in ms.cases_by_category("ambiguous_or_non_animal", MANIFEST):
        assert set(case["accepted_decisions"]) <= {"uncertain", "not_identified"}, (
            f"{case['case_id']} would accept a confident identification of an image "
            "with no identifiable animal"
        )
        assert case["expected_status"] == "completed", (
            f"{case['case_id']}: an inconclusive answer is a completed outcome, not a failure"
        )


def test_ambiguous_cases_record_what_kind_of_ambiguity_they_are():
    kinds = {c["ambiguity_kind"] for c in ms.cases_by_category("ambiguous_or_non_animal", MANIFEST)}
    assert all(kinds), "an ambiguous case has no ambiguity_kind"
    assert len(kinds) >= 2, f"ambiguity kinds are not varied enough: {kinds}"


def test_no_metric_is_both_applicable_and_not_applicable():
    for case in CASES:
        applicability = case["applicability"]
        overlap = set(applicability["metrics_applicable"]) & set(
            applicability["metrics_not_applicable"]
        )
        assert not overlap, f"{case['case_id']} declares {overlap} both ways"


# --- 15. bounded, non-training declaration ---------------------------------

def test_the_benchmark_is_explicitly_marked_bounded_and_not_training_data():
    assert MANIFEST["bounded"] is True
    assert MANIFEST["is_training_data"] is False
    assert MANIFEST["may_tune_thresholds"] is False
    assert MANIFEST["may_tune_prompts"] is False
    assert MANIFEST["derived_from_agent_output"] is False


def test_the_benchmark_note_says_what_it_is_and_is_not():
    note = MANIFEST["_benchmark_note"].upper()
    for phrase in ("BOUNDED", "NOT A TRAINING", "SPRINT 4"):
        assert phrase in note, f"the benchmark note does not state {phrase!r}"


@pytest.mark.parametrize(
    "prohibited", ["training", "fine-tuning", "prompt tuning", "threshold tuning"]
)
def test_the_prohibited_uses_are_recorded_explicitly(prohibited):
    assert prohibited in MANIFEST["prohibited_use"]


def test_the_ground_truth_sources_are_recorded_and_are_not_the_agent():
    sources = MANIFEST["ground_truth_sources"]
    for key in ("species_identity", "gbif", "ncbi", "verified_on"):
        assert sources.get(key), f"ground_truth_sources is missing {key!r}"
    blob = json.dumps(sources).lower()
    assert "recognitionagent" not in blob and "agent output" not in blob


def test_the_thresholds_are_named_so_they_cannot_be_quietly_moved():
    """The note pins the exact gate values the dataset must never be used to
    change. If somebody does change them, this reads as a deliberate act."""
    note = MANIFEST["_benchmark_note"]
    for threshold in ("0.75", "0.08", "0.45"):
        assert threshold in note


# The dataset modules. Phase 2 added a runner to this package, which is what
# Phase 2 is for - but these two describe and fetch data, and a call into the
# agent from either of them would mean the dataset had started depending on the
# thing it exists to measure.
DATASET_ONLY_MODULES = ("manifest_schema.py", "fetch_assets.py")


def test_the_dataset_modules_never_call_the_agent():
    """Originally this asserted the whole package contained no runner, which held
    until Phase 2 legitimately added one. The property worth keeping is narrower
    and permanent: the manifest loader and the asset fetcher must stay free of
    the agent, so loading or fetching the benchmark can never execute it."""
    package = ms.MANIFEST_PATH.parent
    for name in DATASET_ONLY_MODULES:
        source = (package / name).read_text(encoding="utf-8")
        assert "RecognitionAgent" not in source.replace("`RecognitionAgent`", ""), (
            f"{name} references RecognitionAgent"
        )
        assert "AgentRequest(" not in source, f"{name} builds an AgentRequest"


def test_the_dataset_modules_import_nothing_that_can_execute_the_agent():
    package = ms.MANIFEST_PATH.parent
    for name in DATASET_ONLY_MODULES:
        source = (package / name).read_text(encoding="utf-8")
        for forbidden in ("from ..agent import", "from .runner import", "import runner"):
            assert forbidden not in source, f"{name} imports {forbidden!r}"
