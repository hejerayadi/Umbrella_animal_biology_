"""Accepted vocabularies for the Sprint 4 benchmark manifest, plus its loader.

Every vocabulary below is **derived from the agent's own runtime contracts**
rather than restated by hand. That is the whole point of this module: a manifest
cannot name a status, a decision, an error code, a delegation capability or a
media type that the agent does not actually produce, because the accepted sets
are read out of `schema.py`, `domain/models.py`, `domain/errors.py`,
`workflows/state.py` and `config.py` at import time.

If a future change removes a decision value or renames an error code, the
dataset tests fail immediately and loudly, instead of the manifest quietly
encoding an expectation the agent can never satisfy.

This module performs no I/O beyond reading `manifest.json`, contacts nothing,
and never imports or constructs `RecognitionAgent`.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, get_args

from ..config import ALLOWED_MEDIA_TYPES, RECOGNITION_IMAGE_CONTEXT_KEY
from ..domain.errors import ErrorCode
from ..domain.models import Decision
from ..schema import AgentStatus
from ..workflows.state import HELPER_OUTPUT_KEYS

MANIFEST_PATH = Path(__file__).resolve().parent / "manifest.json"
ASSETS_DIRNAME = "assets"

# --- vocabularies, all derived ---------------------------------------------

#: Statuses Recognition can actually return. `continue` is excluded on purpose:
#: no code path in this agent constructs it, so no case may expect it.
VALID_STATUSES: frozenset[str] = frozenset(
    s.value for s in AgentStatus if s is not AgentStatus.CONTINUE
)

#: `identified` / `uncertain` / `not_identified`, straight off the Literal.
VALID_DECISIONS: frozenset[str] = frozenset(get_args(Decision))

#: Every controlled refusal code, plus the one string literal the HTTP boundary
#: uses that is not (yet) an ErrorCode member. See api.py and the Phase 0 report.
VALID_ERROR_CODES: frozenset[str] = frozenset(
    [code.value for code in ErrorCode] + ["INTERNAL_ERROR"]
)

#: The capabilities Recognition may name in `target_agent`, which are exactly
#: the keys of the resume map and exactly valid registry agent names.
VALID_CAPABILITIES: frozenset[str] = frozenset(HELPER_OUTPUT_KEYS)

#: The context keys that satisfy the resume half of the delegation contract.
VALID_RESUME_KEYS: frozenset[str] = frozenset(HELPER_OUTPUT_KEYS.values())

VALID_MEDIA_TYPES: frozenset[str] = frozenset(ALLOWED_MEDIA_TYPES)

#: The manifest's own category vocabulary. Unlike the sets above this one is
#: local to the benchmark - the agent has no notion of a case category.
VALID_CATEGORIES: frozenset[str] = frozenset(
    {
        "real_recognition",
        "ambiguous_or_non_animal",
        "invalid_input_or_dependency_failure",
        "delegation_resume",
    }
)

VALID_DIFFICULTIES: frozenset[str] = frozenset({"clear", "challenging"})

#: Minimums fixed by the Sprint 4 plan.
MIN_TOTAL_CASES = 30
MIN_REAL_CASES = 20
MIN_AMBIGUOUS_CASES = 5
MIN_FAILURE_CASES = 3
MIN_DELEGATION_CASES = 2
MIN_DISTINCT_SPECIES = 10

#: Licence prefixes accepted for a committed source reference.
ACCEPTED_LICENCE_PREFIXES = ("cc0", "cc by", "public domain", "pd")

#: Fields every case carries regardless of category.
COMMON_FIELDS = (
    "case_id",
    "category",
    "instruction",
    "asset",
    "expected_status",
    "expected_decision",
    "accepted_decisions",
    "expected_species",
    "accepted_top_k_labels",
    "expected_gbif_id",
    "expected_ncbi_taxid",
    "expected_delegation_capability",
    "expected_error_code",
    "difficulty",
    "ground_truth_basis",
    "notes",
    "applicability",
)

#: Extra fields required per category.
REQUIRED_BY_CATEGORY: dict[str, tuple[str, ...]] = {
    "real_recognition": (),
    "ambiguous_or_non_animal": ("ambiguity_kind",),
    "invalid_input_or_dependency_failure": ("failure_kind", "asset_mode"),
    "delegation_resume": ("resume_context_key",),
}

#: Fields inside `asset` that must be present and non-empty for every image.
#:
#: `licence_url` is deliberately NOT in this list. A Creative Commons licence has
#: a deed URL and must carry it; a public-domain work has no deed to link to,
#: because public domain is a status rather than a licence. Demanding a URL for
#: a PD file would push whoever maintains this dataset towards inventing one.
#: The stricter, correct rule is enforced separately: see
#: `LICENCES_REQUIRING_A_DEED_URL` and the licence tests.
ASSET_FIELDS = (
    "local_path",
    "source_page",
    "asset_url",
    "licence",
    "attribution",
    "media_type",
    "width",
    "height",
    "bytes",
)

#: Licence families that must carry a deed URL, because one exists.
LICENCES_REQUIRING_A_DEED_URL = ("cc0", "cc by")

#: Licence short names accepted without a deed URL.
PUBLIC_DOMAIN_LICENCES = ("public domain", "pd")


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    """Read the manifest exactly as committed. No normalisation, no defaults.

    Deliberately does not sort, filter or repair anything: a test that wants to
    prove the order is deterministic has to see the real order, and a test that
    wants to prove a field is missing has to see it missing.
    """
    target = path or MANIFEST_PATH
    with open(target, encoding="utf-8") as handle:
        return json.load(handle)


def cases(manifest: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    """The case list, in committed order."""
    return list((manifest or load_manifest())["cases"])


def cases_by_category(
    category: str, manifest: dict[str, Any] | None = None
) -> list[dict[str, Any]]:
    return [case for case in cases(manifest) if case["category"] == category]


__all__ = [
    "ASSETS_DIRNAME",
    "ACCEPTED_LICENCE_PREFIXES",
    "ASSET_FIELDS",
    "COMMON_FIELDS",
    "LICENCES_REQUIRING_A_DEED_URL",
    "PUBLIC_DOMAIN_LICENCES",
    "MANIFEST_PATH",
    "MIN_AMBIGUOUS_CASES",
    "MIN_DELEGATION_CASES",
    "MIN_DISTINCT_SPECIES",
    "MIN_FAILURE_CASES",
    "MIN_REAL_CASES",
    "MIN_TOTAL_CASES",
    "RECOGNITION_IMAGE_CONTEXT_KEY",
    "REQUIRED_BY_CATEGORY",
    "VALID_CAPABILITIES",
    "VALID_CATEGORIES",
    "VALID_DECISIONS",
    "VALID_DIFFICULTIES",
    "VALID_ERROR_CODES",
    "VALID_MEDIA_TYPES",
    "VALID_RESUME_KEYS",
    "VALID_STATUSES",
    "cases",
    "cases_by_category",
    "load_manifest",
]
