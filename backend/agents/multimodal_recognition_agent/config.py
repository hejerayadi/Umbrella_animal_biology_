"""Configuration for the Multimodal Recognition Agent.

Everything is read from the process environment, once, at startup. This module
deliberately does NOT load a `.env` file: whoever starts the service (uvicorn,
the launcher, a container) owns how the environment gets populated, and keeping
that responsibility out of here means no credential can ever be read, cached or
printed by this codebase.

Two rules shape the defaults below:

1. Limits that protect the service (image size, pixel area) have real defaults,
   taken from the Sprint 2 specification.
2. Values that belong to the shared Qdrant contract have NO defaults. They are
   Chahd's to define. An unset value stays unset and the real retriever refuses
   to start, rather than guessing a collection name or a vector dimension and
   querying something incompatible.

The one exception is `mock_embedding_dimension`, which needs *a* number for the
local mock path to run at all. It is labelled `local_default_pending_manifest`
in the provenance of every response so no reader mistakes it for an agreed value.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any

# The one approved image key. This is a constant, not a setting: accepting an
# alternate key from configuration would quietly reopen a frozen contract.
RECOGNITION_IMAGE_CONTEXT_KEY = "recognition_image"

ALLOWED_MEDIA_TYPES = ("image/jpeg", "image/png", "image/webp")

# Local development labels. They are deliberately NOT the strings from the
# specification's example manifest - using those would imply the shared contract
# is frozen when it is not.
LOCAL_MOCK_PROVIDER_VERSION = "local-dev-mock-bioclip2-v0"
LOCAL_DATASET_VERSION = "local-dev-fixtures-v0"


class ConfigError(RuntimeError):
    """Raised at startup when the environment holds an unusable value."""


def _int(name: str, default: int | None) -> int | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


def _float(name: str, default: float | None) -> float | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc


def _str(name: str, default: str | None = None) -> str | None:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip()


def _flag(name: str, default: bool = False) -> bool:
    """A boolean switch. Anything other than an explicit truthy word is False,
    so a typo turns a feature OFF rather than silently ON."""
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _positive(name: str, value: int | None) -> int:
    if value is None or value <= 0:
        raise ConfigError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class ValidationConfig:
    """Bounds applied to an incoming image before it is ever decoded fully."""

    max_image_bytes: int
    max_image_pixels: int
    min_image_width: int
    min_image_height: int

    @property
    def max_encoded_len(self) -> int:
        """Hard bound on the *encoded* payload length.

        Base64 expands 3 bytes into 4 characters, so a payload longer than this
        cannot possibly decode to something within `max_image_bytes`. Checking it
        first means a 200 MB string is rejected without ever being decoded.

        The bound deliberately does not subtract padding: padding characters are
        attacker-controlled and unvalidated at this point, so trusting them to
        shrink the estimate would defeat the guard.
        """
        return 4 * -(-self.max_image_bytes // 3)  # 4 * ceil(max_bytes / 3)


@dataclass(frozen=True)
class ThresholdConfig:
    """Scenario boundaries for the confidence gate.

    These are workflow test boundaries, NOT scientific calibration. They decide
    which branch of the Sprint 2 workflow a request takes; they say nothing
    about biological accuracy.
    """

    identified_min_score: float
    identified_min_margin: float
    uncertain_min_score: float


@dataclass(frozen=True)
class QdrantConfig:
    """The shared collection contract. Every field is Chahd's to supply.

    `is_frozen` is the single question the real retriever asks before it will
    construct itself.
    """

    url: str | None
    collection: str | None
    vector_name: str | None
    expected_dimension: int | None
    expected_distance: str | None
    dataset_version: str | None
    top_k_references: int | None
    timeout_seconds: float

    # Fields that must be present before a real query may be attempted. The API
    # key is not listed: some deployments are open on a private network, so its
    # absence is not proof of an unfrozen contract.
    _REQUIRED = ("url", "collection", "expected_dimension", "expected_distance",
                 "dataset_version", "top_k_references")

    @property
    def missing_contract_fields(self) -> tuple[str, ...]:
        return tuple(name for name in self._REQUIRED if getattr(self, name) is None)

    @property
    def is_frozen(self) -> bool:
        return not self.missing_contract_fields


@dataclass(frozen=True)
class RecognitionConfig:
    validation: ValidationConfig
    thresholds: ThresholdConfig
    qdrant: QdrantConfig

    bioclip_provider_mode: str
    mock_provider_version: str
    mock_embedding_dimension: int
    mock_embedding_dimension_is_local_default: bool

    retrieval_mode: str
    taxonomy_provider_mode: str
    text_analyzer_mode: str

    top_k_species: int
    max_references_per_species: int

    # Local development aid: maps an image's SHA-256 to a fixture vector seed so
    # a known demo image reliably retrieves a known reference. Never required,
    # never populated in production, and it holds no image data - only hashes.
    image_seed_overrides: dict[str, str] = field(default_factory=dict)

    # --- optional reasoning LLM -------------------------------------------
    # Disabled unless explicitly switched on. Note what is NOT here: no base
    # URL, no key, no deployment name. Credentials are read by the adapter from
    # the environment at call time and never enter a configuration object, a
    # log line, a fixture or this source file.
    # Two calls at most, and the workflow spends them in fixed roles: one to
    # plan, one to explain. Never more, whatever the model asks for.
    reasoning_llm_enabled: bool = False
    reasoning_llm_max_calls_per_request: int = 2
    reasoning_llm_timeout_seconds: float = 10.0
    # "disabled" | "fake" | "azure". The code default is `disabled` so a fake
    # brain can never switch itself on in a running service; `.env.example`
    # documents `fake` as the local-development value.
    reasoning_llm_provider_mode: str = "disabled"

    @classmethod
    def from_env(cls) -> RecognitionConfig:
        validation = ValidationConfig(
            max_image_bytes=_positive("MAX_IMAGE_BYTES", _int("MAX_IMAGE_BYTES", 10_485_760)),
            max_image_pixels=_positive("MAX_IMAGE_PIXELS", _int("MAX_IMAGE_PIXELS", 25_000_000)),
            min_image_width=_positive("MIN_IMAGE_WIDTH", _int("MIN_IMAGE_WIDTH", 64)),
            min_image_height=_positive("MIN_IMAGE_HEIGHT", _int("MIN_IMAGE_HEIGHT", 64)),
        )
        # A minimum larger than the maximum can never be satisfied - fail now,
        # loudly, rather than rejecting every request at runtime.
        if validation.min_image_width * validation.min_image_height > validation.max_image_pixels:
            raise ConfigError(
                "MIN_IMAGE_WIDTH * MIN_IMAGE_HEIGHT exceeds MAX_IMAGE_PIXELS; "
                "no image could ever satisfy both limits"
            )

        thresholds = ThresholdConfig(
            identified_min_score=_float("IDENTIFIED_MIN_SCORE", 0.75),
            identified_min_margin=_float("IDENTIFIED_MIN_MARGIN", 0.08),
            uncertain_min_score=_float("UNCERTAIN_MIN_SCORE", 0.45),
        )
        if not 0.0 <= thresholds.uncertain_min_score <= thresholds.identified_min_score <= 1.0:
            raise ConfigError(
                "thresholds must satisfy 0 <= UNCERTAIN_MIN_SCORE <= IDENTIFIED_MIN_SCORE <= 1"
            )
        if thresholds.identified_min_margin < 0.0:
            raise ConfigError("IDENTIFIED_MIN_MARGIN must not be negative")

        qdrant = QdrantConfig(
            url=_str("QDRANT_URL"),
            collection=_str("QDRANT_COLLECTION"),
            vector_name=_str("QDRANT_VECTOR_NAME"),
            expected_dimension=_int("QDRANT_EXPECTED_DIMENSION", None),
            expected_distance=_str("QDRANT_EXPECTED_DISTANCE"),
            dataset_version=_str("QDRANT_DATASET_VERSION"),
            top_k_references=_int("QDRANT_TOP_K_REFERENCES", None),
            timeout_seconds=_float("QDRANT_TIMEOUT_SECONDS", 10.0),
        )

        bioclip_mode = _str("BIOCLIP_PROVIDER_MODE", "mock")
        if bioclip_mode != "mock":
            raise ConfigError(
                "BIOCLIP_PROVIDER_MODE must be 'mock' in Sprint 2; real BioCLIP-2 "
                "inference is explicitly out of scope"
            )

        taxonomy_mode = _str("TAXONOMY_PROVIDER_MODE", "mock")
        if taxonomy_mode != "mock":
            raise ConfigError(
                "TAXONOMY_PROVIDER_MODE must be 'mock' in Sprint 2; live GBIF/NCBI "
                "calls are explicitly out of scope"
            )

        retrieval_mode = _str("RECOGNITION_RETRIEVAL_MODE", "mock")
        if retrieval_mode not in ("mock", "real"):
            raise ConfigError("RECOGNITION_RETRIEVAL_MODE must be 'mock' or 'real'")

        # The mock vector length must match the collection when one is configured.
        # Otherwise every query would be rejected by Qdrant anyway - better to
        # refuse at startup than to fail one request at a time.
        configured_dimension = _int("MOCK_EMBEDDING_DIMENSION", None)
        is_local_default = configured_dimension is None
        dimension = _positive(
            "MOCK_EMBEDDING_DIMENSION",
            configured_dimension if configured_dimension is not None else 32,
        )
        if qdrant.expected_dimension is not None and dimension != qdrant.expected_dimension:
            raise ConfigError(
                "MOCK_EMBEDDING_DIMENSION does not match QDRANT_EXPECTED_DIMENSION; "
                "the query vector would be rejected by the collection"
            )

        raw_overrides = _str("RECOGNITION_IMAGE_SEED_OVERRIDES")
        overrides: dict[str, str] = {}
        if raw_overrides:
            try:
                parsed: Any = json.loads(raw_overrides)
            except json.JSONDecodeError as exc:
                raise ConfigError("RECOGNITION_IMAGE_SEED_OVERRIDES must be a JSON object") from exc
            if not isinstance(parsed, dict) or not all(
                isinstance(k, str) and isinstance(v, str) for k, v in parsed.items()
            ):
                raise ConfigError(
                    "RECOGNITION_IMAGE_SEED_OVERRIDES must map SHA-256 strings to seed strings"
                )
            overrides = parsed

        return cls(
            validation=validation,
            thresholds=thresholds,
            qdrant=qdrant,
            bioclip_provider_mode=bioclip_mode,
            mock_provider_version=_str("BIOCLIP_MOCK_PROVIDER_VERSION", LOCAL_MOCK_PROVIDER_VERSION),
            mock_embedding_dimension=dimension,
            mock_embedding_dimension_is_local_default=is_local_default,
            retrieval_mode=retrieval_mode,
            taxonomy_provider_mode=taxonomy_mode,
            text_analyzer_mode=_str("TEXT_ANALYZER_MODE", "rules"),
            top_k_species=_positive("RECOGNITION_TOP_K_SPECIES", _int("RECOGNITION_TOP_K_SPECIES", 5)),
            max_references_per_species=_positive(
                "RECOGNITION_MAX_REFERENCES_PER_SPECIES",
                _int("RECOGNITION_MAX_REFERENCES_PER_SPECIES", 3),
            ),
            image_seed_overrides=overrides,
            reasoning_llm_enabled=_flag("RECOGNITION_REASONING_LLM_ENABLED"),
            reasoning_llm_max_calls_per_request=_positive(
                "RECOGNITION_LLM_MAX_CALLS_PER_REQUEST",
                min(_int("RECOGNITION_LLM_MAX_CALLS_PER_REQUEST", 2) or 2, 2),
            ),
            reasoning_llm_timeout_seconds=_float("RECOGNITION_LLM_TIMEOUT_SECONDS", 10.0),
            reasoning_llm_provider_mode=_provider_mode(),
        )


def _provider_mode() -> str:
    """Resolve the reasoning provider mode.

    `RECOGNITION_LLM_PROVIDER_MODE` is the switch. The older
    `RECOGNITION_REASONING_LLM_ENABLED=true` still selects Azure, so an existing
    .env keeps working without being rewritten.
    """
    mode = (_str("RECOGNITION_LLM_PROVIDER_MODE") or "").strip().lower()
    if mode:
        if mode not in ("disabled", "fake", "azure"):
            raise ConfigError(
                "RECOGNITION_LLM_PROVIDER_MODE must be 'disabled', 'fake' or 'azure'"
            )
        return mode
    return "azure" if _flag("RECOGNITION_REASONING_LLM_ENABLED") else "disabled"
