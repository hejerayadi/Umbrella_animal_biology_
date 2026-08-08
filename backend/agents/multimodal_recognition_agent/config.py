"""Configuration for the Multimodal Recognition Agent.

Everything is read from the process environment, once, at startup. This module
deliberately does NOT load a `.env` file: whoever starts the service (uvicorn,
the launcher, a container) owns how the environment gets populated, and keeping
that responsibility out of here means no credential can ever be read, cached or
printed by this codebase.

Note what is absent, and is absent on purpose: there is no vector dimension, no
collection name, no distance metric, no dataset version and no `QDRANT_*`
anything. This agent classifies an image into taxonomic labels. It runs no
vector search, so it has nothing to configure one with, and an operator who sets
a `QDRANT_*` variable in this agent's environment will find that nothing reads
it.

Limits that protect the service (image size, pixel area) have real defaults,
taken from the Sprint 2 specification. Thresholds have defaults too, and they
are workflow test boundaries rather than scientific calibration.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

# The one approved image key. This is a constant, not a setting: accepting an
# alternate key from configuration would quietly reopen a frozen contract.
RECOGNITION_IMAGE_CONTEXT_KEY = "recognition_image"

ALLOWED_MEDIA_TYPES = ("image/jpeg", "image/png", "image/webp")

# What the classification boundary is doing in Sprint 2, stated in one word that
# appears verbatim in every response's provenance. It is not "classification" -
# it is a MOCK of classification, and the two must never read the same.
RECOGNITION_MODE_MOCK_CLASSIFICATION = "mock_classification"

# The model the real provider will run, later, through the same interface.
MODEL_TARGET = "BioCLIP-2"

# Version label of the shipped Sprint 2 mock. It matches the `provider_version`
# in `fixtures/mock_bioclip_predictions.json`; the provider refuses a fixture
# that disagrees, so the label in a response is always the label of the file
# that produced it.
MOCK_CLASSIFIER_VERSION = "sprint2-mock-bioclip2-classifier-v1"


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
    about biological accuracy, and in Sprint 2 the scores they compare are
    deterministic mock values.
    """

    identified_min_score: float
    identified_min_margin: float
    uncertain_min_score: float


@dataclass(frozen=True)
class RecognitionConfig:
    validation: ValidationConfig
    thresholds: ThresholdConfig

    bioclip_provider_mode: str
    recognition_mode: str
    mock_provider_version: str
    # Optional path to an alternative classification fixture, so a demo can add
    # its own image hashes without editing the committed test oracle. Holds a
    # path only - never image data.
    classification_fixture_path: str | None

    taxonomy_provider_mode: str
    text_analyzer_mode: str

    # The maximum number of distinct taxa ever returned, and the value handed to
    # the classifier as `top_k`.
    top_k_species: int

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

        bioclip_mode = _str("BIOCLIP_PROVIDER_MODE", "mock")
        if bioclip_mode != "mock":
            raise ConfigError(
                "BIOCLIP_PROVIDER_MODE must be 'mock' in Sprint 2; real BioCLIP-2 "
                "label classification is explicitly out of scope"
            )

        taxonomy_mode = _str("TAXONOMY_PROVIDER_MODE", "mock")
        if taxonomy_mode != "mock":
            raise ConfigError(
                "TAXONOMY_PROVIDER_MODE must be 'mock' in Sprint 2; live GBIF/NCBI "
                "calls are explicitly out of scope"
            )

        return cls(
            validation=validation,
            thresholds=thresholds,
            bioclip_provider_mode=bioclip_mode,
            recognition_mode=RECOGNITION_MODE_MOCK_CLASSIFICATION,
            mock_provider_version=_str(
                "BIOCLIP_MOCK_PROVIDER_VERSION", MOCK_CLASSIFIER_VERSION
            ),
            classification_fixture_path=_str("RECOGNITION_CLASSIFICATION_FIXTURE_PATH"),
            taxonomy_provider_mode=taxonomy_mode,
            text_analyzer_mode=_str("TEXT_ANALYZER_MODE", "rules"),
            top_k_species=_positive("RECOGNITION_TOP_K_SPECIES", _int("RECOGNITION_TOP_K_SPECIES", 5)),
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
