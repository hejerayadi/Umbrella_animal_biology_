"""The species-classification boundary.

BioCLIP-2 is the officially selected model for the real recognition phase. It is
NOT executed in Sprint 2: no package is imported, no weights are downloaded, no
GPU is required, and nothing in this module can reach a network.

What the boundary produces is a ranked list of **taxonomic labels** for one
image. It does not produce an embedding, it does not query a vector store, and
it has no notion of a nearest neighbour or a reference image. That is the whole
architectural point:

    MockBioCLIP2Provider.classify(...)
              -> later replaced by ->
    RealBioCLIP2Provider.classify(...)

The real provider will run BioCLIP-2 / Tree-of-Life label classification and
return Top-K taxonomic labels through this same `classify(image, top_k)`
signature. Replacing the mock requires no vector database, no collection and no
change anywhere else in this agent.

------------------------------------------------------------------------------
WHAT THE SPRINT 2 SCORES ARE
------------------------------------------------------------------------------

`classification_score` in this module is a **deterministic Sprint 2 test score**
read from a controlled fixture. It is not a calibrated probability, not a
softmax output, and not scientific confidence. Its only job is to be reproducible
so the confidence gate, the ranking rules and the workflow branches can be tested
end to end.

The fixture is a test oracle. It is not a biological dataset and must never be
presented as real BioCLIP-2 inference.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..config import (
    BIOCLIP_IMPLEMENTED_MODES,
    BIOCLIP_PROVIDER_MODES,
    MOCK_CLASSIFIER_VERSION,
    RECOGNITION_MODE_MOCK_CLASSIFICATION,
    ConfigError,
    RecognitionConfig,
)
from ..domain.errors import ErrorCode, RecognitionError
from ..domain.models import BioCLIPTaxonPrediction, NormalizedRecognitionInput
from ..domain.ranking import assert_ranked_and_distinct, validate_prediction_payload

_FIXTURE_PATH = (
    Path(__file__).resolve().parent.parent / "fixtures" / "mock_bioclip_predictions.json"
)

# A SHA-256 hex digest, lowercase. Fixture keys are checked against this so a
# truncated or upper-cased key is caught as a malformed oracle rather than
# silently never matching any image.
_SHA256_LENGTH = 64
_HEX_DIGITS = frozenset("0123456789abcdef")


@runtime_checkable
class BioCLIP2Classifier(Protocol):
    """What the workflow needs from a species classifier.

    Deliberately tiny. One image in, an already-ranked list of taxonomic labels
    out. There is nothing here through which a vector, a collection or a
    reference image could enter the agent.
    """

    provider_name: str
    version: str
    recognition_mode: str

    def classify(
        self,
        image: NormalizedRecognitionInput,
        top_k: int,
    ) -> list[BioCLIPTaxonPrediction]:
        ...


def _is_sha256(value: Any) -> bool:
    return (
        isinstance(value, str)
        and len(value) == _SHA256_LENGTH
        and all(char in _HEX_DIGITS for char in value)
    )


class MockBioCLIP2Provider:
    """Deterministic stand-in for BioCLIP-2 label classification.

    Executes no model, imports no BioCLIP package, loads no weights and opens no
    connection. It answers from a controlled fixture keyed by the exact SHA-256
    of the decoded image bytes - the same digest `validation.decode_image`
    computes.

    Two paths, both reproducible:

    - a known demo/test image (its SHA-256 is a fixture key) gets that image's
      controlled, already-ranked list of taxa;
    - anything else gets an EMPTY list, which the confidence gate turns into
      `not_identified`. That is the honest answer for an image the oracle knows
      nothing about, and it is why this class can never attach a random
      biological label to an unknown photograph.
    """

    provider_name = "MockBioCLIP2Provider"
    recognition_mode = RECOGNITION_MODE_MOCK_CLASSIFICATION
    model_target = "BioCLIP-2"

    def __init__(
        self,
        *,
        version: str = MOCK_CLASSIFIER_VERSION,
        predictions: dict[str, list[dict]] | None = None,
        fixture_path: str | Path | None = None,
        fail_with: ErrorCode | None = None,
    ) -> None:
        """`predictions` overrides the fixture file - that is how tests build
        exact scenarios (a clear winner, two close labels, no labels at all).

        `fail_with` makes the provider-unavailable branch testable without
        needing something to actually be unavailable.
        """
        self.version = version
        self._fail_with = fail_with
        self._fixture_path = Path(fixture_path) if fixture_path else _FIXTURE_PATH
        self._inline = predictions
        # Loaded lazily so a malformed oracle surfaces as a controlled failure on
        # a request, not as a crash while the service is starting.
        self._images: dict[str, list[dict]] | None = None
        self._fixture_version: str | None = None

    # -- fixture loading ----------------------------------------------------

    def _load(self) -> dict[str, list[dict]]:
        if self._images is not None:
            return self._images

        if self._inline is not None:
            raw_images: Any = self._inline
            self._fixture_version = self.version
        else:
            try:
                document = json.loads(self._fixture_path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as exc:
                raise RecognitionError(ErrorCode.CLASSIFICATION_FIXTURE_INVALID) from exc
            if not isinstance(document, dict):
                raise RecognitionError(ErrorCode.CLASSIFICATION_FIXTURE_INVALID)

            self._fixture_version = document.get("provider_version")
            if not isinstance(self._fixture_version, str) or not self._fixture_version:
                raise RecognitionError(ErrorCode.CLASSIFICATION_FIXTURE_INVALID)
            raw_images = document.get("images")

        if not isinstance(raw_images, dict):
            raise RecognitionError(ErrorCode.CLASSIFICATION_FIXTURE_INVALID)

        validated: dict[str, list[dict]] = {}
        for key, entries in raw_images.items():
            if not _is_sha256(key):
                raise RecognitionError(ErrorCode.CLASSIFICATION_FIXTURE_INVALID)
            if not isinstance(entries, list):
                raise RecognitionError(ErrorCode.CLASSIFICATION_FIXTURE_INVALID)
            for entry in entries:
                if not validate_prediction_payload(entry):
                    raise RecognitionError(ErrorCode.CLASSIFICATION_FIXTURE_INVALID)
            validated[key] = list(entries)

        self._images = validated
        return validated

    @property
    def fixture_version(self) -> str | None:
        """The `provider_version` the loaded oracle declares. None until loaded."""
        return self._fixture_version

    def knows(self, image_sha256: str) -> bool:
        """Whether the oracle holds an entry for this exact digest."""
        return image_sha256 in self._load()

    # -- the interface the real provider will implement ---------------------

    def classify(
        self,
        image: NormalizedRecognitionInput,
        top_k: int,
    ) -> list[BioCLIPTaxonPrediction]:
        """Top-K taxonomic labels for one image. Deterministic, offline."""

        if self._fail_with is not None:
            raise RecognitionError(self._fail_with)
        if top_k <= 0:
            raise RecognitionError(ErrorCode.CLASSIFICATION_CONTRACT_VIOLATION)

        entries = self._load().get(image.image_sha256)
        if not entries:
            # Unknown image. No label is guessed, invented or sampled - the
            # workflow will report `not_identified`.
            return []

        try:
            predictions = [
                BioCLIPTaxonPrediction(
                    species_id=entry["species_id"],
                    scientific_name=entry["scientific_name"],
                    common_name=entry.get("common_name"),
                    rank=entry.get("rank", "species"),
                    classification_score=float(entry["classification_score"]),
                )
                for entry in entries
            ]
        except Exception as exc:  # noqa: BLE001 - a broken oracle is a controlled failure
            raise RecognitionError(ErrorCode.CLASSIFICATION_FIXTURE_INVALID) from exc

        # The fixture claims to be already ranked. Hold it to that: an unordered
        # or duplicated oracle fails the request rather than being quietly
        # sorted into something that looks correct.
        assert_ranked_and_distinct(
            predictions, error=ErrorCode.CLASSIFICATION_FIXTURE_INVALID
        )

        return predictions[:top_k]


def build_classifier(config: RecognitionConfig) -> BioCLIP2Classifier:
    """Construct the classifier the configured mode names, or refuse.

    The mock is reachable only through `mode == "mock"`. That is the whole
    reason this function exists: the agent used to build `MockBioCLIP2Provider`
    unconditionally whenever no classifier was injected, so a deployment that
    asked for real inference and never noticed the request was ignored would
    have served fixture predictions under a production banner.

    `real` raises here as well as in `RecognitionConfig.from_env`, because a
    config object can also be built directly in code, and the guarantee this
    function makes is "no mock outside mock mode" - not "no mock, provided
    someone went through the environment".
    """
    mode = (config.bioclip_provider_mode or "").strip().lower()

    if mode == "mock":
        return MockBioCLIP2Provider(
            version=config.mock_provider_version,
            fixture_path=config.classification_fixture_path,
        )

    if mode in BIOCLIP_PROVIDER_MODES:
        raise ConfigError(
            f"BIOCLIP_PROVIDER_MODE={mode!r} has no implementation yet "
            "(real BioCLIP-2 inference arrives in Phase 3). Refusing to start "
            "rather than serving mock predictions as real ones."
        )

    raise ConfigError(
        "BIOCLIP_PROVIDER_MODE must be one of: "
        + ", ".join(BIOCLIP_PROVIDER_MODES)
        + f". Got {mode!r}."
    )


# Named so a test can assert the two lists have not drifted apart.
IMPLEMENTED_CLASSIFIER_MODES = BIOCLIP_IMPLEMENTED_MODES
