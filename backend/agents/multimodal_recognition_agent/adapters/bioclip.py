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
import logging
import math
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ..config import (
    BIOCLIP_IMPLEMENTED_MODES,
    BIOCLIP_PROVIDER_MODES,
    MOCK_CLASSIFIER_VERSION,
    RECOGNITION_MODE_MOCK_CLASSIFICATION,
    RECOGNITION_MODE_REMOTE_CLASSIFICATION,
    REMOTE_MODEL_VERSION,
    REMOTE_SPACE_API_NAME,
    REMOTE_SPACE_ID,
    REMOTE_SPACE_MAX_PREDICTIONS,
    REMOTE_SPACE_RANK,
    REMOTE_SPACE_REVISION,
    DEFAULT_REMOTE_CLASSIFIER_TIMEOUT_SECONDS,
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
_logger = logging.getLogger(__name__)

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

    if mode == "remote":
        # Constructed only; no connection is opened until the first classify.
        return RemoteBioCLIP2Provider(
            timeout_seconds=config.remote_classifier_timeout_seconds,
        )

    if mode in BIOCLIP_PROVIDER_MODES:
        raise ConfigError(
            f"BIOCLIP_PROVIDER_MODE={mode!r} has no implementation yet. "
            "Refusing to start rather than serving mock predictions as real ones."
        )

    raise ConfigError(
        "BIOCLIP_PROVIDER_MODE must be one of: "
        + ", ".join(BIOCLIP_PROVIDER_MODES)
        + f". Got {mode!r}."
    )


# Named so a test can assert the two lists have not drifted apart.
IMPLEMENTED_CLASSIFIER_MODES = BIOCLIP_IMPLEMENTED_MODES


# ===========================================================================
# Remote BioCLIP-2, executed on the official public Hugging Face Space
#
# The local design was cancelled. Running BioCLIP-2 in-process would have meant
# caching a 2.66 GB TreeOfLife text-embedding artifact on disk, and that
# exception was refused, so inference happens on the Space the model authors
# publish.
#
# What crosses the boundary is one image and the word "Species". What comes back
# is a ranked list of taxonomic labels. No weight, embedding, index or reference
# corpus is downloaded, and there is still nothing here through which a vector or
# a similarity search could enter the agent.
# ===========================================================================

# The Space returns "Kingdom Phylum Class Order Family Genus species (Common)".
# Ranks are positional and CAN BE EMPTY - a real reply contained
# "Animalia Chordata Squamata  Liolaemidae Ctenoblepharys adspersa", where the
# doubled space is a missing Order. So the taxon string is split on a single
# space and never with str.split(), which would collapse the gap and silently
# shift every rank after it.
_TAXON_RANK_COUNT = 7
_GENUS_INDEX = 5
_SPECIES_INDEX = 6

_SUFFIX_BY_MEDIA_TYPE = {
    "image/png": ".png",
    "image/jpeg": ".jpg",
    "image/webp": ".webp",
}


def parse_space_label(raw: Any) -> tuple[str, str, str | None] | None:
    """Split one Space label into (species_id, scientific_name, common_name).

    Returns None for anything that does not parse. A label that cannot be read
    is dropped rather than guessed at: inventing a binomial from a malformed
    string is how an agent ends up publishing a species nobody predicted.
    """
    if not isinstance(raw, str):
        return None

    text = raw.strip()
    if not text:
        return None

    common: str | None = None
    if text.endswith(")") and " (" in text:
        text, _, trailing = text.rpartition(" (")
        common = trailing[:-1].strip() or None
        text = text.strip()

    parts = text.split(" ")
    if len(parts) != _TAXON_RANK_COUNT:
        return None

    genus = parts[_GENUS_INDEX].strip()
    species = parts[_SPECIES_INDEX].strip()
    if not genus or not species:
        return None

    scientific_name = genus + " " + species
    # Same shape the fixture oracle uses, so nothing downstream can tell the two
    # providers apart by identifier style.
    species_id = (genus + "_" + species).lower()
    return species_id, scientific_name, common


class RemoteBioCLIP2Provider:
    """Real BioCLIP-2 label classification, executed on the official Space.

    Holds the same `classify(image, top_k)` contract as the mock, so the
    workflow, the ranking rules and the confidence gate are untouched.

    The client is built on first use, never at import: constructing this object
    opens no connection, which is what keeps the offline suite offline.
    """

    provider_name = "RemoteBioCLIP2Provider"
    recognition_mode = RECOGNITION_MODE_REMOTE_CLASSIFICATION
    model_target = "BioCLIP-2"
    space_max_predictions = REMOTE_SPACE_MAX_PREDICTIONS

    def __init__(
        self,
        *,
        space_id: str = REMOTE_SPACE_ID,
        api_name: str = REMOTE_SPACE_API_NAME,
        rank: str = REMOTE_SPACE_RANK,
        timeout_seconds: float = DEFAULT_REMOTE_CLASSIFIER_TIMEOUT_SECONDS,
        version: str = REMOTE_MODEL_VERSION,
        client: Any = None,
    ) -> None:
        self.space_id = space_id
        self.api_name = api_name
        self.rank = rank
        self.timeout_seconds = timeout_seconds
        self.version = version
        # Injectable so every offline test drives the real mapping code with no
        # network, no account and no download.
        self._client = client

    # -- lazy client --------------------------------------------------------

    def _ensure_client(self) -> Any:
        if self._client is None:
            # Imported here, not at module scope: importing this package must not
            # pull in an HTTP client or reach the network.
            from gradio_client import Client

            self._client = Client(
                self.space_id,
                verbose=False,
                # Constructing the client fetches the Space's API description
                # over HTTP, so it needs a bound of its own - otherwise the very
                # first step could hang before a deadline is ever consulted.
                httpx_kwargs={"timeout": self.timeout_seconds},
                # The endpoint also returns a sample image of the predicted
                # taxon. It is an illustration, not evidence, so it is never
                # fetched: no download, no temp file, no cleanup to get wrong.
                download_files=False,
            )
        return self._client

    # -- the production interface -------------------------------------------

    def classify(
        self,
        image: NormalizedRecognitionInput,
        top_k: int,
    ) -> list[BioCLIPTaxonPrediction]:
        """Top-K taxonomic labels for one image, from the remote Space."""

        if top_k <= 0:
            raise RecognitionError(ErrorCode.CLASSIFICATION_CONTRACT_VIOLATION)

        payload = self._call(image)
        predictions = self._to_predictions(payload)

        # The provider promises "already ranked, distinct". Hold it to that here
        # rather than downstream, so a misbehaving Space is a controlled failure
        # and not a quietly reordered answer.
        assert_ranked_and_distinct(predictions)
        # Never padded: the Space caps its own output at five, and a configured
        # Top-K above that returns fewer candidates rather than invented ones.
        return predictions[:top_k]

    # -- transport ----------------------------------------------------------

    def _call(self, image: NormalizedRecognitionInput) -> Any:
        """One request to the Space, under ONE overall deadline. No retry loop.

        `Client.predict()` is deliberately not used: it takes no timeout and
        blocks until the server answers, so a stalled queue holds the request
        open indefinitely. The official asynchronous mechanism is used instead -
        `submit()` returns a Job, the wait is bounded, and a Job that overruns is
        cancelled rather than abandoned to a worker thread.

        The deadline spans everything: client construction, upload, queue wait,
        inference and result retrieval. Every failure mode - a sleeping Space, a
        full queue, a timeout, a changed endpoint, a broken client - lands on the
        same controlled `CLASSIFICATION_UNAVAILABLE`, because from the workflow's
        point of view they are all "no classification happened".
        """
        deadline = time.monotonic() + self.timeout_seconds
        temp_path = None
        result = None
        job = None
        try:
            from gradio_client import handle_file

            # The endpoint takes a file. The bytes are the already-validated
            # ones; they go to a private temp file for the length of the call and
            # are removed in `finally`, never kept.
            suffix = _SUFFIX_BY_MEDIA_TYPE.get(image.media_type, ".png")
            descriptor, temp_path = tempfile.mkstemp(suffix=suffix)
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(image.image_bytes)

            client = self._ensure_client()
            job = client.submit(
                handle_file(temp_path),
                self.rank,
                api_name=self.api_name,
            )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("remote classification deadline exhausted")
            # Bounded, single wait. Not a polling loop.
            result = job.result(timeout=remaining)
        except RecognitionError:
            raise
        except Exception as exc:  # noqa: BLE001 - every remote failure is one outcome
            # Type only. A remote error body can echo the request or the Space's
            # internals, and neither belongs in this agent's logs.
            _logger.info(
                "[Recognition] remote classification failed (%s); controlled "
                "CLASSIFICATION_UNAVAILABLE.", type(exc).__name__,
            )
            raise RecognitionError(ErrorCode.CLASSIFICATION_UNAVAILABLE) from exc
        finally:
            # Never leave a job running behind a request that has already given
            # up on it.
            if job is not None and not job.done():
                try:
                    job.cancel()
                except Exception:  # noqa: BLE001 - cleanup must not mask the outcome
                    pass
            _remove_quietly(temp_path)

        # The endpoint also returns a sample image of the predicted taxon and an
        # HTML link. Neither is scientific evidence: the sample image is an
        # illustration, and the link is NOT the taxonomy source - Phase 4 calls
        # the official GBIF API instead. The downloaded sample is deleted here so
        # nothing survives the call.
        if isinstance(result, (list, tuple)):
            if len(result) > 1 and isinstance(result[1], str):
                _remove_quietly(result[1])
            return result[0] if result else None
        return result

    # -- mapping ------------------------------------------------------------

    def _to_predictions(self, payload: Any) -> list[BioCLIPTaxonPrediction]:
        """Turn the Space's reply into the existing typed predictions.

        A reply that does not match the documented shape is a contract
        violation, not an empty result: "the endpoint changed" and "this image
        has no match" must never look the same to the confidence gate.
        """
        if not isinstance(payload, dict):
            raise RecognitionError(ErrorCode.CLASSIFICATION_CONTRACT_VIOLATION)

        confidences = payload.get("confidences")
        if not isinstance(confidences, list):
            raise RecognitionError(ErrorCode.CLASSIFICATION_CONTRACT_VIOLATION)

        predictions: list[BioCLIPTaxonPrediction] = []
        seen: set[str] = set()
        for entry in confidences:
            if not isinstance(entry, dict):
                raise RecognitionError(ErrorCode.CLASSIFICATION_CONTRACT_VIOLATION)

            parsed = parse_space_label(entry.get("label"))
            score = entry.get("confidence")
            if parsed is None or isinstance(score, bool) or not isinstance(score, (int, float)):
                # An unreadable row is skipped, not repaired.
                continue
            score = float(score)
            if not math.isfinite(score):
                continue

            species_id, scientific_name, common_name = parsed
            if species_id in seen:
                # The Space ranks distinct taxa; a repeat would break the
                # distinctness rule, so the first (highest) occurrence wins.
                continue
            seen.add(species_id)

            predictions.append(
                BioCLIPTaxonPrediction(
                    species_id=species_id,
                    scientific_name=scientific_name,
                    common_name=common_name,
                    rank="species",
                    # Ranking score, NOT a calibrated probability - it is a
                    # softmax over ~867k labels and is never presented as one.
                    classification_score=score,
                )
            )

        return predictions

    def provenance(self) -> dict:
        """What this provider will report about itself. Read by the finalizer."""
        return {
            "remote_space_id": self.space_id,
            "remote_space_revision": REMOTE_SPACE_REVISION,
            "model_version": self.version,
        }


def _remove_quietly(path: object) -> None:
    """Delete a temporary file if it exists. A failure here is never fatal."""
    if isinstance(path, str) and path:
        try:
            os.unlink(path)
        except OSError:
            pass
