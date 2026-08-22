"""Phase 3 - real BioCLIP-2 inference on the official public Hugging Face Space.

The local design was cancelled: it required a 2.66 GB TreeOfLife text-embedding
artifact cached on disk, and that exception was refused. Inference now runs on
`imageomics/bioclip-2-demo`, the Space the model authors publish.

Every test here injects a fake Gradio client. Nothing in this module needs a
Hugging Face account, a network connection, a model weight or an embedding file.

The fake responses reproduce shapes actually observed from the Space during the
Phase 3 API checkpoint - including the awkward one:

    "Animalia Chordata Squamata  Liolaemidae Ctenoblepharys adspersa"

where the doubled space is a MISSING Order. That is why label parsing is
positional on a single space and never `str.split()`.
"""
from __future__ import annotations

import math
import time
from pathlib import Path

import pytest

from ..adapters.bioclip import (
    MockBioCLIP2Provider,
    RemoteBioCLIP2Provider,
    build_classifier,
    parse_space_label,
)
from ..agent import RecognitionAgent
from ..config import (
    REMOTE_MODEL_VERSION,
    REMOTE_SPACE_API_NAME,
    REMOTE_SPACE_ID,
    REMOTE_SPACE_MAX_PREDICTIONS,
    REMOTE_SPACE_RANK,
    ConfigError,
    RecognitionConfig,
)
from ..domain.errors import ErrorCode, RecognitionError
from ..schema import AgentRequest, AgentStatus
from .conftest import image_entry, make_config, png_bytes

AGENT_DIR = Path(__file__).resolve().parent.parent

# A real reply, copied from the checkpoint against a public cheetah photograph.
REAL_CONFIDENCES = [
    {"label": "Animalia Chordata Mammalia Carnivora Felidae Acinonyx jubatus (Cheetah)",
     "confidence": 0.656154},
    {"label": "Animalia Chordata Mammalia Carnivora Felidae Acinonyx pardinensis",
     "confidence": 0.30793},
    {"label": "Animalia Chordata Mammalia Carnivora Viverridae Genetta thierryi (Haussa genet)",
     "confidence": 0.011994},
    {"label": "Animalia Chordata Mammalia Carnivora Felidae Panthera pardus (Leopard)",
     "confidence": 0.005778},
    {"label": "Animalia Chordata Squamata  Liolaemidae Ctenoblepharys adspersa",
     "confidence": 0.001194},
]


def label_payload(confidences):
    return {"label": confidences[0]["label"] if confidences else None,
            "confidences": list(confidences)}


class FakeJob:
    """Stands in for `gradio_client.client.Job`.

    Models the parts the provider actually relies on: a bounded `result(timeout)`
    that can raise, a `done()` flag and a `cancel()` that records having been
    called - so a test can prove an overrunning job is not abandoned.
    """

    def __init__(self, result=None, raises=None, hang: bool = False):
        self._result = result
        self._raises = raises
        self._hang = hang
        self._done = False
        self.cancelled = False

    def result(self, timeout=None):
        if self._hang:
            # Never completes. The provider must give up on its own deadline
            # rather than wait here forever - which is exactly the defect that
            # let a full-agent smoke run for the better part of an hour.
            import concurrent.futures

            if timeout:
                time.sleep(min(timeout, 30))
            raise concurrent.futures.TimeoutError("job never completed")
        if self._raises is not None:
            self._done = True
            raise self._raises
        self._done = True
        return self._result

    def done(self):
        return self._done

    def cancel(self):
        self.cancelled = True
        return True


class FakeSpaceClient:
    """Stands in for `gradio_client.Client`. Records calls, sends nothing."""

    def __init__(self, result=None, raises=None, hang: bool = False):
        if result is None:
            result = (label_payload(REAL_CONFIDENCES), None, "<a href='#'>x</a>")
        self._result = result
        self._raises = raises
        self._hang = hang
        self.calls: list[dict] = []
        self.jobs: list[FakeJob] = []

    def submit(self, *args, **kwargs):
        self.calls.append({"args": args, "kwargs": kwargs})
        job = FakeJob(result=self._result, raises=self._raises, hang=self._hang)
        self.jobs.append(job)
        return job

    def predict(self, *args, **kwargs):  # pragma: no cover - must never be used
        raise AssertionError(
            "predict() takes no timeout and blocks forever; the provider must "
            "use submit() + Job.result(timeout=...)"
        )


def remote(client=None, **overrides):
    return RemoteBioCLIP2Provider(client=client or FakeSpaceClient(), **overrides)


def normalized(png: bytes | None = None):
    """A validated image, produced by the real validation path."""
    from ..validation import validate_paired_request

    return validate_paired_request(
        "What animal is this?",
        {"recognition_image": image_entry(png if png is not None else png_bytes())},
        make_config().validation,
    )


# ===========================================================================
# 1 - protocol conformance
# ===========================================================================

def test_the_remote_provider_satisfies_the_classifier_protocol():
    from ..adapters.bioclip import BioCLIP2Classifier

    provider = remote()
    assert isinstance(provider, BioCLIP2Classifier)
    assert provider.provider_name == "RemoteBioCLIP2Provider"
    assert provider.recognition_mode == "remote_bioclip2_open_domain_species"
    assert provider.version == REMOTE_MODEL_VERSION
    assert callable(provider.classify)


def test_it_returns_the_existing_typed_prediction_objects():
    from ..domain.models import BioCLIPTaxonPrediction

    out = remote().classify(normalized(), 5)
    assert out and all(isinstance(p, BioCLIPTaxonPrediction) for p in out)
    assert all(p.rank == "species" for p in out)


# ===========================================================================
# 2 - selected only in remote mode / no silent mock fallback
# ===========================================================================

def test_the_remote_provider_is_built_only_in_remote_mode():
    assert isinstance(build_classifier(make_config(bioclip_provider_mode="remote")),
                      RemoteBioCLIP2Provider)
    assert isinstance(build_classifier(make_config(bioclip_provider_mode="mock")),
                      MockBioCLIP2Provider)


def test_remote_mode_never_silently_falls_back_to_the_mock():
    built = build_classifier(make_config(bioclip_provider_mode="remote"))
    assert not isinstance(built, MockBioCLIP2Provider)


@pytest.mark.parametrize("mode", ["banana", "real", "", "REMOTE-ish"])
def test_an_unknown_mode_never_yields_any_provider(mode):
    with pytest.raises(ConfigError):
        build_classifier(make_config(bioclip_provider_mode=mode))


def test_a_remote_failure_does_not_degrade_to_fixture_predictions():
    """The controlled failure must be a failure, not a quiet mock answer."""
    provider = remote(client=FakeSpaceClient(raises=RuntimeError("space asleep")))
    with pytest.raises(RecognitionError) as caught:
        provider.classify(normalized(), 5)
    assert caught.value.code is ErrorCode.CLASSIFICATION_UNAVAILABLE


# ===========================================================================
# 3 - no network at import or construction
# ===========================================================================

def test_importing_the_package_imports_no_http_client():
    """Run in a FRESH interpreter on purpose.

    Checking `sys.modules` in-process would be meaningless: any earlier test that
    classifies imports the client and leaves it loaded for the rest of the
    session, so the assertion would turn on test ordering rather than on what an
    import actually does.
    """
    import subprocess
    import sys

    probe = (
        "import importlib, sys; "
        "importlib.import_module("
        "'backend.agents.multimodal_recognition_agent.agent'); "
        "bad = [m for m in "
        "('gradio_client', 'torch', 'torchvision', 'open_clip', 'pybioclip', 'bioclip') "
        "if m in sys.modules]; "
        "print(','.join(bad))"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=str(AGENT_DIR.parents[2]),
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "", result.stdout


def test_constructing_the_provider_opens_no_connection():
    provider = RemoteBioCLIP2Provider()
    assert provider._client is None  # lazy: nothing built, nothing sent


def test_the_client_import_is_not_at_module_scope():
    source = (AGENT_DIR / "adapters" / "bioclip.py").read_text(encoding="utf-8")
    for line in source.splitlines():
        assert not line.startswith("import gradio_client")
        assert not line.startswith("from gradio_client")


def test_building_the_agent_in_remote_mode_opens_no_connection():
    agent = RecognitionAgent(make_config(bioclip_provider_mode="remote"))
    assert agent._workflow._classifier._client is None


# ===========================================================================
# 4 - image and Species rank are mapped correctly
# ===========================================================================

def test_the_species_rank_is_always_requested():
    client = FakeSpaceClient()
    remote(client=client).classify(normalized(), 5)

    assert len(client.calls) == 1
    args = client.calls[0]["args"]
    assert args[1] == "Species" == REMOTE_SPACE_RANK
    assert client.calls[0]["kwargs"]["api_name"] == REMOTE_SPACE_API_NAME


def test_the_validated_image_bytes_are_what_is_sent(tmp_path):
    """The provider sends the already-validated bytes, unmodified."""
    png = png_bytes(marker="phase3-remote-marker")
    image = normalized(png)
    sent: dict = {}

    class CapturingClient(FakeSpaceClient):
        def submit(self, *args, **kwargs):
            handle = args[0]
            path = handle.get("path") if isinstance(handle, dict) else getattr(handle, "path", None)
            if path:
                sent["bytes"] = Path(path).read_bytes()
            return super().submit(*args, **kwargs)

    remote(client=CapturingClient()).classify(image, 5)
    assert sent["bytes"] == png == image.image_bytes


def test_the_temporary_upload_file_does_not_survive_the_call():
    seen: dict = {}

    class PathRecordingClient(FakeSpaceClient):
        def submit(self, *args, **kwargs):
            handle = args[0]
            seen["path"] = handle.get("path") if isinstance(handle, dict) else None
            return super().submit(*args, **kwargs)

    remote(client=PathRecordingClient()).classify(normalized(), 5)
    assert seen["path"] and not Path(seen["path"]).exists()


# ===========================================================================
# 5-7 - mapping, Top-K, ordering, deduplication
# ===========================================================================

def test_a_real_five_prediction_response_maps_completely():
    out = remote().classify(normalized(), 5)
    assert len(out) == 5
    assert out[0].scientific_name == "Acinonyx jubatus"
    assert out[0].species_id == "acinonyx_jubatus"
    assert out[0].common_name == "Cheetah"
    assert out[0].classification_score == pytest.approx(0.656154)
    # the missing-Order label still yields the right binomial
    assert out[4].scientific_name == "Ctenoblepharys adspersa"


@pytest.mark.parametrize("top_k", [1, 2, 3, 4, 5])
def test_top_k_slices_without_fabricating(top_k):
    out = remote().classify(normalized(), top_k)
    assert len(out) == top_k
    assert [p.scientific_name for p in out] == [
        "Acinonyx jubatus", "Acinonyx pardinensis", "Genetta thierryi",
        "Panthera pardus", "Ctenoblepharys adspersa",
    ][:top_k]


@pytest.mark.parametrize("top_k", [6, 10, 50])
def test_a_top_k_above_the_space_maximum_is_never_padded(top_k):
    """The Space caps itself at five. Asking for more returns fewer candidates,
    never invented ones."""
    out = remote().classify(normalized(), top_k)
    assert len(out) == REMOTE_SPACE_MAX_PREDICTIONS == 5


def test_a_non_positive_top_k_is_a_contract_violation():
    for bad in (0, -1):
        with pytest.raises(RecognitionError) as caught:
            remote().classify(normalized(), bad)
        assert caught.value.code is ErrorCode.CLASSIFICATION_CONTRACT_VIOLATION


def test_an_out_of_order_response_is_refused_not_silently_sorted():
    unordered = [
        {"label": "Animalia Chordata Mammalia Carnivora Felidae Panthera leo", "confidence": 0.10},
        {"label": "Animalia Chordata Mammalia Carnivora Felidae Panthera onca", "confidence": 0.90},
    ]
    provider = remote(client=FakeSpaceClient(result=(label_payload(unordered), None, "")))
    with pytest.raises(RecognitionError) as caught:
        provider.classify(normalized(), 5)
    assert caught.value.code is ErrorCode.CLASSIFICATION_CONTRACT_VIOLATION


def test_a_duplicated_taxon_keeps_only_the_highest_occurrence():
    duplicated = [
        {"label": "Animalia Chordata Mammalia Carnivora Felidae Panthera leo (Lion)",
         "confidence": 0.80},
        {"label": "Animalia Chordata Mammalia Carnivora Felidae Panthera leo",
         "confidence": 0.10},
    ]
    out = remote(
        client=FakeSpaceClient(result=(label_payload(duplicated), None, ""))
    ).classify(normalized(), 5)
    assert [p.species_id for p in out] == ["panthera_leo"]
    assert out[0].classification_score == pytest.approx(0.80)


# ===========================================================================
# 8-9 - malformed scores and labels
# ===========================================================================

@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf"),
                                   "0.5", None, True])
def test_a_non_finite_or_non_numeric_score_is_dropped(score):
    rows = [
        {"label": "Animalia Chordata Mammalia Carnivora Felidae Panthera leo",
         "confidence": 0.9},
        {"label": "Animalia Chordata Mammalia Carnivora Felidae Panthera onca",
         "confidence": score},
    ]
    out = remote(
        client=FakeSpaceClient(result=(label_payload(rows), None, ""))
    ).classify(normalized(), 5)
    assert [p.species_id for p in out] == ["panthera_leo"]
    assert all(math.isfinite(p.classification_score) for p in out)


def test_a_score_outside_the_allowed_range_is_a_contract_violation():
    rows = [{"label": "Animalia Chordata Mammalia Carnivora Felidae Panthera leo",
             "confidence": 1.5}]
    provider = remote(client=FakeSpaceClient(result=(label_payload(rows), None, "")))
    with pytest.raises(RecognitionError) as caught:
        provider.classify(normalized(), 5)
    assert caught.value.code is ErrorCode.CLASSIFICATION_CONTRACT_VIOLATION


@pytest.mark.parametrize("bad", [
    "", "   ", "OnlyOneToken", "Too few parts here", None, 12345, [],
    "Animalia Chordata Mammalia Carnivora Felidae Panthera",          # 6 ranks
    "Animalia Chordata Mammalia Carnivora Felidae Panthera leo extra",  # 8 ranks
    "Animalia Chordata Mammalia Carnivora Felidae  leo",              # empty genus
])
def test_a_malformed_label_is_never_guessed_into_a_species(bad):
    assert parse_space_label(bad) is None


def test_a_response_whose_labels_are_all_malformed_yields_no_candidates():
    rows = [{"label": "nonsense", "confidence": 0.9}]
    out = remote(
        client=FakeSpaceClient(result=(label_payload(rows), None, ""))
    ).classify(normalized(), 5)
    assert out == []


# ===========================================================================
# 10-14 - malformed payloads and remote failure modes
# ===========================================================================

@pytest.mark.parametrize("payload", [
    None, [], "text", 42, {"confidences": None}, {"confidences": "x"},
    {"no_confidences": []}, {"confidences": [1, 2]},
])
def test_a_malformed_payload_is_a_contract_violation(payload):
    provider = remote(client=FakeSpaceClient(result=(payload, None, "")))
    with pytest.raises(RecognitionError) as caught:
        provider.classify(normalized(), 5)
    assert caught.value.code is ErrorCode.CLASSIFICATION_CONTRACT_VIOLATION


@pytest.mark.parametrize("failure", [
    TimeoutError("timed out"),
    ConnectionError("connection reset"),
    RuntimeError("Queue is full"),
    ValueError("Space is sleeping"),
    KeyError("/lambda"),                       # endpoint / API-contract change
    OSError("client failure"),
])
def test_every_remote_failure_maps_to_classification_unavailable(failure):
    provider = remote(client=FakeSpaceClient(raises=failure))
    with pytest.raises(RecognitionError) as caught:
        provider.classify(normalized(), 5)
    assert caught.value.code is ErrorCode.CLASSIFICATION_UNAVAILABLE


def test_a_remote_failure_never_leaks_the_provider_error_body(caplog):
    secret = "queue-token-abc123-should-never-appear"
    provider = remote(client=FakeSpaceClient(raises=RuntimeError(secret)))
    with caplog.at_level("DEBUG"):
        with pytest.raises(RecognitionError):
            provider.classify(normalized(), 5)
    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert secret not in logged
    assert "RuntimeError" in logged


def test_the_agent_returns_a_controlled_failure_never_an_exception():
    """End to end: a dead Space is a structured `failed`, not a 500."""
    agent = RecognitionAgent(
        make_config(bioclip_provider_mode="remote"),
        classifier=remote(client=FakeSpaceClient(raises=TimeoutError("dead"))),
    )
    result = agent.run(AgentRequest(
        instruction="What animal is this?",
        context={"recognition_image": image_entry(png_bytes())},
    ))
    assert result.status is AgentStatus.FAILED
    assert result.output["error_code"] == "CLASSIFICATION_UNAVAILABLE"


# ===========================================================================
# 15 - the sample image and HTML link are not scientific evidence
# ===========================================================================

def test_the_sample_image_is_deleted_and_the_html_link_is_ignored(tmp_path):
    sample = tmp_path / "sample_of_taxon.jpg"
    sample.write_bytes(b"not-evidence")
    html = '<a href="https://www.gbif.org/species/5219404">GBIF</a>'

    provider = remote(client=FakeSpaceClient(
        result=(label_payload(REAL_CONFIDENCES), str(sample), html)))
    out = provider.classify(normalized(), 5)

    assert not sample.exists(), "the downloaded sample must not survive the call"
    # Nothing from the HTML reaches a candidate: no gbif id is invented here.
    assert all(getattr(p, "gbif_id", None) is None for p in out)


def test_the_provider_does_not_parse_the_gbif_html_link():
    """Phase 4 calls the official GBIF API. The displayed link is not a source."""
    source = (AGENT_DIR / "adapters" / "bioclip.py").read_text(encoding="utf-8")
    executable = "\n".join(
        line for line in source.splitlines() if not line.strip().startswith("#")
    )
    assert "gbif.org" not in executable.lower()


# ===========================================================================
# 17 - arbitrary images are sent, not looked up by digest
# ===========================================================================

def test_an_arbitrary_unregistered_image_is_sent_to_the_provider():
    """The remote provider has no fixture and no digest table: any valid image
    reaches the Space, which is exactly what the mock could not do."""
    client = FakeSpaceClient()
    provider = remote(client=client)

    first = normalized(png_bytes(marker="unregistered-one"))
    second = normalized(png_bytes(marker="unregistered-two"))
    assert first.image_sha256 != second.image_sha256

    assert provider.classify(first, 5)
    assert provider.classify(second, 5)
    assert len(client.calls) == 2


def test_the_remote_provider_has_no_digest_lookup_at_all():
    """Checked on real name/attribute references, never on prose."""
    import ast

    source = (AGENT_DIR / "adapters" / "bioclip.py").read_text(encoding="utf-8")
    node = next(
        n for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.ClassDef) and n.name == "RemoteBioCLIP2Provider"
    )
    names = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
    names |= {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}
    for forbidden in ("image_sha256", "knows", "_load", "fixture_version"):
        assert forbidden not in names, forbidden


# ===========================================================================
# 18-19 - mock mode unchanged, provenance follows the provider that ran
# ===========================================================================

def _run(agent):
    return agent.run(AgentRequest(
        instruction="What animal is this?",
        context={"recognition_image": image_entry(png_bytes())},
    ))


def test_mock_mode_provenance_is_completely_unchanged():
    result = _run(RecognitionAgent(make_config()))
    provenance = result.output["recognition_provenance"]

    assert provenance["recognition_provider"] == "MockBioCLIP2Provider"
    assert provenance["recognition_mode"] == "mock_classification"
    assert provenance["mock_provider_version"] == "sprint2-mock-bioclip2-classifier-v1"
    assert provenance["score_kind"] == "deterministic_sprint2_test_score"
    assert provenance["score_is_probability"] is False
    # No remote-only key leaks into a mock answer.
    for remote_only in ("remote_space_id", "remote_space_revision", "model_version"):
        assert remote_only not in provenance


def test_remote_provenance_reports_the_provider_that_actually_ran():
    agent = RecognitionAgent(
        make_config(bioclip_provider_mode="remote"), classifier=remote()
    )
    provenance = _run(agent).output["recognition_provenance"]

    assert provenance["recognition_provider"] == "RemoteBioCLIP2Provider"
    assert provenance["recognition_mode"] == "remote_bioclip2_open_domain_species"
    assert provenance["model_target"] == "BioCLIP-2"
    assert provenance["model_version"] == REMOTE_MODEL_VERSION == "imageomics/bioclip-2"
    assert provenance["remote_space_id"] == REMOTE_SPACE_ID == "imageomics/bioclip-2-demo"
    assert provenance["remote_space_revision"]
    # A real model version must never be filed under a key named "mock".
    assert provenance["mock_provider_version"] is None
    assert provenance["score_kind"] == "bioclip2_remote_zero_shot_ranking_score"
    # A softmax over ~867k labels is a ranking value, not calibrated confidence.
    assert provenance["score_is_probability"] is False


def test_provenance_follows_a_swapped_provider_not_the_configured_mode():
    """Config says remote; the injected provider is the mock. Provenance must
    describe what RAN, so the two can never drift apart."""
    agent = RecognitionAgent(
        make_config(bioclip_provider_mode="remote"),
        classifier=MockBioCLIP2Provider(),
    )
    provenance = _run(agent).output["recognition_provenance"]
    assert provenance["recognition_provider"] == "MockBioCLIP2Provider"
    assert provenance["recognition_mode"] == "mock_classification"


def test_the_seven_output_keys_are_unchanged_in_remote_mode():
    agent = RecognitionAgent(
        make_config(bioclip_provider_mode="remote"), classifier=remote()
    )
    assert sorted(_run(agent).output.keys()) == [
        "gbif_id", "ncbi_taxid", "recognition", "recognition_candidates",
        "recognition_provenance", "species", "species_id",
    ]


# ===========================================================================
# 20 - no local ML stack, weights, cache, embeddings or similarity symbols
# ===========================================================================

def test_no_local_ml_dependency_is_declared():
    lines = [
        line.strip().lower()
        for line in (AGENT_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    declared = "\n".join(lines)
    for forbidden in ("torch", "torchvision", "open_clip", "open-clip", "pybioclip",
                      "qdrant", "faiss", "chromadb", "pinecone", "weaviate", "milvus",
                      "numpy", "pandas"):
        assert forbidden not in declared, forbidden


def test_the_only_added_client_dependency_is_the_gradio_client():
    declared = [
        line.strip()
        for line in (AGENT_DIR / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    assert "gradio_client==2.6.0" in declared


def test_no_local_ml_module_is_importable_from_the_agent():
    import sys

    for forbidden in ("torch", "torchvision", "open_clip", "pybioclip", "qdrant_client"):
        assert forbidden not in sys.modules, forbidden


def test_no_embedding_weight_or_cache_symbol_is_introduced():
    """Executable code only - docstrings may say what is deliberately absent,
    the same rule the scope gates already apply to every shipped module."""
    from .test_recognition_only_gate import executable_text

    executable = executable_text(AGENT_DIR / "adapters" / "bioclip.py").lower()
    for forbidden in ("txt_emb", "snapshot_download", "hf_hub_download", "safetensors",
                      "state_dict", "from_pretrained", "nearest", "cosine",
                      "faiss", "qdrant", "vector_store"):
        assert forbidden not in executable, forbidden


def test_no_weight_or_embedding_artifact_is_written_by_a_classification():
    """A full remote classification leaves nothing behind on disk."""
    before = {p.name for p in AGENT_DIR.rglob("*") if p.is_file() and ".venv" not in p.parts}
    remote().classify(normalized(), 5)
    after = {p.name for p in AGENT_DIR.rglob("*") if p.is_file() and ".venv" not in p.parts}
    assert after == before


def test_the_configured_timeout_reaches_the_provider():
    config = make_config(bioclip_provider_mode="remote")
    built = build_classifier(config)
    assert built.timeout_seconds == config.remote_classifier_timeout_seconds


def test_the_remote_timeout_is_bounded_and_has_no_retry_loop():
    from ..config import DEFAULT_REMOTE_CLASSIFIER_TIMEOUT_SECONDS

    assert 0 < DEFAULT_REMOTE_CLASSIFIER_TIMEOUT_SECONDS < 600

    from .test_recognition_only_gate import executable_text

    executable = executable_text(AGENT_DIR / "adapters" / "bioclip.py").lower()
    for forbidden in ("for attempt", "while true", "retry", "tenacity", "backoff"):
        assert forbidden not in executable, forbidden


def test_one_classification_issues_exactly_one_remote_call():
    client = FakeSpaceClient()
    remote(client=client).classify(normalized(), 5)
    assert len(client.calls) == 1


# ===========================================================================
# Bounded deadline - the defect that let a full-agent smoke run for ~1 hour
# ===========================================================================

def test_a_job_that_never_completes_times_out_within_the_deadline():
    """The whole point of the correction. `Client.predict()` takes no timeout
    and blocks until the server answers; a stalled queue therefore held the
    request open indefinitely. The provider must now give up on its own."""
    client = FakeSpaceClient(hang=True)
    provider = remote(client=client, timeout_seconds=1.0)

    started = time.monotonic()
    with pytest.raises(RecognitionError) as caught:
        provider.classify(normalized(), 5)
    elapsed = time.monotonic() - started

    assert caught.value.code is ErrorCode.CLASSIFICATION_UNAVAILABLE
    # Deadline plus a deterministic test margin - never unbounded.
    assert elapsed < 1.0 + 5.0, elapsed


def test_an_overrunning_job_is_cancelled_not_abandoned():
    client = FakeSpaceClient(hang=True)
    provider = remote(client=client, timeout_seconds=1.0)

    with pytest.raises(RecognitionError):
        provider.classify(normalized(), 5)

    assert client.jobs and client.jobs[0].cancelled, (
        "an overrunning job must be cancelled, not left running"
    )


def test_the_deadline_is_passed_to_the_bounded_wait():
    """`Job.result` is called WITH a timeout, and it is the configured one."""
    seen = {}

    class DeadlineRecordingJob(FakeJob):
        def result(self, timeout=None):
            seen["timeout"] = timeout
            return super().result(timeout=timeout)

    class DeadlineRecordingClient(FakeSpaceClient):
        def submit(self, *args, **kwargs):
            self.calls.append({"args": args, "kwargs": kwargs})
            job = DeadlineRecordingJob(result=self._result)
            self.jobs.append(job)
            return job

    remote(client=DeadlineRecordingClient(), timeout_seconds=12.5).classify(
        normalized(), 5)

    assert seen["timeout"] is not None, "the wait must be bounded"
    assert 0 < seen["timeout"] <= 12.5


def test_predict_is_never_used_because_it_cannot_be_bounded():
    """A regression guard, checked on real attribute references.

    `Client.predict()` accepts no timeout and blocks until the server answers.
    The provider must reach the Space only through `submit()`.
    """
    import ast

    source = (AGENT_DIR / "adapters" / "bioclip.py").read_text(encoding="utf-8")
    node = next(
        n for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.ClassDef) and n.name == "RemoteBioCLIP2Provider"
    )
    attributes = {n.attr for n in ast.walk(node) if isinstance(n, ast.Attribute)}
    assert "submit" in attributes
    assert "result" in attributes
    assert "cancel" in attributes
    assert "predict" not in attributes


def test_a_completed_job_is_not_cancelled():
    client = FakeSpaceClient()
    remote(client=client).classify(normalized(), 5)
    assert client.jobs and not client.jobs[0].cancelled


# ===========================================================================
# Certification coverage: fewer-than-five, and an absent common name
# ===========================================================================

@pytest.mark.parametrize("count", [1, 2, 3, 4])
def test_a_response_with_fewer_than_five_candidates_is_returned_intact(count):
    """The Space returns at most five, sometimes fewer. Asking for five must
    yield exactly what arrived - never padded up to the requested Top-K."""
    rows = REAL_CONFIDENCES[:count]
    out = remote(
        client=FakeSpaceClient(result=(label_payload(rows), None, ""))
    ).classify(normalized(), 5)

    assert len(out) == count
    assert [p.scientific_name for p in out] == [
        parse_space_label(r["label"])[1] for r in rows
    ]


def test_a_label_without_a_common_name_maps_to_none_not_an_empty_string():
    """`Acinonyx pardinensis` has no vernacular name in the real reply."""
    rows = [REAL_CONFIDENCES[1]]
    out = remote(
        client=FakeSpaceClient(result=(label_payload(rows), None, ""))
    ).classify(normalized(), 5)

    assert out[0].scientific_name == "Acinonyx pardinensis"
    assert out[0].common_name is None


def test_a_missing_intermediate_rank_does_not_shift_the_binomial():
    """The double-space format marks an absent rank. Splitting on whitespace
    instead of a single space would silently shift every later rank and produce
    the wrong genus/species pair."""
    label = "Animalia Chordata Squamata  Liolaemidae Ctenoblepharys adspersa"
    assert label.split(" ") != label.split()          # the trap
    assert len(label.split(" ")) == 7                 # positional ranks preserved
    assert parse_space_label(label) == (
        "ctenoblepharys_adspersa", "Ctenoblepharys adspersa", None
    )
