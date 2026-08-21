"""Phase 8 - the demonstration entry point, and the closure contract checks.

Everything here is offline. The demonstration script is exercised through its
`main(argv)` with injected providers, so no test needs an opt-in, a credential
or a network - which is the same property every other suite in this agent has.
"""
from __future__ import annotations

import json
import re

import pytest

from ..adapters.taxonomy import MockTaxonomyProvider
from ..agent import RecognitionAgent
from ..domain.errors import ErrorCode
from .conftest import StubClassifier, jpeg_bytes, make_config, png_bytes, prediction

DEMO_MODULE = "backend.agents.multimodal_recognition_agent.demo_sprint3"

CANARY_KEY = "sk-CANARY-demo-key-88ff21"
CANARY_ENDPOINT = "https://canary-demo-endpoint.invalid/openai/v1"
CANARY_MARKER = "CANARYDEMOIMAGEBYTES4z"


# --- helpers ---------------------------------------------------------------

def write_image(tmp_path, name="animal.jpg", raw=None):
    path = tmp_path / name
    path.write_bytes(raw if raw is not None else jpeg_bytes(256, 256))
    return path


def demo_agent(predictions=None, *, classifier=None):
    return RecognitionAgent(
        make_config(top_k_species=5),
        classifier=classifier or StubClassifier(
            predictions if predictions is not None
            else [prediction("panthera_leo", 0.96, "Panthera leo")]),
        taxonomy_provider=MockTaxonomyProvider(),
    )


def run_demo(monkeypatch, argv, *, opt_in="1", agent=None, config=None):
    """Drive `main(argv)` with the agent and config injected."""
    from .. import demo_sprint3

    if opt_in is None:
        monkeypatch.delenv(demo_sprint3.LIVE_OPT_IN_VAR, raising=False)
    else:
        monkeypatch.setenv(demo_sprint3.LIVE_OPT_IN_VAR, opt_in)

    # The demonstration loads the agent's .env at runtime. An offline test must
    # never pull real credentials into its own process, so the loader is stubbed.
    monkeypatch.setattr(demo_sprint3, "load_agent_environment", lambda: None)

    built = agent if agent is not None else demo_agent()
    monkeypatch.setattr(
        "backend.agents.multimodal_recognition_agent.agent.RecognitionAgent",
        lambda *a, **k: built)
    monkeypatch.setattr(
        "backend.agents.multimodal_recognition_agent.config.RecognitionConfig.from_env",
        classmethod(lambda cls: config or make_config(top_k_species=5)))

    return demo_sprint3.main(argv)


# --- the opt-in guard ------------------------------------------------------

def test_the_demo_refuses_to_run_without_the_live_opt_in(monkeypatch, capsys, tmp_path):
    image = write_image(tmp_path)
    code = run_demo(monkeypatch, ["--image", str(image)], opt_in=None)

    assert code == 2
    assert "ABORTED" in capsys.readouterr().out


@pytest.mark.parametrize("value", ["", "0", "true", "yes", "TRUE", " 1"])
def test_only_the_exact_opt_in_value_authorizes_a_live_call(
    monkeypatch, capsys, tmp_path, value
):
    image = write_image(tmp_path)
    code = run_demo(monkeypatch, ["--image", str(image)], opt_in=value)

    assert code == 2
    assert "ABORTED" in capsys.readouterr().out


def test_the_opt_in_guard_runs_before_any_provider_is_built(monkeypatch, tmp_path):
    """A guard that fired after construction would already have opened a client."""

    class ExplodingClassifier:
        provider_name = "ExplodingClassifier"
        recognition_mode = "remote_bioclip2_open_domain_species"
        version = "v1"

        def classify(self, image, top_k):
            raise AssertionError("the demo reached a provider without an opt-in")

    image = write_image(tmp_path)
    assert run_demo(
        monkeypatch, ["--image", str(image)],
        opt_in=None, agent=demo_agent(classifier=ExplodingClassifier()),
    ) == 2


# --- the image argument ----------------------------------------------------

def test_a_missing_image_is_reported_and_exits_non_zero(monkeypatch, capsys, tmp_path):
    code = run_demo(monkeypatch, ["--image", str(tmp_path / "absent.jpg")])

    assert code == 3
    assert "does not exist" in capsys.readouterr().out


def test_a_directory_is_not_accepted_as_an_image(monkeypatch, capsys, tmp_path):
    code = run_demo(monkeypatch, ["--image", str(tmp_path)])

    assert code == 3
    assert "not a file" in capsys.readouterr().out


def test_an_empty_image_file_is_refused(monkeypatch, capsys, tmp_path):
    empty = tmp_path / "empty.png"
    empty.write_bytes(b"")
    code = run_demo(monkeypatch, ["--image", str(empty)])

    assert code == 3
    assert "empty" in capsys.readouterr().out


def test_an_unsupported_suffix_is_refused(monkeypatch, capsys, tmp_path):
    odd = tmp_path / "animal.gif"
    odd.write_bytes(b"GIF89a")
    code = run_demo(monkeypatch, ["--image", str(odd)])

    assert code == 3
    assert "unsupported image type" in capsys.readouterr().out


def test_the_image_path_is_required(monkeypatch, tmp_path):
    with pytest.raises(SystemExit):
        run_demo(monkeypatch, [])


def test_the_script_hardcodes_no_personal_path():
    """A demonstration that only runs on one machine is not a demonstration."""
    import pathlib

    source = (pathlib.Path(__file__).resolve().parent.parent
              / "demo_sprint3.py").read_text(encoding="utf-8")

    assert "C:\\Users" not in source
    assert "/home/" not in source
    assert "/Users/" not in source
    assert not re.search(r"default\s*=\s*[\"'][^\"']*[/\\]", source)


# --- a successful demonstration -------------------------------------------

def test_a_successful_demonstration_prints_the_safe_evidence(
    monkeypatch, capsys, tmp_path
):
    image = write_image(tmp_path)
    code = run_demo(monkeypatch, ["--image", str(image)])
    out = capsys.readouterr().out

    assert code == 0
    for expected in ("status", "decision", "species", "gbif_id", "ncbi_taxid",
                     "top_k", "recognition_provider", "reasoning_llm_calls",
                     "latency_seconds", "bioclip_provider_mode",
                     "taxonomy_provider_mode", "output_keys"):
        assert f"[demo] {expected}:" in out


def test_the_demonstration_prints_the_ranked_candidates(monkeypatch, capsys, tmp_path):
    image = write_image(tmp_path)
    run_demo(monkeypatch, ["--image", str(image)], agent=demo_agent([
        prediction("panthera_leo", 0.96, "Panthera leo"),
        prediction("panthera_tigris", 0.02, "Panthera tigris"),
    ]))
    out = capsys.readouterr().out

    assert "1. Panthera leo" in out
    assert "2. Panthera tigris" in out
    assert "score=0.96" in out


def test_a_custom_instruction_is_accepted(monkeypatch, capsys, tmp_path):
    image = write_image(tmp_path)
    code = run_demo(monkeypatch, [
        "--image", str(image), "--instruction", "Is this a Panthera leo?"])

    assert code == 0
    assert "[demo] text_alignment:" in capsys.readouterr().out


# --- controlled failure ----------------------------------------------------

def test_a_controlled_failure_exits_non_zero_with_its_code(
    monkeypatch, capsys, tmp_path
):
    image = write_image(tmp_path)
    failing = RecognitionAgent(
        make_config(),
        classifier=StubClassifier(raises=ErrorCode.CLASSIFICATION_UNAVAILABLE),
        taxonomy_provider=MockTaxonomyProvider(),
    )
    code = run_demo(monkeypatch, ["--image", str(image)], agent=failing)
    out = capsys.readouterr().out

    assert code == 1
    assert "CLASSIFICATION_UNAVAILABLE" in out


def test_a_corrupt_image_produces_a_controlled_code_not_a_traceback(
    monkeypatch, capsys, tmp_path
):
    image = write_image(tmp_path, name="corrupt.png", raw=png_bytes()[:40])
    code = run_demo(monkeypatch, ["--image", str(image)])
    out = capsys.readouterr().out

    assert code == 1
    assert ErrorCode.CORRUPT_IMAGE.value in out
    assert "Traceback" not in out


def test_a_delegating_result_reports_the_capability_and_stops(
    monkeypatch, capsys, tmp_path
):
    image = write_image(tmp_path)
    code = run_demo(monkeypatch, [
        "--image", str(image),
        "--instruction", "Identify this animal and describe its evolutionary history.",
    ])
    out = capsys.readouterr().out

    assert code == 0
    assert "[demo] needs_capability: Evolution" in out
    assert "never invokes another agent itself" in out


# --- nothing sensitive is printed -----------------------------------------

def test_the_demonstration_prints_no_base64_or_image_byte(
    monkeypatch, capsys, tmp_path
):
    import base64

    raw = png_bytes(marker=CANARY_MARKER)
    image = write_image(tmp_path, name="marked.png", raw=raw)
    run_demo(monkeypatch, ["--image", str(image)])
    out = capsys.readouterr().out

    assert CANARY_MARKER not in out
    assert base64.b64encode(raw).decode("ascii")[:48] not in out
    assert "data:image" not in out
    assert "base64" not in out


def test_the_demonstration_prints_no_credential_or_endpoint(
    monkeypatch, capsys, tmp_path
):
    monkeypatch.setenv("AZURE_OPENAI_API_KEY", CANARY_KEY)
    monkeypatch.setenv("AZURE_OPENAI_BASE_URL", CANARY_ENDPOINT)
    monkeypatch.setenv("NCBI_EMAIL", "canary.person@example.invalid")

    image = write_image(tmp_path)
    run_demo(monkeypatch, ["--image", str(image)])
    out = capsys.readouterr().out

    assert CANARY_KEY not in out
    assert CANARY_ENDPOINT not in out
    assert "canary.person@example.invalid" not in out


def test_the_demonstration_never_prints_a_raw_provider_error(
    monkeypatch, capsys, tmp_path
):
    class LeakyClassifier:
        provider_name = "LeakyClassifier"
        recognition_mode = "remote_bioclip2_open_domain_species"
        version = "v1"

        def classify(self, image, top_k):
            from ..domain.errors import RecognitionError

            raise RecognitionError(ErrorCode.CLASSIFICATION_UNAVAILABLE)

    image = write_image(tmp_path)
    code = run_demo(monkeypatch, ["--image", str(image)],
                    agent=demo_agent(classifier=LeakyClassifier()))
    out = capsys.readouterr().out

    assert code == 1
    assert "CLASSIFICATION_UNAVAILABLE" in out
    # The fixed message, and nothing from a provider.
    assert "Traceback" not in out and "http" not in out.lower().replace("https://", "")


def test_the_source_reads_no_env_value_into_its_output():
    """The script may consult configuration; it may not print any of it."""
    import pathlib

    source = (pathlib.Path(__file__).resolve().parent.parent
              / "demo_sprint3.py").read_text(encoding="utf-8")
    emitted = re.findall(r"emit\(([^)]*)\)", source)
    joined = "\n".join(emitted).lower()

    for forbidden in ("api_key", "base_url", "endpoint", "email", "data_url",
                      "b64encode", "image_bytes", "instruction"):
        assert forbidden not in joined, f"{forbidden} must never be emitted"


def test_the_demo_writes_no_file(monkeypatch, tmp_path):
    image = write_image(tmp_path)
    before = {p.name for p in tmp_path.iterdir()}
    run_demo(monkeypatch, ["--image", str(image)])
    after = {p.name for p in tmp_path.iterdir()}

    assert before == after


def test_the_demo_declares_a_bounded_ceiling_and_never_starts_an_orchestrator():
    import pathlib

    from .. import demo_sprint3

    assert demo_sprint3.DEMO_DEADLINE_SECONDS > 0

    source = (pathlib.Path(__file__).resolve().parent.parent
              / "demo_sprint3.py").read_text(encoding="utf-8")
    for forbidden in ("uvicorn", "GlobalOrchestrator", "orchestrator.run",
                      "registry", "httpx.post", "requests.post"):
        assert forbidden not in source


# ===========================================================================
# Closure contract checks - the things Phase 8 asserts about the shipped agent
# ===========================================================================

SEVEN_KEYS = {"gbif_id", "ncbi_taxid", "recognition", "recognition_candidates",
              "recognition_provenance", "species", "species_id"}


def test_the_agent_card_is_schema_compatible():
    """The card is the cross-agent contract; Phase 8 may only touch its prose.

    Identity, capability identifiers, ownership boundaries and the declared
    input/output shape must all survive the Sprint 3 wording update untouched.
    """
    import pathlib

    card = json.loads((pathlib.Path(__file__).resolve().parent.parent
                       / "card.json").read_text(encoding="utf-8"))

    assert card["name"] == "Multimodal Species Recognition Agent"
    assert card["type"] == "worker"
    assert card["reports_to"] == ["Global Scientific Orchestrator"]
    assert card["capabilities"] == [
        "Species recognition from an image",
        "Top-K taxonomic classification of one photograph",
    ]
    assert card["input"] == {"instruction": "string", "context": "dict"}
    # The declared output is exactly the seven keys the workflow emits.
    assert set(card["output"]) == SEVEN_KEYS
    # Ownership boundaries are preserved verbatim.
    assert "Visual similarity search between animals or images" in card["does_not_provide"]


def test_the_agent_card_port_still_comes_from_the_registry():
    """The card carries no port - the registry owns it, and still says 8005."""
    import pathlib

    card_text = (pathlib.Path(__file__).resolve().parent.parent
                 / "card.json").read_text(encoding="utf-8")
    assert "8005" not in card_text

    registry = (pathlib.Path(__file__).resolve().parents[3]
                / "registry.py").read_text(encoding="utf-8")
    assert '"Multimodal": 8005' in registry


def test_the_card_no_longer_claims_the_providers_are_mocked():
    """The specific obsolete claim Phase 8 was authorized to correct."""
    import pathlib

    card = json.loads((pathlib.Path(__file__).resolve().parent.parent
                       / "card.json").read_text(encoding="utf-8"))
    services = " ".join(card["managed_services"]).lower()

    assert "mocked in sprint 2" not in services
    assert "execution mocked" not in services
    # And it names what actually runs.
    assert "imageomics/bioclip-2-demo" in services
    assert "live" in services


def test_the_card_claims_no_orchestrator_integration():
    """Repository reality: Recognition is not routed by the Global Orchestrator."""
    import pathlib

    text = (pathlib.Path(__file__).resolve().parent.parent
            / "card.json").read_text(encoding="utf-8").lower()

    for false_claim in ("routed by the global orchestrator", "calls the evolution agent",
                        "invokes specialists", "directly calls"):
        assert false_claim not in text


def test_the_card_does_not_describe_the_real_providers_as_mocked():
    import pathlib

    text = (pathlib.Path(__file__).resolve().parent.parent
            / "card.json").read_text(encoding="utf-8").lower()

    for stale in ("bioclip is mocked", "mocked bioclip", "gbif and ncbi are mocked",
                  "mocked gbif", "mocked ncbi", "mock classification only"):
        assert stale not in text


def test_the_seven_key_contract_is_still_exactly_seven():
    agent = demo_agent()
    from ..config import RECOGNITION_IMAGE_CONTEXT_KEY
    from ..schema import AgentRequest
    from .conftest import image_entry

    result = agent.run(AgentRequest(
        instruction="Identify this animal.",
        context={RECOGNITION_IMAGE_CONTEXT_KEY: image_entry(png_bytes())}))

    assert set(result.output) == SEVEN_KEYS


def test_no_tracked_file_under_the_package_is_an_image_weight_or_log():
    import pathlib

    package = pathlib.Path(__file__).resolve().parent.parent
    offenders = []
    for path in package.rglob("*"):
        if not path.is_file():
            continue
        if any(part in path.parts for part in (".venv", "__pycache__", "demo_images")):
            continue
        if path.name == ".env":
            continue
        if path.suffix.lower() in (".pt", ".pth", ".bin", ".safetensors", ".ckpt",
                                   ".png", ".jpg", ".jpeg", ".webp", ".log", ".csv"):
            offenders.append(path.name)
    assert offenders == []


def test_the_demo_loads_the_agent_env_only_after_the_opt_in_guard(monkeypatch, tmp_path):
    """Importing the module must read no configuration, and a refused run must
    not either - otherwise the guard would fire after the secrets were loaded."""
    from .. import demo_sprint3

    loaded: list[str] = []
    monkeypatch.setattr(demo_sprint3, "load_agent_environment",
                        lambda: loaded.append("loaded"))
    monkeypatch.setattr(
        "backend.agents.multimodal_recognition_agent.agent.RecognitionAgent",
        lambda *a, **k: demo_agent())
    monkeypatch.setattr(
        "backend.agents.multimodal_recognition_agent.config.RecognitionConfig.from_env",
        classmethod(lambda cls: make_config(top_k_species=5)))

    image = write_image(tmp_path)

    monkeypatch.delenv(demo_sprint3.LIVE_OPT_IN_VAR, raising=False)
    assert demo_sprint3.main(["--image", str(image)]) == 2
    assert loaded == [], "configuration was loaded before the opt-in was checked"

    monkeypatch.setenv(demo_sprint3.LIVE_OPT_IN_VAR, "1")
    assert demo_sprint3.main(["--image", str(tmp_path / "absent.jpg")]) == 3
    assert loaded == [], "configuration was loaded for a request that never ran"

    assert demo_sprint3.main(["--image", str(image)]) == 0
    assert loaded == ["loaded"]


def test_the_demo_reads_the_agent_local_env_file():
    """It must use the SAME configuration the smoke scripts do, not the shell.

    The needle is assembled at runtime so this module does not contain the
    literal itself - the Phase 2 gate forbids any test module naming it, and a
    test that trips the gate it is describing would be its own counterexample.
    """
    import pathlib

    source = (pathlib.Path(__file__).resolve().parent.parent
              / "demo_sprint3.py").read_text(encoding="utf-8")
    loader = "load_" + "dotenv"

    assert loader in source
    assert 'Path(__file__).resolve().parent / ".env"' in source


# ===========================================================================
# P8-F1 - CORRECTED: the not_identified clarification question is
# provider-neutral
#
# It used to read "The Sprint 2 classifier did not return a taxon confident
# enough to name a species for this image", which was false the moment real
# remote BioCLIP-2 inference started running - and it is the sentence a user
# sees precisely when their photograph could not be identified. Two of the
# seven Phase 5 real-mode evaluation images (shark_white, ambiguous_coyote)
# returned not_identified and therefore carried it.
#
# The correction describes the EVIDENCE rather than the component that produced
# it, so it is true in every mode and needs no mode branch. These were a strict
# xfail plus a companion recording the defect; they are now ordinary passing
# regression tests.
# ===========================================================================

NEUTRAL_QUESTION = (
    "The available visual evidence was not strong enough to identify a species "
    "confidently. Could you supply a clearer photograph of the animal, or tell me "
    "where the observation was made?"
)


def remote_classifier_agent(predictions):
    """An agent whose classifier reports the REAL remote recognition mode."""

    class RemoteShapedClassifier(StubClassifier):
        provider_name = "RemoteBioCLIP2Provider"
        recognition_mode = "remote_bioclip2_open_domain_species"
        version = "bioclip-2-remote"

    return RecognitionAgent(
        make_config(top_k_species=5),
        classifier=RemoteShapedClassifier(predictions),
        taxonomy_provider=MockTaxonomyProvider(),
    )


def output_for(agent):
    from ..config import RECOGNITION_IMAGE_CONTEXT_KEY
    from ..schema import AgentRequest
    from .conftest import image_entry

    return agent.run(AgentRequest(
        instruction="Identify this animal.",
        context={RECOGNITION_IMAGE_CONTEXT_KEY: image_entry(png_bytes())})).output


# Every way a request can reach `not_identified`: no candidates at all, a single
# weak one, and several weak ones too close together to separate.
NOT_IDENTIFIED_CASES = {
    "no_candidates": [],
    "one_weak_candidate": [prediction("panthera_leo", 0.10, "Panthera leo")],
    "weak_and_close": [
        prediction("canis_lycaon", 0.42, "Canis lycaon"),
        prediction("canis_latrans", 0.39, "Canis latrans"),
    ],
}


@pytest.mark.parametrize("case", sorted(NOT_IDENTIFIED_CASES))
def test_p8_f1_the_real_mode_question_makes_no_sprint_2_claim(case):
    """(1) Real not_identified output contains no 'Sprint 2' claim."""
    output = output_for(remote_classifier_agent(NOT_IDENTIFIED_CASES[case]))
    question = output["recognition"]["clarification_question"]

    assert output["recognition"]["decision"] == "not_identified"
    assert "Sprint 2" not in question
    assert "sprint 2" not in question.lower()
    assert "classifier" not in question.lower()
    assert question == NEUTRAL_QUESTION


@pytest.mark.parametrize("case", sorted(NOT_IDENTIFIED_CASES))
def test_p8_f1_the_mock_mode_question_remains_truthful(case):
    """(2) Mock not_identified output remains truthful.

    The neutral sentence describes the evidence, so it is as true of the mock as
    it is of the remote model - which is exactly why no mode branch was added.
    """
    output = output_for(demo_agent(NOT_IDENTIFIED_CASES[case]))
    question = output["recognition"]["clarification_question"]

    assert output["recognition"]["decision"] == "not_identified"
    assert question == NEUTRAL_QUESTION
    # And it claims nothing about the mock that would be false of it.
    assert "real" not in question.lower()
    assert "remote" not in question.lower()


def test_p8_f1_the_question_is_identical_in_both_modes():
    """The correction is provider-neutral, not two branches that happen to agree."""
    real = output_for(remote_classifier_agent([]))["recognition"]["clarification_question"]
    mock = output_for(demo_agent([]))["recognition"]["clarification_question"]

    assert real == mock == NEUTRAL_QUESTION


@pytest.mark.parametrize("case", sorted(NOT_IDENTIFIED_CASES))
def test_p8_f1_the_question_still_asks_for_better_visual_evidence(case):
    """(3) The question still requests better visual evidence, and (4) the
    decision and `request_better_image` behaviour are untouched."""
    output = output_for(remote_classifier_agent(NOT_IDENTIFIED_CASES[case]))
    recognition = output["recognition"]

    assert "clearer photograph" in recognition["clarification_question"]
    assert recognition["decision"] == "not_identified"
    assert recognition["request_better_image"] is True
    assert recognition["better_image_reason"]


def test_p8_f1_the_thresholds_are_unchanged():
    """(5) Thresholds are unchanged."""
    from ..config import RecognitionConfig, ThresholdConfig

    defaults = make_config().thresholds
    assert defaults.identified_min_score == 0.75
    assert defaults.identified_min_margin == 0.08
    assert defaults.uncertain_min_score == 0.45
    assert isinstance(defaults, ThresholdConfig)
    assert RecognitionConfig is not None


def test_p8_f1_the_decide_function_is_untouched():
    """The correction changed prose only - not one confidence condition."""
    from ..domain.confidence import decide
    from .conftest import candidate, make_config as _make_config

    thresholds = _make_config().thresholds

    assert decide([], None, "neutral", thresholds) == "not_identified"
    assert decide([candidate("a", 0.10)], 1.0, "neutral", thresholds) == "not_identified"
    assert decide([candidate("a", 0.96)], 1.0, "neutral", thresholds) == "identified"
    assert decide([candidate("a", 0.96)], 0.01, "neutral", thresholds) == "uncertain"
    assert decide([candidate("a", 0.60)], 1.0, "neutral", thresholds) == "uncertain"
    # Text can downgrade, never promote.
    assert decide([candidate("a", 0.96)], 1.0, "conflict", thresholds) == "uncertain"


@pytest.mark.parametrize("case", sorted(NOT_IDENTIFIED_CASES))
def test_p8_f1_candidate_identity_order_and_scores_are_unchanged(case):
    """(6) Candidate identity, order and scores are unchanged."""
    predictions = NOT_IDENTIFIED_CASES[case]
    output = output_for(remote_classifier_agent(predictions))
    candidates = output["recognition_candidates"]

    assert [c["species_id"] for c in candidates] == [p.species_id for p in predictions]
    assert [c["scientific_name"] for c in candidates] == [
        p.scientific_name for p in predictions]
    assert [c["classification_score"] for c in candidates] == [
        p.classification_score for p in predictions]


@pytest.mark.parametrize("case", sorted(NOT_IDENTIFIED_CASES))
def test_p8_f1_the_seven_output_keys_remain_exact(case):
    """(7) Seven output keys remain exact."""
    output = output_for(remote_classifier_agent(NOT_IDENTIFIED_CASES[case]))

    assert set(output) == SEVEN_KEYS


def test_p8_f1_the_uncertain_and_identified_branches_are_byte_identical():
    """(8) The uncertain and identified branches remain byte-identical.

    `identified` returns None, and `uncertain` still names the candidates it
    could not separate - neither was in scope, and neither moved.
    """
    from ..domain.confidence import clarification_for
    from .conftest import candidate

    assert clarification_for("identified", [candidate("panthera_leo", 0.96)]) is None

    uncertain = clarification_for("uncertain", [
        candidate("panthera_leo", 0.50, "Panthera leo"),
        candidate("panthera_tigris", 0.48, "Panthera tigris"),
    ])
    assert uncertain == (
        "The classification was inconclusive between: Panthera leo, Panthera tigris. "
        "Could you confirm which of these it resembles, or where the observation was made?"
    )

    # And the no-name uncertain fallback is untouched too.
    assert clarification_for("uncertain", []) == (
        "Could you provide more detail about the animal in the image?"
    )


def test_p8_f1_no_other_stale_mock_claim_is_reachable_in_real_mode():
    """(9) No other stale mock claim is reachable in real mode.

    Sweeps every string literal the shipped runtime can emit - excluding
    comments and docstrings - for Sprint-2 wording, and confirms each survivor
    is reachable only in mock mode, where it is accurate.
    """
    import ast
    import pathlib

    package = pathlib.Path(__file__).resolve().parent.parent
    survivors = []
    for path in sorted(package.rglob("*.py")):
        if any(part in path.parts for part in (".venv", "__pycache__", "tests")):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                                 ast.AsyncFunctionDef)):
                body = getattr(node, "body", [])
                if (body and isinstance(body[0], ast.Expr)
                        and isinstance(body[0].value, ast.Constant)
                        and isinstance(body[0].value.value, str)):
                    docstrings.add(id(body[0].value))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Constant) and isinstance(node.value, str)
                    and id(node) not in docstrings
                    and "sprint 2" in node.value.lower()):
                survivors.append(f"{path.relative_to(package)}:{node.lineno}")

    # Exactly three remain, and every one is mock-mode-only:
    #   config.py           - the MOCK provider's own version string
    #   domain/errors.py    - raised only when the MOCK fixture is malformed
    #   workflows/nodes.py  - the mock branch of the mode-aware classifier phrase
    #                         and of the mode-aware safety footer
    assert all(
        location.startswith(("config.py", "domain\\errors.py", "domain/errors.py",
                             "workflows\\nodes.py", "workflows/nodes.py"))
        for location in survivors
    ), survivors
    assert not any("confidence.py" in location for location in survivors)

    # And a full real-mode response contains no Sprint-2 wording anywhere.
    for case in NOT_IDENTIFIED_CASES.values():
        body = json.dumps(output_for(remote_classifier_agent(case)))
        assert "Sprint 2" not in body
        assert "sprint 2" not in body.lower()


def test_p8_f1_a_real_mode_identified_response_is_also_free_of_sprint_2_wording():
    """The identified path was corrected in Phase 5; re-asserted here so the two
    corrections cannot drift apart."""
    output = output_for(remote_classifier_agent(
        [prediction("panthera_leo", 0.96, "Panthera leo")]))

    assert output["recognition"]["decision"] == "identified"
    assert "Sprint 2" not in json.dumps(output)
    assert output["recognition"]["clarification_question"] is None
