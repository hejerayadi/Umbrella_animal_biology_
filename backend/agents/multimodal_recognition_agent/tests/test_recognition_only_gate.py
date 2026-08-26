from __future__ import annotations

import json
import pathlib
import re

import pytest

from ..adapters.bioclip import MockBioCLIP2Provider
from ..adapters.reasoning_llm import ALLOWED_PLAN_STEPS
from ..adapters.taxonomy import MockTaxonomyProvider
from ..agent import RecognitionAgent
from ..config import RECOGNITION_IMAGE_CONTEXT_KEY, ConfigError, RecognitionConfig
from ..schema import AgentRequest, AgentStatus
from .conftest import (
    FORBIDDEN_TRACKED_SUFFIXES,
    StubClassifier,
    forbidden_tracked,
    git_tracked_paths,
    image_entry,
    make_config,
    png_bytes,
    prediction,
)

PACKAGE = pathlib.Path(__file__).resolve().parent.parent
REPO_ROOT = PACKAGE.parent.parent.parent

# Everything the agent actually ships, minus its own installed dependencies and
# the generated demo images.
def shipped_files(*suffixes: str) -> list[pathlib.Path]:
    return [
        path
        for suffix in suffixes
        for path in PACKAGE.rglob(f"*{suffix}")
        if ".venv" not in path.parts
        and "__pycache__" not in path.parts
        and "demo_images" not in path.parts
    ]


def executable_text(path: pathlib.Path) -> str:
    """Everything in a Python file EXCEPT its comments and docstrings.

    The distinction is the whole point of this module. Documentation may - and
    should - record that the vector architecture was removed; executable code
    may not reference it. So identifiers, imports, attribute names, keywords and
    ordinary string literals are all checked, while prose is not.
    """
    import ast

    tree = ast.parse(path.read_text(encoding="utf-8"))
    docstrings: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef,
                             ast.AsyncFunctionDef)):
            body = getattr(node, "body", [])
            if (body and isinstance(body[0], ast.Expr)
                    and isinstance(body[0].value, ast.Constant)
                    and isinstance(body[0].value.value, str)):
                docstrings.add(id(body[0].value))

    parts: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            parts.append(node.id)
        elif isinstance(node, ast.Attribute):
            parts.append(node.attr)
        elif isinstance(node, ast.arg):
            parts.append(node.arg)
        elif isinstance(node, ast.keyword) and node.arg:
            parts.append(node.arg)
        elif isinstance(node, ast.alias):
            parts.append(node.name)
            parts.append(node.asname or "")
        elif isinstance(node, ast.ImportFrom):
            parts.append(node.module or "")
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            parts.append(node.name)
        elif isinstance(node, ast.Constant) and isinstance(node.value, str):
            if id(node) not in docstrings:
                parts.append(node.value)
    return "\n".join(parts).lower()


def assignment_lines(path: pathlib.Path) -> list[str]:
    """The NAME=VALUE lines of a dotenv file, without its commentary."""
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#") and "=" in line
    ]


def request_with(instruction="Identify this animal.", extra_context=None):
    context = {RECOGNITION_IMAGE_CONTEXT_KEY: image_entry(png_bytes())}
    context.update(extra_context or {})
    return AgentRequest(instruction=instruction, context=context)


def build_agent(predictions=None):
    return RecognitionAgent(
        make_config(),
        classifier=StubClassifier(predictions or []),
        taxonomy_provider=MockTaxonomyProvider(),
    )


# ===========================================================================
# 38 - no Qdrant anywhere in the Recognition runtime
# ===========================================================================

def test_no_executable_runtime_code_references_qdrant():
    """Comments and docstrings may record the removal. Code may not depend on it.

    Scoped to the shipped runtime: the tests below deliberately spell the word
    out in assertions, which is the opposite of depending on it.
    """
    offenders = [
        str(path.relative_to(PACKAGE))
        for path in shipped_files(".py")
        if path.parent.name != "tests" and "qdrant" in executable_text(path)
    ]
    assert not offenders, offenders


def test_no_test_module_imports_a_vector_client():
    """The suite may name Qdrant in an assertion; it may never import one."""
    pattern = re.compile(
        r"^\s*(?:import|from)\s+(qdrant_client|qdrant|faiss|chromadb|pinecone|torch"
        r"|open_clip|bioclip|pybioclip)\b",
        re.MULTILINE,
    )
    offenders = [
        path.name for path in shipped_files(".py")
        if pattern.search(path.read_text(encoding="utf-8"))
    ]
    assert not offenders, offenders


@pytest.mark.parametrize(
    "term", ["qdrant", "vector", "embedding", "nearest", "cosine", "collection"]
)
def test_no_executable_runtime_code_references_the_vector_architecture(term):
    """The shipped runtime only - the tests deliberately name what must be
    absent, and are checked separately by the test above."""
    offenders = []
    for path in shipped_files(".py"):
        if path.parent.name == "tests":
            continue
        if term in executable_text(path):
            offenders.append(str(path.relative_to(PACKAGE)))
    assert not offenders, offenders


def test_no_qdrant_module_file_survives():
    for removed in ("adapters/qdrant_mock.py", "adapters/qdrant_real.py",
                    "adapters/retrieval.py", "fixtures/mock_references.json",
                    "fixtures/mock_embeddings.json"):
        assert not (PACKAGE / removed).exists(), removed


def test_no_qdrant_dependency_is_declared():
    """Checked on the DECLARED requirement lines, not on the explanatory comments.

    Same rule the `.env.example` gate below already uses, and the same rule
    `executable_text` applies to source: documentation may describe the
    architecture - including naming what deliberately is NOT installed - while
    the declarations themselves may not pull any of it in.
    """
    lines = [
        line.strip().lower()
        for line in (PACKAGE / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    declared = "\n".join(lines)
    for forbidden in ("qdrant", "torch", "open-clip", "open_clip", "bioclip",
                      "faiss", "chromadb", "pinecone", "weaviate", "milvus"):
        assert forbidden not in declared, forbidden


def test_no_qdrant_or_vector_variable_is_declared_in_the_environment_example():
    """Checked on the variable declarations, not on the explanatory comments."""
    declared = "\n".join(assignment_lines(PACKAGE / ".env.example")).upper()
    for forbidden in ("QDRANT", "VECTOR", "COLLECTION", "EMBEDDING_DIMENSION",
                      "DATASET_VERSION", "DISTANCE", "RETRIEVAL", "SEED_OVERRIDES",
                      "REFERENCES_PER_SPECIES"):
        assert forbidden not in declared, forbidden


def test_the_config_object_exposes_no_vector_field():
    config = make_config()
    for removed in ("qdrant", "retrieval_mode", "mock_embedding_dimension",
                    "max_references_per_species", "image_seed_overrides",
                    "mock_embedding_dimension_is_local_default"):
        assert not hasattr(config, removed), removed


def test_the_config_ignores_any_qdrant_variable_left_in_the_environment(monkeypatch):
    """An operator's stale .env cannot switch a vector path back on, because
    there is no code that reads one."""
    for name, value in {
        "QDRANT_URL": "https://example.invalid",
        "QDRANT_COLLECTION": "species_refs",
        "QDRANT_EXPECTED_DIMENSION": "512",
        "RECOGNITION_RETRIEVAL_MODE": "real",
        "MOCK_EMBEDDING_DIMENSION": "512",
    }.items():
        monkeypatch.setenv(name, value)

    config = RecognitionConfig.from_env()
    assert config.recognition_mode == "mock_classification"
    assert "qdrant" not in json.dumps(config.__dict__, default=str).lower()


def test_no_qdrant_appears_in_a_response():
    result = build_agent([prediction("panthera_leo", 0.96, "Panthera leo")]).run(request_with())
    assert "qdrant" not in json.dumps(result.output, default=str).lower()


def test_the_card_advertises_no_vector_service():
    card = json.loads((PACKAGE / "card.json").read_text(encoding="utf-8"))
    services = json.dumps(card["managed_services"]).lower()
    for forbidden in ("qdrant", "vector", "retrieval", "collection"):
        assert forbidden not in services, forbidden


# ===========================================================================
# 39 - similarity is not an advertised capability, nor a reachable workflow
# ===========================================================================

def test_similarity_is_not_an_advertised_capability():
    card = json.loads((PACKAGE / "card.json").read_text(encoding="utf-8"))
    capabilities = json.dumps(card["capabilities"]).lower()
    for forbidden in ("similar", "similarity", "nearest", "resembl"):
        assert forbidden not in capabilities, forbidden


def test_the_card_states_what_the_agent_does_not_provide():
    card = json.loads((PACKAGE / "card.json").read_text(encoding="utf-8"))
    declined = json.dumps(card["does_not_provide"]).lower()
    assert "similarity search" in declined
    assert "vector" in declined


def test_similarity_is_not_a_plan_step():
    for step in ALLOWED_PLAN_STEPS:
        assert "similar" not in step
        assert "retriev" not in step
        assert "embed" not in step


def test_a_similarity_request_is_declined_in_the_response():
    result = build_agent([prediction("panthera_leo", 0.96, "Panthera leo")]).run(
        request_with("Which species look similar to this one?")
    )

    recognition = result.output["recognition"]
    assert result.status is AgentStatus.COMPLETED
    assert recognition["unsupported_capability"] == "visual_similarity_search"
    assert any("not a capability" in w for w in recognition["warnings"])
    # And it still answered the part it does own.
    assert recognition["decision"] == "identified"


def test_a_plain_request_declines_nothing():
    result = build_agent([prediction("panthera_leo", 0.96, "Panthera leo")]).run(
        request_with("Identify this animal.")
    )
    assert "unsupported_capability" not in result.output["recognition"]


def test_no_output_field_names_similarity():
    result = build_agent([prediction("panthera_leo", 0.96, "Panthera leo")]).run(request_with())
    serialized = json.dumps(result.output, default=str)

    # The word may appear only inside the honest refusal string, never as a
    # field name or a score label.
    assert "similarity_score" not in serialized
    assert "similarity_is_probability" not in serialized
    assert "recognition_similar" not in serialized


# ===========================================================================
# 36 / 37 - real BioCLIP-2 stays behind `remote`; live GBIF/NCBI now real (P4)
# ===========================================================================

@pytest.mark.parametrize("mode", ["real", "bioclip2", "torch", "hf", ""])
def test_a_non_mock_bioclip_mode_is_refused_at_startup(monkeypatch, mode):
    monkeypatch.setenv("BIOCLIP_PROVIDER_MODE", mode or "real")
    with pytest.raises(ConfigError) as caught:
        RecognitionConfig.from_env()
    assert "BIOCLIP_PROVIDER_MODE" in str(caught.value)


@pytest.mark.parametrize("mode", ["live", "gbif", "http"])
def test_a_non_mock_taxonomy_mode_is_refused_at_startup(monkeypatch, mode):
    """`real` is excluded here since Phase 4 implemented it - see
    test_phase1_config_hardening.py for its selectable/credential tests."""
    monkeypatch.setenv("TAXONOMY_PROVIDER_MODE", mode)
    with pytest.raises(ConfigError) as caught:
        RecognitionConfig.from_env()
    assert "TAXONOMY_PROVIDER_MODE" in str(caught.value)


def test_no_module_can_reach_a_biological_service():
    """No IUCN host anywhere in the shipped runtime, and GBIF/NCBI hosts stay
    confined to the one module that owns that boundary.

    Phase 3 narrowed this gate for BioCLIP-2's Space host; Phase 4 does the same
    for GBIF/NCBI. IUCN is still not integrated anywhere, so it stays forbidden
    everywhere with no allowlist at all.
    """
    forbidden_hosts = ("apiv3.iucnredlist.org", "iucnredlist.org")

    # The classification boundary, and nothing else, may name the BioCLIP Space.
    bioclip_host_allowed = {"config.py", "bioclip.py"}
    bioclip_hosts = ("huggingface.co", "imageomics")

    # The taxonomy boundary, and nothing else, may name GBIF/NCBI hosts.
    taxonomy_host_allowed = {"taxonomy.py"}
    taxonomy_hosts = ("api.gbif.org", "gbif.org", "eutils.ncbi.nlm.nih.gov",
                      "ncbi.nlm.nih.gov")

    offenders = []
    for path in shipped_files(".py"):
        if path.name == "smoke_test_azure.py":  # a standalone script, never imported
            continue
        if path.parent.name == "tests":  # this file names the hosts on purpose
            continue
        text = path.read_text(encoding="utf-8").lower()
        offenders += [f"{path.name}: {host}" for host in forbidden_hosts if host in text]
        if path.name not in bioclip_host_allowed:
            offenders += [f"{path.name}: {host}" for host in bioclip_hosts if host in text]
        if path.name not in taxonomy_host_allowed:
            offenders += [f"{path.name}: {host}" for host in taxonomy_hosts if host in text]
    assert not offenders, offenders


def test_no_http_client_is_imported_at_module_level_by_the_taxonomy_adapter():
    """Phase 4's real providers need an HTTP client, but importing this module
    must still open no connection - so the import may only happen lazily,
    inside a method, never at module scope. Same rule bioclip.py's
    gradio_client import already follows.
    """
    source = (PACKAGE / "adapters" / "taxonomy.py").read_text(encoding="utf-8")
    # Column-anchored: only a TOP-LEVEL import is caught. An indented one
    # inside a function/method is exactly what Phase 4 is allowed to do.
    pattern = re.compile(
        r"^(?:import|from)\s+(requests|httpx|urllib|http|socket|aiohttp)\b",
        re.MULTILINE,
    )
    assert pattern.search(source) is None


def test_the_taxonomy_fixture_says_it_is_not_live_data():
    fixture = json.loads(
        (PACKAGE / "fixtures" / "mock_taxonomy.json").read_text(encoding="utf-8")
    )
    note = fixture["_fixture_note"].lower()
    assert "not retrieved from gbif" in note


def test_the_classification_fixture_says_it_is_not_model_output():
    fixture = json.loads(
        (PACKAGE / "fixtures" / "mock_bioclip_predictions.json").read_text(encoding="utf-8")
    )
    assert "test oracle" in fixture["_fixture_note"].lower()
    assert fixture["recognition_mode"] == "mock_classification"


# ===========================================================================
# 35 - provenance names all three mocks, every time
# ===========================================================================

@pytest.mark.parametrize(
    "instruction",
    ["Identify this animal.", "Is this a lion?", "This is a polar bear."],
)
def test_every_completed_response_declares_all_three_mocks(instruction):
    result = build_agent([prediction("panthera_leo", 0.96, "Panthera leo")]).run(
        request_with(instruction)
    )
    provenance = result.output["recognition_provenance"]

    assert provenance["model_target"] == "BioCLIP-2"
    assert provenance["recognition_mode"] == "mock_classification"
    assert provenance["gbif_mode"] == "mock"
    assert provenance["ncbi_mode"] == "mock"


def test_a_not_identified_response_still_declares_the_mocks():
    provenance = build_agent([]).run(request_with()).output["recognition_provenance"]
    assert provenance["recognition_mode"] == "mock_classification"
    assert provenance["gbif_mode"] == "mock"
    assert provenance["ncbi_mode"] == "mock"


def test_the_mock_mode_string_is_never_plain_classification():
    """`mock_classification` and `classification` are different claims."""
    provider = MockBioCLIP2Provider()
    assert provider.recognition_mode == "mock_classification"
    assert provider.recognition_mode != "classification"


def test_the_score_is_labelled_a_test_score_not_a_probability():
    result = build_agent([prediction("panthera_leo", 0.96, "Panthera leo")]).run(request_with())
    provenance = result.output["recognition_provenance"]

    assert provenance["score_is_probability"] is False
    assert provenance["score_kind"] == "deterministic_sprint2_test_score"
    assert result.output["recognition"]["score_is_probability"] is False


# ===========================================================================
# 31 / 32 / 33 - service boundary and cross-agent contract
# ===========================================================================

def test_recognition_is_still_registered_on_port_8005():
    """Read-only check against the shared registry: the port did not move."""
    from backend.registry import _AGENT_FOLDERS, _AGENT_PORTS

    assert _AGENT_PORTS["Multimodal"] == 8005
    assert _AGENT_FOLDERS["Multimodal"] == "multimodal_recognition_agent"


def test_the_documented_run_command_names_port_8005():
    for document in ("api.py", "description.md"):
        text = (PACKAGE / document).read_text(encoding="utf-8")
        assert "--port 8005" in text, document


def test_a_completed_output_merges_into_a_shared_context():
    """The orchestrator merges `output` into the shared context. Every key must
    be a plain JSON-serializable value, and `species` must be present for the
    downstream agents that read it."""
    result = build_agent([prediction("panthera_leo", 0.96, "Panthera leo")]).run(request_with())

    shared: dict = {"user_question": "what is this?"}
    shared.update(result.output)

    assert shared["species"] == "Panthera leo"
    assert shared["user_question"] == "what is this?"
    # Round-trips through JSON without a custom encoder.
    assert json.loads(json.dumps(shared)) == shared


def test_the_agent_imports_no_peer_and_holds_no_peer_url():
    forbidden = ("localhost:800", "127.0.0.1:800", "_AGENT_URL",
                 "backend.agents.genome_agent", "backend.agents.evolution_agent",
                 "backend.agents.biodiversity_agent", "backend.registry",
                 "backend.orchestrator")
    offenders = []
    for path in shipped_files(".py"):
        if path.parent.name == "tests":
            continue
        text = path.read_text(encoding="utf-8")
        offenders += [f"{path.name}: {name}" for name in forbidden if name in text]
    assert not offenders, offenders


def test_delegation_is_a_capability_hint_not_an_agent_address():
    result = build_agent([prediction("panthera_leo", 0.96, "Panthera leo")]).run(
        request_with("What is the evolutionary history of this animal?")
    )

    assert result.status is AgentStatus.NEEDS_AGENT
    assert result.target_agent == "Evolution"
    assert "http" not in (result.prompt_to_target_agent or "")
    assert "800" not in (result.prompt_to_target_agent or "")


# ===========================================================================
# 34 / 40 - no leakage, and every branch terminates
# ===========================================================================

def test_no_response_contains_image_bytes_or_base64():
    result = build_agent([prediction("panthera_leo", 0.96, "Panthera leo")]).run(request_with())
    serialized = json.dumps(result.output, default=str)

    for forbidden in ("data:image", "base64", "iVBOR", "image_bytes"):
        assert forbidden not in serialized, forbidden


@pytest.mark.parametrize(
    "instruction, context",
    [
        ("Identify this animal.", None),                       # completed
        ("What is the genome of this animal?", None),          # needs_agent
        ("Identify this animal.", {}),                         # failed
        ("", None),                                            # failed: no text
    ],
)
def test_every_branch_terminates_with_the_shared_contract(instruction, context):
    agent = build_agent([prediction("panthera_leo", 0.96, "Panthera leo")])
    request = (
        AgentRequest(instruction=instruction, context=context)
        if context is not None else request_with(instruction)
    )
    result = agent.run(request)

    assert result.status in (AgentStatus.COMPLETED, AgentStatus.NEEDS_AGENT,
                             AgentStatus.FAILED)
    assert hasattr(result, "output")
    if result.status is AgentStatus.COMPLETED:
        assert isinstance(result.output, dict)


def test_no_secret_model_weight_or_raw_image_is_committed():
    """`.env` exists locally and is git-ignored; nothing else may be added.

    The rule is about what is COMMITTED, so it asks git rather than walking the
    filesystem. Generated and downloaded material - `fixtures/demo_images/` and
    the Sprint 4 benchmark's `evaluation/assets/` - lives on disk in git-ignored
    directories and is correctly invisible here.
    """
    ignored = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8").splitlines()
    assert ".env" in [line.strip() for line in ignored], "the .env file must stay ignored"
    agent_ignore = (PACKAGE / ".gitignore").read_text(encoding="utf-8")
    assert "fixtures/demo_images/" in agent_ignore

    offenders = forbidden_tracked(git_tracked_paths(PACKAGE))
    assert offenders == [], f"forbidden files are tracked under the package: {offenders}"

    assert ".env" not in {path.name for path in git_tracked_paths(PACKAGE)}


@pytest.mark.parametrize(
    "name",
    ["leaked.png", "photo.JPG", "sample.jpeg", "frame.webp",
     "model.pt", "weights.safetensors", "checkpoint.ckpt", "adapter.bin",
     "run.log"],
)
def test_a_tracked_forbidden_file_would_fail_the_gate(name):
    """The gate must still bite. Fed a tracked path with a forbidden suffix, the
    predicate has to report it - otherwise moving from a filesystem walk to git
    would have quietly disarmed the rule instead of sharpening it."""
    assert forbidden_tracked([pathlib.Path("evaluation/assets") / name]) == [name]


def test_a_forbidden_file_is_caught_even_inside_an_ignored_directory_path():
    """Directory name no longer buys an exemption. Under the old filesystem walk,
    anything under `demo_images/` was skipped by name; if such a file were ever
    genuinely committed, the gate would have missed it. Git-truth does not care
    where the file sits - only whether it is tracked."""
    assert forbidden_tracked(
        [pathlib.Path("fixtures/demo_images/demo_identified.png")]
    ) == ["demo_identified.png"]


def test_the_ignored_benchmark_assets_do_not_fail_the_gate():
    """The Sprint 4 benchmark downloads real photographs into
    `evaluation/assets/`. They are git-ignored and never committed, so they must
    not trip a gate about what is committed - which is exactly what they used to
    do when the gate walked the filesystem."""
    assets = PACKAGE / "evaluation" / "assets"
    tracked_names = {str(path) for path in git_tracked_paths(PACKAGE)}
    assert not any("evaluation/assets" in name.replace("\\", "/")
                   for name in tracked_names), "a benchmark asset is tracked"

    if assets.is_dir():
        on_disk = [p for p in assets.iterdir() if p.is_file()]
        # Whether or not the fetch has been run, nothing here may be tracked.
        assert forbidden_tracked(git_tracked_paths(PACKAGE)) == []
        for path in on_disk:
            assert path.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"), path.name


def test_the_benchmark_asset_directory_is_ignored():
    ignore = (PACKAGE / "evaluation" / ".gitignore").read_text(encoding="utf-8")
    assert "assets/" in ignore
    assert "results/" in ignore


def test_no_forbidden_file_is_tracked_anywhere_under_the_package():
    """The live assertion, stated once more on its own so a failure names it
    plainly rather than arriving inside a longer gate."""
    tracked = git_tracked_paths(PACKAGE)
    assert tracked, "git reported no tracked files - the gate would pass vacuously"
    assert forbidden_tracked(tracked, FORBIDDEN_TRACKED_SUFFIXES + (".csv",)) == []


def test_the_environment_example_holds_no_secret_value():
    for line in assignment_lines(PACKAGE / ".env.example"):
        name, _, value = line.partition("=")
        if name.strip().upper() in ("AZURE_OPENAI_API_KEY", "AZURE_OPENAI_BASE_URL"):
            assert value.strip() == "", f"{name} must ship empty"
