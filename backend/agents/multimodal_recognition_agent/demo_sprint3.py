"""Sprint 3 demonstration entry point for the standalone Recognition Agent.

Runs ONE real recognition request against the providers the environment already
configures, and prints only safe evidence. It is a demonstration of the agent
that exists - not a test, not a benchmark, and not an orchestrator.

    RECOGNITION_LIVE_SMOKE=1 python -m backend.agents.multimodal_recognition_agent.demo_sprint3 \
        --image path/to/animal.jpg \
        --instruction "Identify the animal in this image."

Refuses to run without the explicit opt-in, the same guard the two smoke
scripts use, so no import, test run or stray invocation can reach a live
service.

The image path is supplied by the caller and is never committed: nothing under
this package tracks an image, and this script writes no file at all.

WHAT IT PRINTS: status, decision, Top-K species and scores, GBIF/NCBI
identifiers, provider modes, LLM call count, latency and warnings.

WHAT IT NEVER PRINTS: Base64, image bytes, credentials, endpoints, .env values,
prompts, or a raw provider error body. A failure is reported by its controlled
code alone.

BOUNDED EXECUTION: the whole request runs under DEMO_DEADLINE_SECONDS (240s) in
this process, on top of the per-provider deadlines the agent already enforces.
Exceeding it exits non-zero rather than waiting.

EXIT CODES
    0  the request completed, or delegated to a capability
    1  the request failed under a controlled error code
    2  the live opt-in was absent
    3  the image path is missing or unusable
    4  the bounded deadline was exceeded

This never starts the Global Orchestrator. The Recognition Agent is standalone;
when it needs another capability it says so in its result and stops.
"""
from __future__ import annotations

import argparse
import base64
import concurrent.futures
import os
import sys
import time
from pathlib import Path

# The opt-in. Absent or not exactly "1", nothing is contacted.
LIVE_OPT_IN_VAR = "RECOGNITION_LIVE_SMOKE"

# The demonstration's own ceiling, above whatever the providers enforce.
DEMO_DEADLINE_SECONDS = 240.0

MEDIA_TYPE_BY_SUFFIX = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

DEFAULT_INSTRUCTION = "Identify the animal in this image."


def parse_arguments(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="demo_sprint3",
        description="One real standalone Recognition request, printed safely.",
    )
    parser.add_argument(
        "--image", required=True,
        help="Path to a local JPEG, PNG or WEBP photograph of an animal.",
    )
    parser.add_argument(
        "--instruction", default=DEFAULT_INSTRUCTION,
        help="The text instruction accompanying the image.",
    )
    return parser.parse_args(argv)


def resolve_image(raw_path: str) -> tuple[Path | None, str | None]:
    """Validate the caller's path. Returns `(path, problem)`.

    Deliberately dynamic: there is no default and no built-in path, so the
    demonstration cannot depend on one machine's filesystem.
    """
    if not raw_path or not raw_path.strip():
        return None, "no image path was supplied"

    path = Path(raw_path).expanduser()
    if not path.exists():
        return None, "the image path does not exist"
    if not path.is_file():
        return None, "the image path is not a file"
    if path.suffix.lower() not in MEDIA_TYPE_BY_SUFFIX:
        return None, (
            "unsupported image type; accepted suffixes are "
            + ", ".join(sorted(MEDIA_TYPE_BY_SUFFIX))
        )
    if path.stat().st_size == 0:
        return None, "the image file is empty"
    return path, None


def build_request(path: Path, instruction: str):
    """Wrap the caller's image in the shared request contract.

    The data URL is built here and never printed: it is the one value in this
    script that must not reach a terminal.
    """
    from .config import RECOGNITION_IMAGE_CONTEXT_KEY
    from .schema import AgentRequest

    raw = path.read_bytes()
    media_type = MEDIA_TYPE_BY_SUFFIX[path.suffix.lower()]
    data_url = f"data:{media_type};base64," + base64.b64encode(raw).decode("ascii")

    return AgentRequest(
        instruction=instruction,
        context={RECOGNITION_IMAGE_CONTEXT_KEY: {
            "data_url": data_url,
            "filename": path.name,
        }},
    ), len(raw), media_type


def load_agent_environment() -> None:
    """Load the agent's own git-ignored `.env`, as the smoke scripts do.

    Called from `main` AFTER the opt-in check, never at import, so importing
    this module reads no configuration and the offline suite stays untouched by
    it. Without this the demonstration would silently run on whatever defaults
    the shell happens to carry - which in practice means mock mode, honestly
    reported but not the thing anyone asked to see.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:  # the demonstration still runs on the ambient env
        return
    env_path = Path(__file__).resolve().parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)


def emit(label: str, value) -> None:
    """The only way anything reaches the terminal."""
    print(f"[demo] {label}: {value}")


def run_bounded(agent, request):
    """One request under the demonstration's own ceiling."""
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(agent.run, request)
        try:
            return future.result(timeout=DEMO_DEADLINE_SECONDS), time.monotonic() - started
        except concurrent.futures.TimeoutError:
            return None, time.monotonic() - started


def report_completed(output: dict) -> None:
    """Print the safe fields of a completed result, and only those."""
    recognition = output.get("recognition", {}) or {}
    provenance = output.get("recognition_provenance", {}) or {}
    candidates = output.get("recognition_candidates", []) or []

    emit("decision", recognition.get("decision"))
    emit("text_alignment", recognition.get("text_alignment"))
    emit("species", output.get("species"))
    emit("species_id", output.get("species_id"))
    emit("gbif_id", output.get("gbif_id"))
    emit("ncbi_taxid", output.get("ncbi_taxid"))
    emit("score_is_probability", recognition.get("score_is_probability"))

    emit("top_k", f"{len(candidates)} candidate(s)")
    for position, candidate in enumerate(candidates, start=1):
        print(
            f"[demo]   {position}. {candidate.get('scientific_name')}"
            f"  score={candidate.get('classification_score')}"
            f"  gbif={candidate.get('gbif_id')}"
            f"  ncbi={candidate.get('ncbi_taxid')}"
            f"  taxonomy={candidate.get('taxonomy_status')}"
        )

    emit("recognition_provider", provenance.get("recognition_provider"))
    emit("recognition_mode", provenance.get("recognition_mode"))
    emit("gbif_mode", provenance.get("gbif_mode"))
    emit("ncbi_mode", provenance.get("ncbi_mode"))
    emit("taxonomy_executed", provenance.get("taxonomy_executed"))
    emit("taxonomy_degraded", provenance.get("taxonomy_degraded"))
    emit("reasoning_llm_provider", provenance.get("reasoning_llm_provider"))
    emit("reasoning_llm_calls", provenance.get("reasoning_llm_calls"))
    emit("plan_source", provenance.get("plan_source"))
    emit("explanation_source", provenance.get("explanation_source"))
    emit("output_keys", sorted(output))

    for warning in recognition.get("warnings", []) or []:
        emit("warning", warning)
    if recognition.get("clarification_question"):
        emit("clarification_question", recognition["clarification_question"])
    if recognition.get("unsupported_capability"):
        emit("unsupported_capability", recognition["unsupported_capability"])


def main(argv=None) -> int:
    arguments = parse_arguments(argv)

    if os.getenv(LIVE_OPT_IN_VAR) != "1":
        emit("ABORTED", f"live opt-in absent; set {LIVE_OPT_IN_VAR}=1 to authorize "
                        "real provider calls")
        return 2

    path, problem = resolve_image(arguments.image)
    if path is None:
        # The caller's own path is echoed back - it is theirs, not a secret -
        # but nothing is read from it and nothing else is disclosed.
        emit("ABORTED", problem)
        return 3

    # Only now, once the caller has opted in and supplied a usable image.
    load_agent_environment()

    from .agent import RecognitionAgent
    from .config import RecognitionConfig
    from .schema import AgentStatus

    config = RecognitionConfig.from_env()
    emit("bioclip_provider_mode", config.bioclip_provider_mode)
    emit("taxonomy_provider_mode", config.taxonomy_provider_mode)
    emit("reasoning_llm_provider_mode", config.reasoning_llm_provider_mode)
    emit("top_k_species", config.top_k_species)
    emit("deadline_seconds", DEMO_DEADLINE_SECONDS)

    request, byte_size, media_type = build_request(path, arguments.instruction)
    emit("image", f"{media_type}, {byte_size} bytes")

    agent = RecognitionAgent(config)
    result, seconds = run_bounded(agent, request)
    emit("latency_seconds", round(seconds, 3))

    if result is None:
        emit("FAILED", "the demonstration deadline was exceeded")
        return 4

    emit("status", result.status.value)

    if result.status is AgentStatus.NEEDS_AGENT:
        # Recognition names a capability it does not own. It does NOT call it:
        # routing is the Global Orchestrator's job, and this demonstration
        # deliberately stops here.
        emit("needs_capability", result.target_agent)
        emit("delegation_prompt_present", bool(result.prompt_to_target_agent))
        emit("note", "Recognition emits a capability hint and stops; it never "
                     "invokes another agent itself.")
        return 0

    if result.status is AgentStatus.FAILED:
        output = result.output if isinstance(result.output, dict) else {}
        # The controlled code and its fixed message. Never a provider body.
        emit("error_code", output.get("error_code"))
        emit("error", output.get("error"))
        return 1

    report_completed(result.output or {})
    return 0


if __name__ == "__main__":
    sys.exit(main())
