"""Controlled live smoke for the Recognition Agent's real GPT-5 mini path.

Exercises the production reasoning path end to end - planner, then grounded
explainer - through the real `AzureGPT5MiniProvider` and the normal agent
workflow. It reproduces no provider logic of its own: the adapter under test is
the one the service runs.

It replaces a one-call connectivity probe that could not demonstrate anything
Phase 2 asks for: no planner, no explainer, no call accounting, no plan source.

SAFETY

Nothing this script prints can carry a secret. It reports stage markers,
booleans, counters and durations - never the endpoint, the key, `.env`
contents, a prompt, a response body, image bytes, Base64, the shared context,
or a taxonomy identifier. Every line is flushed, so redirecting output cannot
hide where a run stopped.

USAGE

It refuses to run without an explicit opt-in, so no test run, import or stray
invocation can spend money. Run it as a module from the repository root, with
this agent's venv active:

    RECOGNITION_LIVE_SMOKE=1 python -u -m \
        backend.agents.multimodal_recognition_agent.smoke_test_azure

Redirect output somewhere OUTSIDE the repository. Exit status is 0 only when
both logical calls completed and every Phase 2 assertion held; any missing
evidence exits non-zero.
"""
from __future__ import annotations

import base64
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

AGENT_DIR = Path(__file__).resolve().parent

# This agent's own git-ignored .env - never `backend/.env`, which belongs to the
# orchestrator and holds none of this deployment's settings. Spelled out in full
# rather than reusing AGENT_DIR so the path is checkable without importing this
# module, which is what the offline tests do.
AGENT_ENV_PATH = Path(__file__).resolve().parent / ".env"

# The opt-in. Absent or not exactly "1", the script exits without calling out.
LIVE_OPT_IN_VAR = "RECOGNITION_LIVE_SMOKE"

EXPECTED_DEPLOYMENT_ALIAS = "umbrella-gpt5-mini"

DEMO_IMAGE = AGENT_DIR / "fixtures" / "demo_images" / "demo_identified.png"
INSTRUCTION = "What animal is in this photo? Please identify the species."


def emit(marker: str, value: object = "") -> None:
    """One redacted progress line, flushed immediately."""
    print(f"[smoke] {marker}: {value}" if value != "" else f"[smoke] {marker}",
          flush=True)


class _CountingClient:
    """Counts HTTP attempts around the real SDK client.

    Started and completed are counted separately, so a run that dies mid-request
    is distinguishable from one that never issued the request - the exact
    distinction the previous attempt could not make. It forwards to the real
    client and inspects nothing but call metadata.
    """

    def __init__(self, inner) -> None:
        self._inner = inner
        self.attempts_started = 0
        self.attempts_completed = 0
        self.store_flags: list[object] = []
        self.responses = self

    def create(self, **kwargs):
        self.attempts_started += 1
        self.store_flags.append(kwargs.get("store"))
        result = self._inner.responses.create(**kwargs)
        self.attempts_completed += 1
        return result


def main() -> int:
    if os.getenv(LIVE_OPT_IN_VAR) != "1":
        emit("ABORTED", f"live opt-in absent; set {LIVE_OPT_IN_VAR}=1 to authorize")
        return 2

    emit("stage", "loading agent-local .env")
    load_dotenv(AGENT_ENV_PATH)

    # Imported after the opt-in check so an accidental invocation touches
    # nothing and constructs nothing.
    from .adapters.reasoning_llm import AzureGPT5MiniProvider, AzureSettings
    from .agent import RecognitionAgent
    from .config import RecognitionConfig
    from .schema import AgentRequest, AgentStatus

    emit("stage", "resolving configuration")
    try:
        config = RecognitionConfig.from_env()
        settings = AzureSettings.from_env(
            timeout_seconds=config.reasoning_llm_timeout_seconds
        )
    except Exception as exc:
        # Type only. A configuration message may name a variable; it must never
        # be echoed here in case a future message quotes a value.
        emit("BLOCKED", f"configuration failed ({type(exc).__name__})")
        return 3

    emit("deployment_alias", settings.deployment)
    emit("deployment_alias_expected",
         settings.deployment == EXPECTED_DEPLOYMENT_ALIAS)
    emit("llm_mode", config.reasoning_llm_provider_mode)
    emit("bioclip_mode", config.bioclip_provider_mode)
    emit("taxonomy_mode", config.taxonomy_provider_mode)
    emit("call_budget", config.reasoning_llm_max_calls_per_request)

    if not DEMO_IMAGE.is_file():
        emit("BLOCKED", "demo image missing")
        return 3

    context = {
        "recognition_image": {
            "data_url": "data:image/png;base64,"
            + base64.b64encode(DEMO_IMAGE.read_bytes()).decode("ascii"),
            "filename": DEMO_IMAGE.name,
        }
    }

    emit("stage", "building real azure provider (lazy client)")
    provider = AzureGPT5MiniProvider(settings)
    # Force the lazy client into existence, then wrap it, so the production
    # construction path - including max_retries and timeout - is the one used.
    inner = provider._ensure_client()
    counter = _CountingClient(inner)
    provider._client = counter

    emit("planner_started", False)
    emit("explainer_started", False)
    emit("stage", "running agent (planner then grounded explainer)")

    started = time.perf_counter()
    try:
        result = RecognitionAgent(config, reasoning_llm=provider).run(
            AgentRequest(instruction=INSTRUCTION, context=context)
        )
    except Exception as exc:
        emit("FAIL", f"agent raised ({type(exc).__name__})")
        emit("logical_calls_started", provider.plan_calls + provider.explain_calls)
        emit("http_attempts_started", counter.attempts_started)
        emit("http_attempts_completed", counter.attempts_completed)
        return 4
    duration_ms = round((time.perf_counter() - started) * 1000)

    emit("planner_started", provider.plan_calls >= 1)
    emit("planner_completed", counter.attempts_completed >= 1)
    emit("explainer_started", provider.explain_calls >= 1)
    emit("explainer_completed", counter.attempts_completed >= 2)
    emit("planner_before_explainer",
         provider.plan_calls >= provider.explain_calls)
    emit("logical_calls", provider.plan_calls + provider.explain_calls)
    emit("http_attempts_started", counter.attempts_started)
    emit("http_attempts_completed", counter.attempts_completed)
    emit("no_transport_retry",
         counter.attempts_started
         == provider.plan_calls + provider.explain_calls)
    emit("store_false_on_every_request",
         all(flag is False for flag in counter.store_flags))
    emit("duration_ms", duration_ms)
    emit("status", result.status.value)

    if result.status is not AgentStatus.COMPLETED:
        emit("BLOCKED", "agent did not complete")
        return 5

    provenance = result.output["recognition_provenance"]
    emit("plan_source", provenance.get("plan_source"))
    emit("plan_rejected", provenance.get("plan_rejected"))
    emit("explanation_source", provenance.get("explanation_source"))
    emit("reasoning_llm_calls", provenance.get("reasoning_llm_calls"))
    emit("structured_output_parsed", provenance.get("plan_source") == "llm")
    emit("explanation_non_empty",
         bool((result.output["recognition"].get("explanation") or "").strip()))
    emit("seven_output_keys", sorted(result.output.keys()))

    checks = {
        "deployment_alias": settings.deployment == EXPECTED_DEPLOYMENT_ALIAS,
        "planner_completed": counter.attempts_completed >= 1,
        "explainer_completed": counter.attempts_completed >= 2,
        "exactly_two_logical_calls":
            provider.plan_calls + provider.explain_calls == 2,
        "exactly_two_http_attempts": counter.attempts_started == 2,
        "no_transport_retry":
            counter.attempts_started == counter.attempts_completed == 2,
        "plan_source_llm": provenance.get("plan_source") == "llm",
        "plan_not_rejected": provenance.get("plan_rejected") is False,
        "explanation_source_llm": provenance.get("explanation_source") == "llm",
        "store_false": all(flag is False for flag in counter.store_flags),
        "planner_before_explainer":
            provider.plan_calls >= provider.explain_calls,
    }
    for name, passed in checks.items():
        emit(f"check.{name}", passed)

    if all(checks.values()):
        emit("RESULT", "PASS")
        return 0
    emit("RESULT", "FAIL")
    emit("failed_checks", [name for name, ok in checks.items() if not ok])
    return 1


if __name__ == "__main__":
    sys.exit(main())
