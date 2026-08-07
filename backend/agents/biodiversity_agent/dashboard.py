"""Streamlit test harness for the Biodiversity Agent Orchestrator.

Run it from the repo root:

    streamlit run backend/agents/biodiversity_agent/dashboard.py

The page is a manual replacement for the Global Scientific Orchestrator:
it takes a free-form user prompt, uses the configured LLM to detect
the intent (which of the four biodiversity features to run, plus the
species / region), then dispatches an ``AgentRequest`` to the real
``BiodiversityOrchestrator`` and renders the result:

- The intent classification JSON (what the LLM decided).
- The list of LangGraph nodes that executed (proves routing works).
- The interactive folium map (via the ``map_url`` in the result).
- The full ``AgentResult`` dict, pretty-printed.

Everything else stays untouched - the orchestrator, workers and
services are called exactly as they will be in production. The
dashboard is a test client, not a new codepath.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

import streamlit as st

# Support running this file both as a module (``streamlit run backend/...``)
# and directly (``streamlit run dashboard.py`` from the agent folder).
import sys
_ROOT = Path(__file__).resolve().parents[3]  # repo root
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.agents.biodiversity_agent.framework.llm_client import (  # noqa: E402
    LLMUnavailable,
    get_llm,
)
from backend.agents.biodiversity_agent.orchestrator import (  # noqa: E402
    BiodiversityOrchestrator,
)
from backend.agents.biodiversity_agent.schema import (  # noqa: E402
    AgentRequest,
    AgentStatus,
    BiodiversityFeature,
)


# ---------- page config ----------

st.set_page_config(
    page_title="Biodiversity Orchestrator — Sprint 2 demo",
    page_icon=":world_map:",
    layout="wide",
)

# One orchestrator per Streamlit session so we do not rebuild the
# LangGraph state machine on every keystroke.
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = BiodiversityOrchestrator()


# ---------- intent detection (mimics what the Global Orchestrator will do) ----------


_INTENT_SYSTEM_PROMPT = """You are the Global Scientific Orchestrator for the Umbrella BioHub.
Given a user's free-form question, decide which BIODIVERSITY feature to run and
extract the parameters. Respond with a strict JSON object, no prose, no code
fences, no extra keys:

{
  "feature": "species_distribution_map" | "habitat_visualization" | "biodiversity_hotspots" | "migration_analysis",
  "species_name": "<scientific or common name, or null>",
  "region": "<continent or country, default 'global'>"
}

Rules:
- species_distribution_map: user asks WHERE a species is observed (point map).
- habitat_visualization: user asks about a species HABITAT / conservation status.
- biodiversity_hotspots: user asks about REGIONS with many species (no single species).
- migration_analysis: user asks about MIGRATION / seasonal movement of a species.
- If the user asks about hotspots, species_name is null.
- Output ONLY the JSON. No markdown."""


async def _classify_intent(prompt: str) -> dict:
    llm = get_llm()
    from langchain_core.messages import HumanMessage, SystemMessage

    response = await llm.ainvoke(
        [SystemMessage(content=_INTENT_SYSTEM_PROMPT), HumanMessage(content=prompt)]
    )
    text = response.content.strip()
    # Strip fences the LLM sometimes ignores rules and adds.
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text)


# ---------- helpers ----------


def _dataclass_to_dict(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return {k: _dataclass_to_dict(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {k: _dataclass_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_dataclass_to_dict(v) for v in obj]
    if isinstance(obj, AgentStatus):
        return obj.value
    return obj


async def _run_with_trace(request: AgentRequest) -> tuple[Any, list[str]]:
    """Run the orchestrator while collecting every LangGraph node that fires."""

    orch: BiodiversityOrchestrator = st.session_state.orchestrator
    trace: list[str] = []

    # ``astream`` yields one dict per node execution. The dict keys are
    # the node names LangGraph advanced through this tick.
    async for chunk in orch._graph.astream(  # noqa: SLF001 — demo only
        __import__(
            "backend.agents.biodiversity_agent.orchestrator.biodiversity_orchestrator",
            fromlist=["BiodiversityState"],
        ).BiodiversityState(request=request),
        stream_mode="updates",
    ):
        for node_name in chunk.keys():
            trace.append(node_name)

    # Final state has the aggregated result under ``result``.
    # We re-run the full ainvoke to grab it cleanly; the astream loop
    # above was only for the trace.
    result = await orch.run(request)
    return result, trace


def _render_map_url(map_url: str | None) -> None:
    if not map_url:
        st.info("No map to display for this feature.")
        return
    if map_url.startswith("file://"):
        local_path = map_url.replace("file://", "")
        if Path(local_path).exists():
            html = Path(local_path).read_text(encoding="utf-8")
            st.components.v1.html(html, height=500, scrolling=False)
            return
    st.markdown(f"[Open map]({map_url})")


# ---------- sidebar ----------

st.sidebar.title("Biodiversity Orchestrator")
st.sidebar.markdown(
    "**Sprint 2 test harness.** This page simulates the Global "
    "Orchestrator: it turns a natural-language prompt into an "
    "`AgentRequest` and dispatches it to your real LangGraph."
)

preset = st.sidebar.selectbox(
    "Pick a preset prompt",
    [
        "— custom —",
        "Where do African elephants live?",
        "Show me the habitat of polar bears.",
        "Show biodiversity hotspots in Africa.",
        "What is the migration route of the Arctic tern?",
        "Ou vivent les elephants d'Afrique ?",  # French common name — Qdrant test
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Demos**")
run_parallel = st.sidebar.button(
    "Test parallel dispatch",
    help="Runs species_distribution_map + migration_analysis concurrently.",
)
run_escalation = st.sidebar.button(
    "Test NEEDS_AGENT escalation",
    help="Feeds a worker that always returns NEEDS_AGENT to prove the bubble-up.",
)


# ---------- main ----------

st.title("Biodiversity Agent Orchestrator — live demo")

st.markdown(
    "Type any question about animal distribution, habitat, hotspots or "
    "migration. The dashboard classifies the intent (via GPT-5-mini), "
    "dispatches to the LangGraph orchestrator, and shows every node "
    "that fires along the way."
)

default_prompt = "" if preset == "— custom —" else preset
prompt = st.text_area("Prompt", value=default_prompt, height=90)
run_button = st.button("Run", type="primary")


# ---------- execution ----------


def _run_prompt(prompt: str) -> None:
    if not prompt.strip():
        st.warning("Please type a prompt or pick a preset.")
        return

    col1, col2 = st.columns([1, 1])

    # Step 1 — intent detection
    with col1:
        st.subheader("1 · Intent detection")
        with st.spinner("Asking the LLM to classify the request..."):
            try:
                intent = asyncio.run(_classify_intent(prompt))
            except LLMUnavailable as exc:
                st.error(str(exc))
                return
            except Exception as exc:
                st.error(f"Intent classification failed: {exc}")
                return
        st.json(intent)

    # Step 2 — build request + run
    request = AgentRequest(
        instruction=prompt,
        context={},
        feature=intent.get("feature"),
        species_name=intent.get("species_name"),
        region=intent.get("region") or "global",
    )

    with col2:
        st.subheader("2 · Orchestrator execution trace")
        with st.spinner("Running the LangGraph state machine..."):
            t0 = time.perf_counter()
            try:
                result, trace = asyncio.run(_run_with_trace(request))
            except Exception as exc:
                st.error(f"Orchestrator failed: {exc}")
                return
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
        st.markdown(f"**Nodes visited** ({elapsed_ms} ms):")
        for i, node in enumerate(trace, 1):
            st.markdown(f"{i}. `{node}`")

    st.markdown("---")

    # Step 3 — result panels
    st.subheader("3 · Result")
    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Interactive map**")
        _render_map_url(result.map_url)
    with right:
        st.markdown("**AgentResult (structured)**")
        st.json(_dataclass_to_dict(result))


def _run_parallel_demo() -> None:
    st.subheader("Parallel dispatch — distribution + migration in one turn")
    request = AgentRequest(
        instruction="Where do African elephants live and how do they migrate?",
        context={
            "features": [
                BiodiversityFeature.SPECIES_DISTRIBUTION_MAP.value,
                BiodiversityFeature.MIGRATION_ANALYSIS.value,
            ]
        },
        species_name="Loxodonta africana",
    )
    with st.spinner("Running two workers in parallel..."):
        result, trace = asyncio.run(_run_with_trace(request))
    st.markdown("**Nodes visited**")
    st.write(trace)
    left, right = st.columns([3, 2])
    with left:
        _render_map_url(result.map_url)
    with right:
        st.json(_dataclass_to_dict(result))


def _run_escalation_demo() -> None:
    from backend.agents.biodiversity_agent.orchestrator.services.qdrant_client import (
        SpeciesTaxonomyService,
        _OfflineBackend,
    )
    from backend.agents.biodiversity_agent.workers.habitat.mock import HabitatMock
    from backend.agents.biodiversity_agent.workers.hotspots.mock import HotspotsMock
    from backend.agents.biodiversity_agent.workers.migration.mock import MigrationMock

    class _NeedsAgentMock:
        def run(self, req):
            from backend.agents.biodiversity_agent.schema import AgentResult
            return AgentResult(
                status=AgentStatus.NEEDS_AGENT,
                target_agent="Trait Discovery Agent",
                prompt_to_target_agent=(
                    f"Resolve traits driving the distribution of "
                    f"'{req.species_name}'."
                ),
                source_agents=["Species Distribution Agent"],
            )

    orch = BiodiversityOrchestrator(
        workers={
            BiodiversityFeature.SPECIES_DISTRIBUTION_MAP: _NeedsAgentMock(),
            BiodiversityFeature.HABITAT_VISUALIZATION: HabitatMock(),
            BiodiversityFeature.BIODIVERSITY_HOTSPOTS: HotspotsMock(),
            BiodiversityFeature.MIGRATION_ANALYSIS: MigrationMock(),
        },
        taxonomy=SpeciesTaxonomyService(_OfflineBackend()),
    )

    st.subheader("Escalation demo — child worker returns NEEDS_AGENT")
    request = AgentRequest(
        instruction="Where do African elephants live?",
        context={},
        feature=BiodiversityFeature.SPECIES_DISTRIBUTION_MAP.value,
        species_name="Loxodonta africana",
    )
    with st.spinner("Running..."):
        result = asyncio.run(orch.run(request))
    st.json(_dataclass_to_dict(result))
    if result.status is AgentStatus.NEEDS_AGENT:
        st.success(
            f"Correctly escalated to **{result.target_agent}** — the "
            f"Global Orchestrator would dispatch it from here."
        )
    else:
        st.error("Escalation was NOT bubbled up — something is wrong.")


# ---------- dispatch buttons ----------

if run_button:
    _run_prompt(prompt)
elif run_parallel:
    _run_parallel_demo()
elif run_escalation:
    _run_escalation_demo()
else:
    st.info(
        "Enter a prompt above and hit **Run**, or use the demo buttons "
        "in the sidebar."
    )
