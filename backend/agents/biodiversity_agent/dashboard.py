"""Streamlit test harness for the Biodiversity Agent Orchestrator.

Run it from the repo root:

    streamlit run backend/agents/biodiversity_agent/dashboard.py

The page is a manual replacement for the Global Scientific Orchestrator:
it takes a free-form user prompt, uses the configured LLM to detect
the intent (which of the four biodiversity features to run), then
dispatches an ``AgentRequest`` to the real ``BiodiversityOrchestrator``
and displays the resulting map.

Everything else stays untouched — the orchestrator, workers and services
are called exactly as they will be in production. The dashboard is a
test client, not a new codepath.
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
    page_title="Biodiversity Orchestrator",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# One orchestrator per Streamlit session so we do not rebuild the
# LangGraph state machine on every keystroke.
if "orchestrator" not in st.session_state:
    st.session_state.orchestrator = BiodiversityOrchestrator()


# ---------- page-wide CSS (typography, spacing, dark-mode compatible) ----------

_PAGE_CSS = """
<style>
  /* tighten default streamlit padding on the main content */
  .block-container { padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1400px; }

  /* headings — a bit tighter and warmer */
  h1 { font-size: 2rem !important; font-weight: 700 !important;
       letter-spacing: -0.5px; margin-bottom: 0.2rem !important; }
  h1 + p { color: #7A8B7A; font-size: 0.95rem; margin-top: 0 !important; }

  /* run button — bigger, forest green, obvious */
  .stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #2C5F2D 0%, #4A7A4B 100%);
    color: white; border: none;
    padding: 0.6rem 2.5rem; font-size: 1rem; font-weight: 600;
    border-radius: 10px; letter-spacing: 0.3px;
    box-shadow: 0 4px 12px rgba(44, 95, 45, 0.30);
    transition: transform 0.12s ease-out, box-shadow 0.12s ease-out;
  }
  .stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 18px rgba(44, 95, 45, 0.40);
  }

  /* prompt textarea — slightly larger typography */
  .stTextArea textarea { font-size: 1rem !important; line-height: 1.4 !important; }

  /* horizontal separators — subtle */
  hr { margin: 1.6rem 0 !important; border-color: rgba(151, 188, 98, 0.22) !important; }

  /* sidebar polish */
  section[data-testid="stSidebar"] { border-right: 1px solid rgba(151, 188, 98, 0.15); }
  section[data-testid="stSidebar"] h1 { font-size: 1.15rem !important; }
</style>
"""
st.markdown(_PAGE_CSS, unsafe_allow_html=True)


# ---------- intent detection ----------


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


# ---------- service badge rendering (routing visualization) ----------

# Each service gets a distinctive sticker glyph — a recognizable emoji
# rather than a boring dot — so the routing decision is legible at a
# glance in the demo video.
_SERVICES: list[dict] = [
    {
        "feature": BiodiversityFeature.SPECIES_DISTRIBUTION_MAP.value,
        "label":   "Species Distribution",
        "sticker": "📍",
        "caption": "Where does it live",
    },
    {
        "feature": BiodiversityFeature.HABITAT_VISUALIZATION.value,
        "label":   "Habitat",
        "sticker": "🌳",
        "caption": "Habitat regions",
    },
    {
        "feature": BiodiversityFeature.BIODIVERSITY_HOTSPOTS.value,
        "label":   "Hotspots",
        "sticker": "🔥",
        "caption": "Density heatmap",
    },
    {
        "feature": BiodiversityFeature.MIGRATION_ANALYSIS.value,
        "label":   "Migration",
        "sticker": "🐦",
        "caption": "Seasonal routes",
    },
]


def _badge_html(service: dict, selected: bool) -> str:
    """Compact horizontal badge: sticker on the left, label on the right.
    Designed to stack vertically in a narrow side column next to the
    full-width map. Selected = green gradient; not selected = muted."""

    label   = service["label"]
    sticker = service["sticker"]

    if selected:
        return f"""
        <div style="
            background: linear-gradient(135deg, #2C5F2D 0%, #4A7A4B 100%);
            color: white;
            padding: 12px 14px;
            border-radius: 12px;
            border: 2px solid #97BC62;
            box-shadow: 0 6px 16px rgba(44, 95, 45, 0.35);
            display: flex; align-items: center; gap: 12px;
            transform: scale(1.02);
            transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
        ">
            <div style="font-size: 26px; line-height: 1; flex-shrink: 0;
                        filter: drop-shadow(0 2px 3px rgba(0,0,0,0.25));">
                {sticker}
            </div>
            <div style="flex: 1; min-width: 0;">
                <div style="font-size: 14px; font-weight: 700;
                            letter-spacing: 0.2px; line-height: 1.2;">
                    {label}
                </div>
                <div style="font-size: 8.5px; font-weight: 700;
                            letter-spacing: 2.5px; color: #C8E6C9;
                            margin-top: 4px;">
                    SELECTED
                </div>
            </div>
        </div>
        """
    return f"""
    <div style="
        background: rgba(120, 140, 120, 0.08);
        color: #7A8B7A;
        padding: 12px 14px;
        border-radius: 12px;
        border: 1px solid rgba(120, 140, 120, 0.20);
        display: flex; align-items: center; gap: 12px;
        opacity: 0.68;
        transition: all 0.35s cubic-bezier(0.16, 1, 0.3, 1);
    ">
        <div style="font-size: 24px; line-height: 1; flex-shrink: 0;
                    opacity: 0.55; filter: grayscale(0.7);">
            {sticker}
        </div>
        <div style="flex: 1; min-width: 0;">
            <div style="font-size: 13px; font-weight: 600; line-height: 1.2;">
                {label}
            </div>
        </div>
    </div>
    """


def _render_service_badges(container, highlighted: list[str] | None = None) -> None:
    """Render the four service badges stacked vertically inside
    ``container``. Designed for the narrow right-hand column."""

    highlighted = highlighted or []
    with container.container():
        for i, service in enumerate(_SERVICES):
            st.markdown(
                _badge_html(service, service["feature"] in highlighted),
                unsafe_allow_html=True,
            )
            if i < len(_SERVICES) - 1:
                st.markdown("<div style='height:8px'></div>",
                            unsafe_allow_html=True)


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


def _render_map_url(map_url: str | None) -> None:
    """Embed a folium map inline. Handles the ``file://`` URI parsing
    quirks between Windows (``file:///C:/...``) and Linux
    (``file:///path``), and falls back to a link + diagnostic if the
    file cannot be located."""

    if not map_url:
        st.info("No map to display for this feature.")
        return

    if map_url.startswith("file://"):
        from urllib.parse import urlparse, unquote

        parsed = urlparse(map_url.split("#", 1)[0])
        raw = unquote(parsed.path)
        if len(raw) > 3 and raw[0] == "/" and raw[2] == ":":
            raw = raw[1:]
        local_path = Path(raw)
        if local_path.exists():
            html = local_path.read_text(encoding="utf-8")
            st.components.v1.html(html, height=620, scrolling=False)
            return
        st.warning(
            f"Map file not found on disk: `{local_path}`.\n\n"
            "This usually means the renderer failed silently — check that "
            "`folium` is installed (`pip install folium`) and re-run the "
            "prompt."
        )
    st.markdown(f"[Open map in a new tab]({map_url})")


# ---------- sidebar ----------

st.sidebar.markdown("### 🌿 Biodiversity Orchestrator")
st.sidebar.caption("Sprint 2 · Group E · Umbrella BioHub")

st.sidebar.markdown("---")
st.sidebar.markdown("**Try a preset**")
preset = st.sidebar.selectbox(
    "preset",
    label_visibility="collapsed",
    options=[
        "— custom —",
        "Where do African elephants live?",
        "Show me the habitat of polar bears.",
        "Show biodiversity hotspots in Africa.",
        "What is the migration route of the Arctic tern?",
        "Ou vivent les elephants d'Afrique ?",
    ],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Advanced demos**")
run_parallel = st.sidebar.button(
    "⚡ Parallel dispatch",
    help="Runs species_distribution + migration concurrently (asyncio.gather).",
    use_container_width=True,
)
run_escalation = st.sidebar.button(
    "↗️ NEEDS_AGENT escalation",
    help="Feeds a worker that always returns NEEDS_AGENT to prove bubble-up.",
    use_container_width=True,
)


# ---------- main ----------

st.title("Biodiversity Orchestrator")
st.markdown(
    "Type any question about animal distribution, habitat, hotspots or "
    "migration — the orchestrator routes it to the right worker."
)

default_prompt = "" if preset == "— custom —" else preset
prompt_col, button_col = st.columns([5, 1])
with prompt_col:
    prompt = st.text_area(
        "prompt",
        value=default_prompt,
        height=100,
        placeholder="e.g. Where do African elephants live?",
        label_visibility="collapsed",
    )
with button_col:
    st.markdown("<div style='height: 22px'></div>", unsafe_allow_html=True)
    run_button = st.button("Run  ▶", type="primary", use_container_width=True)

st.markdown("")  # small gap

# Two-column layout: the map fills the left (wide) column, the four
# service badges are stacked vertically on the right so the routing
# decision stays visible without hogging horizontal space.
map_col, badges_col = st.columns([4, 1], gap="medium")

with badges_col:
    st.markdown(
        "<div style='font-size:11px;font-weight:600;color:#7A8B7A;"
        "letter-spacing:2px;text-transform:uppercase;margin-bottom:10px;'>"
        "Services</div>",
        unsafe_allow_html=True,
    )
    _badges_placeholder = st.empty()
    _render_service_badges(_badges_placeholder, highlighted=[])

with map_col:
    _map_placeholder = st.empty()
    with _map_placeholder.container():
        st.markdown(
            "<div style='padding: 60px 30px; text-align: center;"
            "background: rgba(120, 140, 120, 0.05); border-radius: 14px;"
            "border: 1px dashed rgba(120, 140, 120, 0.25);'>"
            "<div style='font-size: 48px; opacity: 0.35;'>🗺️</div>"
            "<div style='color: #7A8B7A; margin-top: 12px; font-size: 14px;'>"
            "Type a prompt above and hit <b>Run</b> to see the map here."
            "</div></div>",
            unsafe_allow_html=True,
        )


# ---------- execution ----------


def _run_prompt(prompt: str) -> None:
    """Video-friendly minimal flow: light up the picked service badge in
    the side column, then embed the map in the main column. No JSON
    panels, no trace list — the badge lighting up + map appearing tells
    the whole routing story."""

    if not prompt.strip():
        st.warning("Please type a prompt or pick a preset.")
        return

    with st.spinner("Classifying the prompt..."):
        try:
            intent = asyncio.run(_classify_intent(prompt))
        except LLMUnavailable as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error(f"Intent classification failed: {exc}")
            return

    picked_feature = intent.get("feature")
    if picked_feature:
        time.sleep(0.5)
        _render_service_badges(_badges_placeholder, highlighted=[picked_feature])

    request = AgentRequest(
        instruction=prompt,
        context={},
        feature=picked_feature,
        species_name=intent.get("species_name"),
        region=intent.get("region") or "global",
    )
    with st.spinner("Running the orchestrator..."):
        try:
            result = asyncio.run(st.session_state.orchestrator.run(request))
        except Exception as exc:
            with _map_placeholder.container():
                st.error(f"Orchestrator failed: {exc}")
            return

    with _map_placeholder.container():
        _render_map_url(result.map_url)


def _run_parallel_demo() -> None:
    parallel_features = [
        BiodiversityFeature.SPECIES_DISTRIBUTION_MAP.value,
        BiodiversityFeature.MIGRATION_ANALYSIS.value,
    ]
    time.sleep(0.4)
    _render_service_badges(_badges_placeholder, highlighted=parallel_features)

    request = AgentRequest(
        instruction="Where do African elephants live and how do they migrate?",
        context={"features": parallel_features},
        species_name="Loxodonta africana",
    )
    with st.spinner("Running two workers in parallel..."):
        result = asyncio.run(st.session_state.orchestrator.run(request))

    with _map_placeholder.container():
        _render_map_url(result.map_url)


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

    time.sleep(0.4)
    _render_service_badges(
        _badges_placeholder,
        highlighted=[BiodiversityFeature.SPECIES_DISTRIBUTION_MAP.value],
    )
    request = AgentRequest(
        instruction="Where do African elephants live?",
        context={},
        feature=BiodiversityFeature.SPECIES_DISTRIBUTION_MAP.value,
        species_name="Loxodonta africana",
    )
    with st.spinner("Running..."):
        result = asyncio.run(orch.run(request))

    with _map_placeholder.container():
        if result.status is AgentStatus.NEEDS_AGENT:
            st.success(
                f"Escalated to **{result.target_agent}** — the Global "
                f"Orchestrator would dispatch it from here. No map for this run."
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
