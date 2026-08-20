"""Conversational dashboard for M3 - Biodiversity Hotspots.

    streamlit run backend/agents/biodiversity_agent/dashboard_modules.py

A chat surface: you name any place - or an animal - in plain language, the module answers with
a sentence, a real map and ranked hotspots. Under it, every answer is produced by
the same ``HotspotsWorker`` the Biodiversity Orchestrator calls - the page is a
client of the worker, never a second code path, and never a canned demo. The
numbers on screen come from a live GBIF query.

The design follows the Agent 6 house style: dark forest ground, amber accent,
Instrument Serif titles over Space Grotesk body with JetBrains Mono for figures.

What the documents require is still all here, folded into the answer instead of a
form: the status, the plain-language summary, the ranked hotspots, the caveats in
full (doc §7.3), ``parameters_used`` (goal G5) and the raw payload the
orchestrator receives. There are no controls: every setting is the documented
default, eps and min_samples are chosen per question by the scan, and each answer
lists what it used and what each setting does. A slider reading 120 km while the
scan ran at 60 km is how someone reports parameters that were never applied.

``dashboard.py`` next door is unchanged: that one demonstrates Sprint 2
orchestration across the four workers.
"""

from __future__ import annotations

import hashlib
import html
import json
import time
import sys
from dataclasses import asdict, is_dataclass
from enum import Enum
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parents[3]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.agents.biodiversity_agent.schema import AgentRequest  # noqa: E402
from backend.agents.biodiversity_agent.workers.hotspots.followup import (  # noqa: E402
    looks_like_a_new_question,
    resolve_choice,
)
from backend.agents.biodiversity_agent.workers.hotspots.status import (  # noqa: E402
    M3Outcome,
    build_result,
    outcome_of,
)
from backend.agents.biodiversity_agent.workers.hotspots.worker import (  # noqa: E402
    HotspotsWorker,
)

st.set_page_config(page_title="Agent 6 · M3 Biodiversity Hotspots",
                   page_icon=":world_map:", layout="centered")

# ----------------------------------------------------------------- design

INK = "#0E1A12"        # page ground
PANEL = "#152218"      # bubbles and panels
RAISED = "#1C2A1E"     # cards sitting on a panel
TEXT = "#E6E2D3"       # warm off-white
MUTED = "#9FAE95"      # sage, for secondary copy and figures
AMBER = "#E0B23C"      # accent: badge, user bubble, ranks, primary button
CLAY = "#D97757"       # warnings and threatened counts
LEAF = "#8FD14F"       # good news: silhouette, match percentages
LINE = "rgba(230, 226, 211, 0.14)"

# Streamlit renders this through a markdown parser, which splits an HTML block at
# any blank line and escapes whatever follows. So the stylesheet has to be one
# contiguous run of lines with no blank lines and no /* */ comments inside it -
# otherwise the CSS shows up on the page as text. Explanations therefore live
# here rather than in the block below.
#
#   header      amber badge, Instrument Serif title, mono subtitle
#   bubbles     assistant 14/14/14/4 on panel, user amber 14/14/4/14
#   cards       one per ranked hotspot, raised surface, mono figures
#   strip       the quality numbers, mono, separation in leaf green
#   controls    pill chips, dark chat input, framed expanders
_CSS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">'
    '<link href="https://fonts.googleapis.com/css2?'
    'family=Instrument+Serif:ital@0;1'
    '&family=Space+Grotesk:wght@400;500;600'
    '&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">'
    "<style>"
    f".stApp,[data-testid='stAppViewContainer']{{background:{INK};}}"
    f"[data-testid='stHeader']{{background:{INK};}}"
    f"[data-testid='stSidebar']{{background:{PANEL};border-right:1px solid {LINE};}}"
    f".stApp,.stApp p,.stApp li,.stApp label,.stApp div"
    f"{{font-family:'Space Grotesk',sans-serif;color:{TEXT};}}"
    ".block-container{padding-top:3rem;max-width:860px;}"
    f"[data-testid='stBottom'],[data-testid='stBottom']>div,"
    f"[data-testid='stBottomBlockContainer']{{background:{INK};}}"
    ".m3-head{display:flex;align-items:center;gap:14px;margin-bottom:6px;}"
    ".m3-head>div:last-child{min-width:0;}"
    f".m3-badge{{width:44px;height:44px;border-radius:10px;background:{AMBER};"
    f"color:{INK};font:600 15px 'JetBrains Mono',monospace;display:flex;"
    "align-items:center;justify-content:center;flex:none;}"
    f".m3-title{{font-family:'Instrument Serif',serif;font-size:25px;"
    f"line-height:1.15;color:{TEXT};white-space:nowrap;overflow:hidden;"
    "text-overflow:ellipsis;}"
    f".m3-sub{{font:400 12.5px 'JetBrains Mono',monospace;color:{MUTED};"
    "margin-top:3px;letter-spacing:.02em;}"
    f".m3-rule{{height:1px;background:{LINE};margin:16px 0 20px;}}"
    ".m3-row{display:flex;gap:10px;margin:0 0 16px;align-items:flex-start;}"
    ".m3-row.user{justify-content:flex-end;}"
    f".m3-avatar{{width:28px;height:28px;border-radius:8px;background:{RAISED};"
    f"border:1px solid {LINE};color:{AMBER};flex:none;"
    "font:600 10.5px 'JetBrains Mono',monospace;display:flex;"
    "align-items:center;justify-content:center;}"
    f".m3-bubble{{background:{PANEL};border:1px solid {LINE};"
    "border-radius:14px 14px 14px 4px;padding:13px 16px;max-width:88%;"
    "font-size:15px;line-height:1.55;}"
    f".m3-bubble.user{{background:{AMBER};color:{INK};border:none;"
    "border-radius:14px 14px 4px 14px;font-weight:500;}"
    f".m3-bubble.user *{{color:{INK};}}"
    f".m3-card{{background:{RAISED};border:1px solid {LINE};border-radius:8px;"
    "padding:11px 14px;margin:8px 0;}"
    ".m3-card-head{display:flex;align-items:baseline;gap:10px;}"
    f".m3-rank{{font:400 11.5px 'JetBrains Mono',monospace;color:{AMBER};}}"
    ".m3-name{font-weight:600;font-size:15px;}"
    f".m3-metrics{{font:400 11.5px 'JetBrains Mono',monospace;color:{MUTED};"
    "margin-top:5px;display:flex;gap:18px;flex-wrap:wrap;}"
    f".m3-species{{font-size:12.5px;color:{MUTED};margin-top:6px;"
    "font-style:italic;}"
    ".m3-strip{display:flex;gap:22px;flex-wrap:wrap;margin:4px 0 2px;"
    f"font:400 11.5px 'JetBrains Mono',monospace;color:{MUTED};}}"
    f".m3-strip b{{color:{TEXT};font-weight:600;}}"
    f".m3-strip .leaf b{{color:{LEAF};}}"
    f".m3-caption{{font-size:12.5px;color:{MUTED};margin:6px 0 2px;}}"
    f".m3-steps{{background:{PANEL};border:1px solid {LINE};border-radius:12px;"
    f"padding:9px 14px;margin:0 0 14px 38px;max-width:calc(88% - 38px);}}"
    ".m3-step{display:flex;justify-content:space-between;gap:16px;padding:2px 0;"
    f"font:400 11.5px 'JetBrains Mono',monospace;color:{MUTED};}}"
    f".m3-step.live{{color:{AMBER};}}"
    ".m3-step.done{opacity:.72;}"
    f".m3-step-t{{color:{TEXT};opacity:.55;flex:none;}}"
    f".m3-caret{{color:{AMBER};animation:m3blink 1s steps(2,start) infinite;}}"
    "@keyframes m3blink{to{visibility:hidden;}}"
    ".m3-status{display:flex;align-items:center;gap:8px;margin:0 0 12px 38px;"
    f"font:400 11.5px 'JetBrains Mono',monospace;color:{AMBER};}}"
    f".m3-status.done{{color:{MUTED};}}"
    f".m3-status-t{{margin-left:auto;color:{TEXT};opacity:.45;}}"
    f".m3-dot{{color:{AMBER};animation:m3spin 1.6s linear infinite;"
    "display:inline-block;}"
    "@keyframes m3spin{to{transform:rotate(360deg);}}"
    f".stButton>button{{background:transparent;color:{MUTED};"
    "border:1px solid rgba(230,226,211,.22);border-radius:100px;"
    "font-size:12.5px;padding:5px 14px;}"
    f".stButton>button:hover{{border-color:{AMBER};color:{AMBER};}}"
    f"[data-testid='stChatInput']{{background:{PANEL};border:1px solid {LINE};"
    "border-radius:16px;}"
    f"[data-testid='stChatInput'] textarea{{color:{TEXT};}}"
    f"[data-testid='stExpander']{{border:1px solid {LINE};border-radius:10px;"
    f"background:{PANEL};}}"
    f"[data-testid='stExpander'] summary p{{font:400 12.5px 'JetBrains Mono',"
    f"monospace;color:{MUTED};}}"
    f".m3-foot{{font:400 11.5px 'JetBrains Mono',monospace;color:{MUTED};"
    "text-align:center;margin-top:22px;opacity:.75;}"
    "</style>"
)

st.markdown(_CSS, unsafe_allow_html=True)

# Same rule as the stylesheet: one contiguous block, no blank lines.
st.markdown(
    '<div class="m3-head">'
    '<div class="m3-badge">M3</div>'
    '<div><div class="m3-title">Agent 6 &mdash; Biodiversity Agent</div>'
    '<div class="m3-sub">Module M3 · Biodiversity Hotspots</div></div>'
    '</div><div class="m3-rule"></div>',
    unsafe_allow_html=True)


# ----------------------------------------------------------------- helpers


def to_plain(value: Any) -> Any:
    """Dataclasses, enums and tuples into something ``st.json`` can print."""

    if is_dataclass(value) and not isinstance(value, type):
        return {k: to_plain(v) for k, v in asdict(value).items()}
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(k): to_plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_plain(v) for v in value]
    return value


def build_stamp() -> str:
    """Six hex characters identifying the code this page is running.

    A browser tab keeps its transcript across a server restart, so an old answer
    can sit on screen looking current - which has twice sent someone hunting for
    a bug that was already fixed. This is printed in the footer: if the stamp is
    the one you expect, what you are reading came from that code.
    """

    fingerprint = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for relative in ("dashboard_modules.py",
                     "workers/hotspots/worker.py",
                     "workers/hotspots/followup.py",
                     "workers/common/geocode.py",
                     "workers/common/config.py"):
        target = here / relative
        if target.exists():
            fingerprint.update(target.read_bytes())
    return fingerprint.hexdigest()[:6]


# How many repaints one streamed bubble is allowed.
_STREAM_REPAINTS = 12


def bubble(text: str, role: str = "assistant", *, stream: bool = False) -> None:
    """One chat bubble. Text is escaped: it can carry a region the user typed.

    ``stream`` reveals the text as it is written, the way a generated answer
    normally arrives. It is used only on the turn being produced - a replayed
    turn appears at once, because re-typing an answer the reader has already seen
    wastes their time.
    """

    safe = html.escape(text).replace("\n", "<br>")
    if role == "user":
        st.markdown(
            f'<div class="m3-row user"><div class="m3-bubble user">{safe}</div></div>',
            unsafe_allow_html=True)
        return

    def wrap(body: str, caret: bool = False) -> str:
        mark = '<span class="m3-caret">&#9611;</span>' if caret else ""
        return (f'<div class="m3-row"><div class="m3-avatar">M3</div>'
                f'<div class="m3-bubble">{body}{mark}</div></div>')

    if not stream:
        st.markdown(wrap(safe), unsafe_allow_html=True)
        return

    slot = st.empty()
    words = safe.split(" ")
    # A fixed number of repaints, not a fixed number of words: each repaint costs
    # a Streamlit round trip, so a long summary would otherwise take seconds to
    # appear - measured at ~5 s for one sentence before this cap.
    step = max(3, len(words) // _STREAM_REPAINTS)
    for position in range(0, len(words), step):
        slot.markdown(wrap(" ".join(words[:position + step]), caret=True),
                      unsafe_allow_html=True)
        time.sleep(0.03)
    slot.markdown(wrap(safe), unsafe_allow_html=True)


def status_html(steps, *, running: bool) -> str:
    """The one-line progress row: what it is doing now, and for how long.

    A running answer shows its current step; a finished one shows the count. The
    full step list stays available, collapsed, so the detail is one click away
    rather than filling the transcript.
    """

    if not steps:
        return ""
    elapsed, message = steps[-1]
    dot = '<span class="m3-dot">&#10022;</span>' if running else "&#183;"
    if running:
        return (f'<div class="m3-status">{dot} <b>{html.escape(message)}</b>'
                f'<span class="m3-status-t">{elapsed:.1f}s &#183; '
                f'step {len(steps)}</span></div>')
    return (f'<div class="m3-status done">{dot} ran {len(steps)} steps'
            f'<span class="m3-status-t">{elapsed:.1f}s</span></div>')


def steps_html(steps, *, live: bool) -> str:
    """The running step list, each line stamped with the time it was reported at.

    Streamlit redraws a placeholder on every call, so appending to this list and
    re-rendering is what makes the wait legible: the user watches the pipeline
    move instead of watching a spinner. The block is kept after the run, so the
    timing of a finished answer stays inspectable.
    """

    rows = []
    for position, (elapsed, message) in enumerate(steps):
        state = "live" if (live and position == len(steps) - 1) else "done"
        mark = "&#9656;" if state == "live" else "&middot;"
        rows.append(f'<div class="m3-step {state}">'
                    f'<span>{mark} {html.escape(message)}</span>'
                    f'<span class="m3-step-t">{elapsed:.1f}s</span></div>')
    return f'<div class="m3-steps">{"".join(rows)}</div>'


def ask_for_a_region(payload: dict, key: str) -> None:
    """Turn NEEDS_CLARIFICATION into a question the user can answer in one click.

    A module that stops at "I could not resolve that" makes the user retype the
    whole sentence. The study areas are a closed list, so the follow-up is a
    choice: pick one and the analysis continues to a complete answer.
    """

    options = payload.get("options") or payload.get("known_regions") or []

    # A retry says only what is new. Repeating the quoted question and the whole
    # explanation makes the second ask longer than the first, which reads as if
    # the module had forgotten it was already asking.
    if payload.get("retry"):
        bubble(payload["retry"])
    else:
        # Quote what was asked: several turns later, "I need one study area" on
        # its own no longer says which question it belongs to.
        lines = []
        if payload.get("asked"):
            lines.append('You asked: "' + payload["asked"] + '"')
        lines.append(payload.get("message", "I could not resolve that region."))
        bubble("\n\n".join(lines))

    if not options:
        return
    st.markdown('<div class="m3-caption">Click one, type its number &mdash; or '
                'name any other place: a country, region, island or park.</div>',
                unsafe_allow_html=True)
    for row_start in range(0, len(options), 3):
        row = options[row_start:row_start + 3]
        for column, (position, option) in zip(
                st.columns(3), enumerate(row, start=row_start + 1)):
            with column:
                if st.button(f"{position}. {option.title()}",
                             key=f"ask_{key}_{option}", use_container_width=True):
                    # The answer is stored, not acted on here: this block may be
                    # rendered below the input section, so the run has to start
                    # from the top with the choice already in hand.
                    st.session_state.answer_region = option
                    st.rerun()


# What each setting does, in the order it is applied. The sidebar used to hold
# these as controls; the answer holds them as facts, which is the more useful
# arrangement: a reader needs to know what ran, and a slider does not tell them.
_SETTING_NOTES: list[tuple[str, str]] = [
    ("study area",
     "The bounding box every record was pulled from. 'design document' means one "
     "of the six regions specified in the M3 design; 'OpenStreetMap' means the "
     "place was resolved by gazetteer lookup, and the resolved name is worth "
     "checking."),
    ("taxonomic scope",
     "Which branch of life was counted. Animalia by default, so plants and fungi "
     "are outside every figure here."),
    ("years",
     "The record date range requested from GBIF. The paged path returns the "
     "newest records first, so a capped run is a recent snapshot."),
    ("cell size",
     "The grid square each coordinate was binned into. Smaller cells find "
     "tighter hotspots and more empty cells; larger ones smooth detail away."),
    ("noise floor",
     "A cell with fewer records than this was dropped before clustering, because "
     "richness measured from three records is not a measurement."),
    ("index",
     "What each cell was scored on. richness counts species; Shannon and Simpson "
     "also weigh how evenly the records are spread across them."),
    ("effort correction",
     "Whether richness was divided by observation intensity. Uncorrected "
     "richness is biodiversity multiplied by how hard anyone looked."),
    ("eps",
     "DBSCAN's neighbourhood radius in kilometres. Two cells join the same "
     "hotspot if they are within it. Larger merges neighbours; smaller "
     "fragments one hotspot into several."),
    ("min_samples",
     "How much weight must sit inside that radius for a cell to be a cluster "
     "core. Higher suppresses small hotspots and sends more cells to noise."),
    ("tuning",
     "Whether eps and min_samples were chosen by scanning 28 combinations and "
     "keeping the best silhouette, or left at the documented defaults."),
    ("record budget",
     "The ceiling on records retrieved. Above it the region is sampled rather "
     "than censused, and the sampled share is stated in the limitations."),
    ("hotspots returned",
     "How many ranked hotspots are listed. Presentation only - it does not "
     "change the clustering."),
]


def _tuning_sentence(payload, quality) -> str:
    """How eps and min_samples came to be what they are.

    Three cases, and the difference matters: a scan that chose them, a scan that
    ran and found nothing admissible, and no scan at all. Reporting the middle
    case as "tuning was off" describes work that did happen as work that did not.
    """

    if quality.tuned:
        return ("chosen by scanning 28 combinations and keeping the best "
                "silhouette.")
    if payload.parameters_used.get("auto_tune"):
        return ("kept at the documented defaults: the scan ran, but no "
                "combination produced at least two clusters inside the plausible "
                "noise band.")
    return "the documented defaults; the scan was not run."


def _slugify(name: str) -> str:
    """A filename fragment. Place names carry slashes and accents."""

    kept = "".join(character if character.isalnum() else "_"
                   for character in (name or "region").lower())
    return "_".join(part for part in kept.split("_") if part) or "region"


def settings_table(payload) -> list[dict]:
    """Every setting that shaped this answer, with its value and what it does."""

    used = payload.parameters_used
    quality = payload.quality
    species = used.get("mode") == "species"

    tuned = ("scanned: 28 combinations, best silhouette kept" if quality.tuned
             else "off: the documented defaults were used")
    effort = ("applied: " + payload.effort_correction.method
              if payload.effort_correction.applied
              else "not applied - " + (payload.effort_correction.note
                                       or "the ranking still reflects observer effort"))

    values = {
        "study area": (f"{payload.region} - {payload.study_area_km2:,.0f} km2, "
                       f"from {used.get('study_area_source', 'design document')}"),
        "taxonomic scope": (used.get("species_name") or used.get("taxon_filter")
                            or "Animalia"),
        "years": f"{used.get('year_from')} to {used.get('year_to')}",
        "cell size": ("not used - this path clusters records, not cells"
                      if species else f"{used.get('cell_size_km')} km"),
        "noise floor": ("not used on this path" if species
                        else f"{used.get('min_records_per_cell')} records per cell"),
        "index": ("record count - with one species there is no richness to rank"
                  if species else str(used.get("index", "richness"))),
        "effort correction": effort,
        "eps": f"{quality.eps_km:.0f} km",
        "min_samples": str(quality.min_samples),
        "tuning": tuned,
        "record budget": (f"{used.get('max_records'):,} records "
                          f"({payload.records_retrieved:,} retrieved, "
                          f"{payload.records_analysed:,} survived cleaning)"),
        "hotspots returned": str(used.get("top_n", len(payload.ranked_hotspots))),
    }
    return [{"setting": name, "value": values.get(name, "-"), "what it does": note}
            for name, note in _SETTING_NOTES]


def render_answer(result, key: str = "live", *, stream: bool = False) -> None:
    """Render one worker result as an assistant turn.

    ``stream`` is set for the turn being produced, so its opening sentence is
    written out rather than appearing complete.
    """

    payload = result.output

    # --- the paths that carry no analysis: ask, or explain, and stop.
    if outcome_of(result) is M3Outcome.NEEDS_CLARIFICATION:
        ask_for_a_region(payload, key)
        return

    if not hasattr(payload, "quality"):
        bubble(str(payload if payload else "I could not complete that analysis."))
        return

    quality = payload.quality
    effort = payload.effort_correction

    # --- the sentence a non-specialist reads first.
    plural = "s" if quality.n_clusters != 1 else ""
    species_mode = payload.parameters_used.get("mode") == "species"

    if species_mode:
        # A different question, so a different sentence. Calling these
        # biodiversity hotspots would be false: one species has no richness.
        named = payload.parameters_used.get("species_name", "that species")
        where = ("worldwide" if payload.region == "worldwide"
                 else f"in {payload.region.title()}")
        bubble(
            f"I looked at {payload.records_analysed:,} cleaned records of "
            f"{named} {where} and found {quality.n_clusters} area{plural} where "
            f"those records concentrate. This is observation density, not "
            f"biodiversity - a busy cluster can be one well-watched reserve.",
            stream=stream)
    elif quality.n_clusters == 1 and quality.noise_share < 0.05:
        # "found 1 area where species concentrate most" would claim a comparison
        # that did not happen: every cell is in that area.
        bubble(
            f"I looked at {payload.records_analysed:,} cleaned wildlife records "
            f"across {payload.region.title()} (about "
            f"{payload.study_area_km2:,.0f} km²) and the whole of it came out as "
            f"one area - at this scale it reads as uniformly rich, rather than "
            f"having a richest part.", stream=stream)
    else:
        bubble(
            f"I looked at {payload.records_analysed:,} cleaned wildlife records across "
            f"{payload.region.title()} (about {payload.study_area_km2:,.0f} km²) and "
            f"found {quality.n_clusters} area{plural} where species concentrate most."
            + ("" if effort.applied else
               " Effort correction was off for this run, so the ranking still reflects "
               "how hard people looked."), stream=stream)

    # One line saying what the answer *is*. Without it a reader sees "1 area" and
    # has to guess whether that is a place, a statistic or a failure. Kept to a
    # caption rather than a paragraph: the detail lives in the expanders.
    if species_mode:
        meaning = ("Each area below is a group of nearby records of this species, "
                   "ranked by how many records it holds - observation density, "
                   "not abundance.")
    else:
        cell = payload.parameters_used.get("cell_size_km", 42)
        meaning = (f"A hotspot here is a group of neighbouring {cell} km cells "
                   f"whose species count stays high after correcting for how much "
                   f"recording effort went into them. Ranked by species count.")
    st.markdown(f'<div class="m3-caption">{html.escape(meaning)}</div>',
                unsafe_allow_html=True)

    # --- the map, in the same visual frame as the bubble above it.
    path = payload.render_spec.html_path
    if path and Path(path).exists():
        st.components.v1.html(Path(path).read_text(encoding="utf-8"),
                              height=520, scrolling=False)
        st.markdown(
            '<div class="m3-caption">Scroll or drag to explore · '
            + ('warmer colour = more records of this species'
               if species_mode else
               'one dot per cell: bigger and darker = more species. Hover for '
               'its figures, click for all of them')
            + ' · the ⛶ button top-right enlarges the map to full screen.</div>',
            unsafe_allow_html=True)

    # --- ranked hotspots as cards.
    for hotspot in payload.ranked_hotspots:
        species = ", ".join(s.scientific_name for s in hotspot.top_species[:3])
        st.markdown(
            '<div class="m3-card">'
            '<div class="m3-card-head">'
            f'<span class="m3-rank">#{hotspot.rank}</span>'
            f'<span class="m3-name">{html.escape(hotspot.label)}</span>'
            '</div><div class="m3-metrics">'
            + (f'<span>{hotspot.record_count:,} records</span>'
               f'<span>{hotspot.area_km2:,.0f} km&sup2;</span>'
               f'<span>{hotspot.dominant_country or ""}</span>'
               if species_mode else
               f'<span>{hotspot.species_count:,} species</span>'
               f'<span>{hotspot.record_count:,} records</span>'
               f'<span>{hotspot.area_km2:,.0f} km&sup2;</span>'
               f'<span>{hotspot.cell_count} cells</span>')
            + '</div>'
            f'<div class="m3-species">{html.escape(species)}</div>'
            '</div>',
            unsafe_allow_html=True)

    if not payload.ranked_hotspots:
        bubble("No cluster met the density threshold here. That is a real answer "
               "rather than a failure: this region is too sparsely surveyed at "
               "these settings. A larger eps or a coarser cell would group more.")

    # --- the quality figures. Every label reads as words: "separation n/a" tells
    #     a reader nothing about why it is n/a.
    if quality.silhouette is None:
        separation = "n/a (needs 2+ areas)"
    elif quality.silhouette >= 0.7:
        separation = f"{quality.silhouette:+.3f} (well separated)"
    elif quality.silhouette >= 0.4:
        separation = f"{quality.silhouette:+.3f} (moderate)"
    else:
        separation = f"{quality.silhouette:+.3f} (weak - the areas run together)"

    counted = (f'<span>records <b>{payload.records_analysed:,}</b></span>'
               if species_mode else
               f'<span>cells <b>{payload.cells_analysed:,}</b></span>'
               f'<span>unassigned <b>{quality.noise_share:.0%}</b></span>'
               f'<span>species <b>{payload.species_analysed:,}</b></span>')
    st.markdown(
        '<div class="m3-strip">'
        f'<span class="leaf">separation <b>{separation}</b></span>'
        + counted
        + f'<span>confidence <b>{payload.confidence}</b></span>'
        '</div>',
        unsafe_allow_html=True)

    # --- how it was calculated: the documents' own requirements, on demand.
    with st.expander("How was this calculated?"):
        st.markdown(
            f"**Pipeline.** GBIF count probe → paged retrieval → five cleaning "
            f"rules → equal-area grid of "
            f"{payload.parameters_used.get('cell_size_km', 42)} km cells → "
            f"richness, Shannon, Simpson and Chao1 per cell → effort correction "
            f"→ DBSCAN over those cells, haversine metric, eps "
            f"{quality.eps_km:.0f} km → ranked hotspots.\n\n"
            f"**DBSCAN, not K-Means.** Nobody knows how many hotspots a region "
            f"has, so no *k* can be supplied; a hotspot following a river is not "
            f"convex; and K-Means would force every cell into a cluster, making "
            f"the Sahara a hotspot. DBSCAN labels sparse cells as noise, which is "
            f"why {quality.noise_share:.0%} of cells here are unassigned.\n\n"
            f"**eps = {quality.eps_km:.0f} km, min_samples = {quality.min_samples}**"
            f"{_tuning_sentence(payload, quality)} "
            f"Two runs produced identical clusters: "
            f"{'yes' if quality.reproducible else 'no'}.\n\n"
            f"**Effort correction.** {effort.method}. Uncorrected richness is not "
            f"biodiversity, it is biodiversity multiplied by how hard anyone "
            f"looked. Each cell is compared with the regional median "
            f"({effort.median_effort:,.0f} records), damped by a square root and "
            f"capped at 3×; {effort.cells_downweighted} cells were down-weighted "
            f"and {effort.cells_upweighted} lifted, changing "
            f"{effort.top5_changed} of the top five.")

        # The scan and the cleaning report used to be printed as full tables:
        # 28 rows and 6 rows of numbers a reader has to add up themselves. One
        # sentence each says the same thing, and both tables are in the download.
        if quality.scan:
            admissible = [row for row in quality.scan if row.get("admissible")]
            # The best row, not quality.silhouette: when nothing qualifies, the
            # run keeps the fallback and the reported silhouette is None - which
            # is exactly the case that used to crash this line.
            best = max((row for row in admissible
                        if row.get("silhouette") is not None),
                       key=lambda row: row["silhouette"], default=None)
            outcome = (
                f"best silhouette {best['silhouette']:+.3f} at eps "
                f"{best['eps_km']:.0f} km, min_samples {best['min_samples']}"
                if best else
                f"none qualified, so the fallback was kept: eps "
                f"{quality.eps_km:.0f} km, min_samples {quality.min_samples}")
            st.markdown(
                f"**The parameter scan** replaced GridSearchCV, which needs the "
                f"labels clustering does not have: {len(quality.scan)} "
                f"combinations tried, {len(admissible)} admissible (2+ clusters, "
                f"noise inside the plausible band), {outcome}.")

        removed = {row["rule"]: row["removed"] for row in payload.cleaning_report
                   if row.get("removed")}
        kept_share = next((row.get("share_of_input") for row in payload.cleaning_report
                           if row["rule"] == "kept"), None)
        st.markdown(
            f"**Cleaning.** {payload.records_retrieved:,} retrieved, "
            f"{payload.records_analysed:,} kept"
            + (f" ({kept_share:.0%})" if kept_share else "")
            + (": " + ", ".join(f"{count:,} {rule}"
                                for rule, count in removed.items())
               if removed else " - nothing was removed") + ".")

    with st.expander(f"Limitations to know ({len(payload.warnings)})", expanded=False):
        for warning in payload.warnings:
            st.markdown(f'<div class="m3-caption">· {html.escape(warning)}</div>',
                        unsafe_allow_html=True)

    with st.expander("Every setting this answer used, and what it does",
                     expanded=False):
        st.dataframe(pd.DataFrame(settings_table(payload)), hide_index=True,
                     use_container_width=True)

    with st.expander("Reproduce this run, or take the data"):
        st.markdown("**The settings that produced it** - everything needed to "
                    "repeat it exactly (doc goal G5):")
        st.code(json.dumps(payload.parameters_used, indent=2, default=str),
                language="json")

        # The payload used to be printed inline: several thousand lines, with
        # every cluster listed twice (the contract carries both `clusters` and
        # `ranked_hotspots`). Nobody reads that on a page - it is a file.
        plain = to_plain(payload)
        grid = plain.get("grid") or []
        as_json = json.dumps(plain, indent=2, default=str)
        st.download_button(
            f"Full payload · JSON ({len(as_json) // 1024} KB)",
            data=as_json,
            file_name=f"m3_{_slugify(payload.region)}.json",
            mime="application/json", key=f"dl_json_{key}")

        st.markdown(
            f'<div class="m3-caption">That file holds the {len(grid):,} graded '
            f'cells, all {len(payload.clusters)} clusters with their top species, '
            f'the {len(quality.scan)}-row parameter scan, the cleaning report and '
            f'the render specification the frontend draws from.</div>',
            unsafe_allow_html=True)


# ----------------------------------------------------------------- controls

# No settings panel. Every knob it held is a documented default, and the answer
# now reports what actually ran - which is what a reader needs. A slider that
# says 120 km while the scan ran at 60 km is worse than no slider at all.
with st.sidebar:
    st.markdown('<div class="m3-sub">BIODIVERSITY HOTSPOTS</div>',
                unsafe_allow_html=True)
    st.caption("Every analysis uses the parameters the design document specifies, "
               "with eps and min_samples chosen per question by an explicit scan. "
               "Each answer lists what it used, and what each setting does.")
    st.divider()
    if st.button("Clear conversation"):
        st.session_state.pop("turns", None)
        st.session_state.pop("pending", None)
        st.rerun()


# ----------------------------------------------------------------- state


if "turns" not in st.session_state:
    st.session_state.turns = []          # list[(role, payload)]
    st.session_state.greeted = False

if not st.session_state.greeted:
    st.session_state.greeted = True

# Replay the conversation so far. Results are re-rendered from the stored
# AgentResult, not recomputed - the worker is called once per question.
bubble("Hi — I'm the Biodiversity Hotspots module. Name any place — a country, "
       "region, island or park — and I'll show you where species concentrate "
       "most, with a map. Name an animal instead and I'll show where its records "
       "concentrate. Answers come from a live GBIF query: about half a minute the "
       "first time a place is asked for, a few seconds after that. Each step is "
       "shown as it runs.")

for position, (role, item) in enumerate(st.session_state.turns):
    if role == "user":
        bubble(item, "user")
    elif role == "trace":
        total = item[-1][0] if item else 0.0
        with st.expander(f"Ran {len(item)} steps · {total:.1f}s"):
            st.markdown(steps_html(item, live=False), unsafe_allow_html=True)
    else:
        render_answer(item, key=str(position))


# ----------------------------------------------------------------- input


# No fixed list of regions here. Any place a gazetteer knows is a valid study
# area, so a row of six buttons would say the opposite of what is true.
chip_clicked = None

typed = st.chat_input("Ask about any place — “hotspots in Borneo”, “biodiversity "
                      "of Costa Rica”, “hotspots for tigers”")

# "in the Congo Basin" but "in Kenya" - the definite article belongs to the
# named features, not to the countries and islands.
_TAKES_THE = {"congo basin", "amazon", "sahara"}

question = None
region = None          # set when the study area is already decided
pending = st.session_state.get("pending")

# An option clicked on the module's own follow-up question is an answer to it,
# and continues the same task.
answered = st.session_state.pop("answer_region", None)
if answered:
    chip_clicked = answered
if chip_clicked:
    article = "the " if chip_clicked in _TAKES_THE else ""
    question = (f"Where are the biodiversity hotspots in "
                f"{article}{chip_clicked.title()}?")
    region = chip_clicked
elif typed:
    question = typed
    if pending:
        # A reply while a question of ours is open is read as its answer first:
        # "2", "the second one", "congo", "SE asia" all resolve to an option.
        choice = resolve_choice(typed, pending["options"])
        if choice:
            region = choice
        elif not looks_like_a_new_question(typed, pending["options"]):
            # Not an answer, and not a different region either - so ask again
            # rather than run an analysis nobody asked for. No worker call:
            # nothing has changed except the attempt.
            st.session_state.turns.append(("user", typed))
            bubble(typed, "user")
            again = dict(pending["payload"])
            again["retry"] = ("I did not catch which one you meant. I work "
                              "through one place at a time, so pick one below "
                              "- or name another place.")
            asking = build_result(M3Outcome.NEEDS_CLARIFICATION, output=again,
                                  source_agents=["Biodiversity Hotspots Agent"])
            st.session_state.turns.append(("assistant", asking))
            render_answer(asking, key=str(len(st.session_state.turns) - 1))
            question = None

if question:
    st.session_state.turns.append(("user", question))
    bubble(question, "user")

    # The worker is the whole module: it resolves the region from the question
    # itself, so nothing is pre-decided here.
    request = AgentRequest(
        instruction=question,
        # Only set when a click or a follow-up answer already decided it; a typed
        # question is left for the worker to read, as before.
        region=region,
        # No settings to pass: the worker applies the documented defaults and
        # tunes eps and min_samples itself, per question.
        context={})

    trace = []
    slot = st.empty()
    began = time.time()

    def report(message: str) -> None:
        trace.append((time.time() - began, message))
        slot.markdown(status_html(trace, running=True), unsafe_allow_html=True)

    report("reading the question")
    result = HotspotsWorker().run(request, progress=report)
    trace.append((time.time() - began, "done"))
    slot.markdown(status_html(trace, running=False), unsafe_allow_html=True)

    # Kept in the transcript, so the timing of an answer stays visible once the
    # next question is asked.
    st.session_state.turns.append(("trace", trace))

    # A question of ours stays open until it is answered, so the next typed reply
    # can be read against its options.
    if outcome_of(result) is M3Outcome.NEEDS_CLARIFICATION:
        asked_payload = result.output
        asked_payload.setdefault("asked", question)
        st.session_state.pending = {
            "options": (asked_payload.get("options")
                        or asked_payload.get("known_regions") or []),
            "payload": asked_payload,
        }
    else:
        st.session_state.pop("pending", None)

    st.session_state.turns.append(("assistant", result))

    # The same key the replay will give this turn, so an option button clicked
    # here still exists on the next pass - a widget whose key changes between
    # passes loses the click.
    # The detail collapses to a single line, the way a finished action reads in a
    # transcript; the live status row above it is no longer needed.
    slot.empty()
    with st.expander(f"Ran {len(trace)} steps · {trace[-1][0]:.1f}s"):
        st.markdown(steps_html(trace, live=False), unsafe_allow_html=True)

    render_answer(result, key=str(len(st.session_state.turns) - 1), stream=True)

st.markdown(
    '<div class="m3-foot">Live GBIF Occurrence Search · effort-corrected '
    'richness · DBSCAN (haversine) · any place a gazetteer knows · build '
    f'{build_stamp()}</div>',
    unsafe_allow_html=True)
