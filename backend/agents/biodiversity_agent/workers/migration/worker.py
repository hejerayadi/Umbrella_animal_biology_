"""Real Migration worker - adapts Miriam's Random Forest predictor.

Miriam's ``predict.py`` was written as a standalone Streamlit-backed
script; this adapter wraps it under the platform's
``AgentRequest -> AgentResult`` contract without changing her code.

Three real-world constraints drive the adapter's shape:

1. **Lazy import**. Miriam's module opens a QdrantClient, loads a
   90MB SentenceTransformer and unpickles the Random Forest all at
   *module load time*. Doing that during ``from ... import
   MigrationWorker`` would crash the orchestrator whenever Qdrant
   credentials are absent - which is every unit test and every CI run.
   So the import is deferred to the first ``run()`` call, and a missing
   backend degrades to ``AgentStatus.FAILED`` with a clear message.

2. **Species vocabulary**. Miriam's Random Forest was trained on
   *French common names* (``"cigogne blanche"``, ``"baleine à
   bosse"``, ``"papillon monarque"``). Our intent classifier returns
   *Latin binomials* (``Ciconia ciconia`` etc.). This adapter carries
   the translation, and refuses politely for any species the model was
   not trained on rather than returning a random misprediction.

3. **Two-step call**. The M-migration flow is ``search Qdrant for
   observed points → pick one → predict next position with the
   Random Forest``. Miriam's Streamlit app puts the second step on a
   map click; here we do both in one shot: seed the prediction with
   the first observed point so the frontend gets a complete route
   without a follow-up interaction.

The result carries an in-memory folium map (observed = leaf-green,
predicted next = amber, dashed line between the two) so the dashboard
renders the same visual language as the other three workers.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any

from ...schema import AgentRequest, AgentResult, AgentStatus


# ---------- species vocabulary ---------------------------------------
# The Random Forest was trained on these three species. The keys are the
# Latin binomials our intent classifier returns; the values are the
# French common names Miriam's model was trained on. Extending this
# table requires re-training the model with new samples, so it is kept
# in lockstep with ``migration_model.pkl``.

_LATIN_TO_FRENCH: dict[str, str] = {
    "ciconia ciconia":        "cigogne blanche",
    "megaptera novaeangliae": "baleine à bosse",
    "danaus plexippus":       "papillon monarque",
}

_TRAINED_LATIN = [
    "Ciconia ciconia (white stork)",
    "Megaptera novaeangliae (humpback whale)",
    "Danaus plexippus (monarch butterfly)",
]


class MigrationWorker:
    """Production Migration worker backed by Miriam's Random Forest."""

    SOURCE = "Migration Analysis Agent"

    def run(self, request: AgentRequest) -> AgentResult:
        species_latin = (request.species_name or "").strip()
        if not species_latin:
            return self._failed(
                "species_name is required for migration analysis"
            )

        french = _LATIN_TO_FRENCH.get(species_latin.lower())
        if french is None:
            return self._failed(
                f"The migration model was not trained on '{species_latin}'. "
                f"Trained species: {', '.join(_TRAINED_LATIN)}."
            )

        # 1. Lazy-load Miriam's predict module. A missing Qdrant URL,
        # missing sentence-transformers, or a corrupt pickle all land
        # here as a clean FAILED with the reason surfaced.
        try:
            predict_mod = self._load_predict_module()
        except Exception as exc:  # noqa: BLE001 - any import-time failure
            return self._failed(
                f"Migration backend unavailable: {type(exc).__name__}: {exc}. "
                "Ensure QDRANT_URL and QDRANT_API_KEY are set, and that "
                "sentence-transformers + qdrant-client are installed."
            )

        # 2. Search Qdrant for observed occurrences of the species.
        instruction = request.instruction or french
        try:
            _detected, observed_routes = predict_mod.process_agent_query(
                instruction
            )
        except Exception as exc:  # noqa: BLE001
            return self._failed(
                f"Qdrant search failed: {type(exc).__name__}: {exc}"
            )

        if not observed_routes:
            return self._failed(
                f"No observations found in Qdrant for '{french}'. "
                "Miriam's Qdrant collection may be empty or the query did "
                "not match any indexed observation."
            )

        # 3. Predict next position from the first observed point.
        first = observed_routes[0]
        try:
            prediction = predict_mod.predict_next_route(
                species=french,
                current_lat=first["start_lat"],
                current_lon=first["start_lon"],
                event_date=first.get("event_date"),
            )
        except Exception as exc:  # noqa: BLE001
            return self._failed(
                f"Random Forest prediction failed: {type(exc).__name__}: {exc}"
            )

        # 4. Explain the prediction with SHAP + LLM (both optional; a
        # failure here downgrades the payload but does not fail the
        # whole request - the map and coordinates are still useful).
        shap_values = self._compute_shap(
            french, first, prediction,
        )
        explanation = self._explain_with_llm(
            french=french,
            prediction=prediction,
            shap_values=shap_values,
            first_observation=first,
        )

        # 5. Render a folium map (observed green, predicted amber).
        map_url = self._render_map(french, observed_routes, prediction)

        payload = {
            "species_name":      species_latin,
            "french_name":       french,
            "migration_pattern": (
                "Random Forest prediction seeded from observed GBIF records"
            ),
            "observed_routes":   observed_routes,
            "predicted_next": {
                "lat": prediction["predicted_lat"],
                "lon": prediction["predicted_lon"],
                "month": prediction["month"],
                "day":   prediction["day"],
            },
            "shap_values":       shap_values,
            "explanation":       explanation,
            "map_url":           map_url,
        }

        route_pairs: list[tuple[float, float]] = [
            (r["start_lat"], r["start_lon"]) for r in observed_routes
        ]
        route_pairs.append(
            (prediction["predicted_lat"], prediction["predicted_lon"])
        )

        return AgentResult(
            status=AgentStatus.COMPLETED,
            output=payload,
            map_url=map_url,
            migration_route=route_pairs,
            source_agents=[self.SOURCE],
        )

    # ---------- internal ----------

    def _load_predict_module(self):
        """Import ``miriam.predict`` on first use. Cached by ``sys.modules``.

        ``predict.py`` reads ``os.environ`` and opens the pickle at
        import time; both must succeed for the import to complete. We
        do not need to change cwd because the ``MODEL_PATH`` line was
        patched in place to resolve relative to the file.
        """

        import importlib
        import sys

        name = "backend.agents.biodiversity_agent.workers.migration.miriam.predict"
        if name in sys.modules:
            return sys.modules[name]
        return importlib.import_module(name)

    def _compute_shap(
        self,
        french: str,
        first_observation: dict[str, Any],
        prediction: dict[str, Any],
    ) -> dict[str, dict[str, float]] | None:
        """Miriam's SHAP explainer: per-feature contribution to lat & lon.

        Returns ``{"latitude": {feature: shap}, "longitude": {feature: shap}}``.
        ``None`` on failure - SHAP or the model file may be missing on a
        fresh install; we do not want that to fail the whole request.
        """

        try:
            import importlib
            xai = importlib.import_module(
                "backend.agents.biodiversity_agent.workers.migration.miriam.xai"
            )
            return xai.explain_prediction(
                species=french,
                current_lat=first_observation["start_lat"],
                current_lon=first_observation["start_lon"],
                month=prediction["month"],
                day=prediction["day"],
            )
        except Exception:  # noqa: BLE001 - shap import or a bad species key
            return None

    @staticmethod
    def _explain_with_llm(
        french: str,
        prediction: dict[str, Any],
        shap_values: dict[str, dict[str, float]] | None,
        first_observation: dict[str, Any],
    ) -> str | None:
        """Ask Azure OpenAI (GPT-5-mini, already configured for intent
        classification) to turn the prediction into one paragraph of
        French that a biologist can read.

        Miriam's original pipeline used Gemini for the same job; we swap
        for our existing LLM to avoid pulling in a second SDK and a
        second API key. Returns ``None`` if the LLM is not available.

        SHAP values are ADDED to the prompt when available, so the
        explanation gets richer; when SHAP is not installed the LLM
        still gets a useful summary and produces a shorter but valid
        paragraph. The explanation is never gated on SHAP.
        """

        # Base prompt: works with or without SHAP.
        prompt_lines = [
            f"Espèce : {french}",
            f"Point de départ observé : "
            f"lat={first_observation['start_lat']:.3f}, "
            f"lon={first_observation['start_lon']:.3f}",
            f"Date : mois={prediction['month']}, jour={prediction['day']}",
            f"Position prédite : "
            f"lat={prediction['predicted_lat']:.3f}, "
            f"lon={prediction['predicted_lon']:.3f}",
        ]

        # Compute the delta - the LLM needs it to say "tire vers le nord".
        d_lat = prediction["predicted_lat"] - first_observation["start_lat"]
        d_lon = prediction["predicted_lon"] - first_observation["start_lon"]
        prompt_lines.append(
            f"Déplacement prédit : {d_lat:+.2f}° en latitude, "
            f"{d_lon:+.2f}° en longitude"
        )

        # Optional SHAP block - only when the extractor gave us numbers.
        if shap_values is not None:
            prompt_lines.append("")
            prompt_lines.append("Contributions SHAP à la latitude prédite :")
            for feature, value in shap_values["latitude"].items():
                prompt_lines.append(f"  - {feature} : {value:+.4f}")
            prompt_lines.append("")
            prompt_lines.append("Contributions SHAP à la longitude prédite :")
            for feature, value in shap_values["longitude"].items():
                prompt_lines.append(f"  - {feature} : {value:+.4f}")
            system = (
                "Tu es un biologiste. À partir des contributions SHAP "
                "d'un Random Forest, écris UN paragraphe court en français "
                "(3-4 phrases) qui explique pourquoi le modèle a prédit "
                "cette position : nomme la feature la plus influente pour "
                "la latitude puis pour la longitude, rattache ça à la "
                "biologie de l'espèce (saison, route migratoire connue). "
                "Pas de listes, pas de markdown."
            )
        else:
            system = (
                "Tu es un biologiste. Écris UN paragraphe court en "
                "français (3-4 phrases) qui explique en termes "
                "biologiques pourquoi cette espèce est prédite à cette "
                "nouvelle position à cette date : mentionne la saison, "
                "la route migratoire connue de l'espèce, et si le "
                "déplacement prédit (nord/sud, est/ouest) est cohérent. "
                "Pas de listes, pas de markdown."
            )

        try:
            from ...framework.llm_client import LLMUnavailable, get_llm
            from langchain_core.messages import HumanMessage, SystemMessage

            try:
                llm = get_llm()
            except LLMUnavailable:
                return None

            response = llm.invoke([
                SystemMessage(content=system),
                HumanMessage(content="\n".join(prompt_lines)),
            ])
            text = getattr(response, "content", str(response))
            return text.strip() if text else None
        except Exception as exc:  # noqa: BLE001 - never let the explanation crash the answer
            # Surface the reason to the operator via a print (visible in
            # the Streamlit terminal) so a failing key or a wrong
            # endpoint does not look like "the explanation is broken".
            print(f"[MigrationWorker] LLM explanation failed: "
                  f"{type(exc).__name__}: {exc}")
            return None

    def _render_map(
        self,
        french: str,
        observed: list[dict[str, Any]],
        prediction: dict[str, Any],
    ) -> str | None:
        """Folium heat + line for the dashboard.

        Return ``None`` if folium is not installed - the payload still
        carries the coordinates, so a frontend that has its own map
        renderer is unaffected.
        """

        try:
            import folium
        except ImportError:
            return None

        if not observed:
            return None

        # Center on the first observation; a compact zoom shows the
        # regional pattern without leaving the observed points offscreen.
        center = (observed[0]["start_lat"], observed[0]["start_lon"])
        fmap = folium.Map(
            location=center, zoom_start=4, tiles="CartoDB positron",
        )

        # Observed points - leaf green, hover for region + date.
        for obs in observed:
            popup_html = (
                f"<b>{obs.get('region', '-') }</b><br>"
                f"{obs.get('event_date', '-')}<br>"
                f"{obs['start_lat']:.3f}, {obs['start_lon']:.3f}"
            )
            folium.CircleMarker(
                location=(obs["start_lat"], obs["start_lon"]),
                radius=6, color="#8FD14F", fill=True, fill_color="#8FD14F",
                fill_opacity=0.85, weight=1,
                popup=folium.Popup(popup_html, max_width=220),
                tooltip=obs.get("region", "observed"),
            ).add_to(fmap)

        # Predicted next - amber marker with a star icon.
        pred_popup = (
            f"<b>Predicted next position</b><br>"
            f"{prediction['predicted_lat']:.3f}, "
            f"{prediction['predicted_lon']:.3f}<br>"
            f"month {prediction['month']}, day {prediction['day']}"
        )
        folium.Marker(
            location=(prediction["predicted_lat"], prediction["predicted_lon"]),
            icon=folium.Icon(color="orange", icon="star", prefix="fa"),
            popup=folium.Popup(pred_popup, max_width=240),
            tooltip="predicted next",
        ).add_to(fmap)

        # Dashed line from the seeding observation to the prediction.
        seed = observed[0]
        folium.PolyLine(
            locations=[
                (seed["start_lat"], seed["start_lon"]),
                (prediction["predicted_lat"], prediction["predicted_lon"]),
            ],
            color="#E0B23C", weight=3, dash_array="6,6", opacity=0.85,
        ).add_to(fmap)

        # Legend so the reader is not left guessing what the colours mean.
        legend = (
            '<div style="position:fixed;bottom:20px;left:20px;'
            'background:rgba(21,34,24,0.92);color:#E6E2D3;padding:10px 14px;'
            "border:1px solid rgba(230,226,211,0.14);border-radius:8px;"
            "font:12px \\'JetBrains Mono\\',monospace;z-index:9999;\">"
            f"<b>{french.title()}</b><br>"
            '<span style="color:#8FD14F">&#9679;</span> observed<br>'
            '<span style="color:#E0B23C">&#9733;</span> predicted next'
            "</div>"
        )
        fmap.get_root().html.add_child(folium.Element(legend))

        # Write to a temp HTML file and hand back a ``file://`` URI the
        # dashboard can read and embed.
        fd, path = tempfile.mkstemp(suffix=".html", prefix="migration_")
        os.close(fd)
        fmap.save(path)
        return Path(path).resolve().as_uri()

    @staticmethod
    def _failed(message: str) -> AgentResult:
        return AgentResult(
            status=AgentStatus.FAILED,
            output=message,
            source_agents=[MigrationWorker.SOURCE],
        )
