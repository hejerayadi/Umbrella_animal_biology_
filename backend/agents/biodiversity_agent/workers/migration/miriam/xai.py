
import os
import pickle
import shap
import numpy as np

# Patched by MigrationWorker adapter:
#  - MODEL_PATH resolves relative to this file (was cwd-dependent).
#  - The gemini_api import is moved into the ``__main__`` block below
#    so importing this module never requires the Gemini SDK/key. The
#    orchestrator uses Azure OpenAI for natural-language explanation
#    instead; xai stays pure SHAP.

_MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "migration_model.pkl",
)

# Charger le modèle
with open(_MODEL_PATH, "rb") as f:
    saved_model = pickle.load(f)

model = saved_model["model"]
species_mapping = saved_model["species_mapping"]


FEATURE_NAMES = [
    "espèce",
    "latitude",
    "longitude",
    "mois",
    "jour"
]


def explain_prediction(
    species,
    current_lat,
    current_lon,
    month,
    day
):
    """Explique la prédiction du Random Forest avec SHAP."""

    species_value = species_mapping[
        species.lower().strip()
    ]

    features = np.array([[
        species_value,
        float(current_lat),
        float(current_lon),
        month,
        day
    ]])

    explanations = {}

    # Random Forest qui prédit la latitude
    latitude_model = model.estimators_[0]
    latitude_explainer = shap.TreeExplainer(
        latitude_model
    )
    latitude_shap = latitude_explainer.shap_values(
        features
    )[0]

    # Random Forest qui prédit la longitude
    longitude_model = model.estimators_[1]
    longitude_explainer = shap.TreeExplainer(
        longitude_model
    )
    longitude_shap = longitude_explainer.shap_values(
        features
    )[0]

    explanations["latitude"] = dict(
        zip(FEATURE_NAMES, latitude_shap)
    )

    explanations["longitude"] = dict(
        zip(FEATURE_NAMES, longitude_shap)
    )

    return explanations


if __name__ == "__main__":

    result = explain_prediction(
        species="cigogne blanche",
        current_lat=56.374599,
        current_lon=10.850209,
        month=8,
        day=21
    )

    print("\n=== XAI - LATITUDE ===")

    for feature, value in result["latitude"].items():
        print(
            f"{feature} : {value:.4f}"
        )

    print("\n=== XAI - LONGITUDE ===")

    for feature, value in result["longitude"].items():
        print(
            f"{feature} : {value:.4f}"
        )

    xai_text = f"""
Explication XAI de la prédiction du modèle Random Forest.

Influence sur la latitude :
{result["latitude"]}

Influence sur la longitude :
{result["longitude"]}
"""

    question = (
        "Pourquoi le modèle Random Forest a-t-il "
        "prévu cette position ?"
    )

    # Import moved inside __main__: keeps xai.py importable without
    # gemini_api.py (which is not shipped in the adapter package).
    try:
        from gemini_api import explain_with_gemini  # type: ignore
        gemini_response = explain_with_gemini(question, xai_text)
        print("\n=== EXPLICATION GEMINI ===")
        print(gemini_response)
    except ImportError:
        print("\n(gemini_api not installed - SHAP output above is the whole answer.)")