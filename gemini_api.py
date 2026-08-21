import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

API_KEY = os.getenv("MigrationAgent")

if not API_KEY:
    raise ValueError("❌ Clé Gemini introuvable")

client = genai.Client(api_key=API_KEY)


def answer_migration_question(
    question,
    observations,
    predicted_route=None
):
    if not question:
        return "❌ Aucune question."

    if not observations:
        return "❌ Aucune observation disponible."

    data = ""

    for obs in observations:
        weather = obs.get("weather") or {}

        data += f"""
Espèce : {obs.get("species")}
Région : {obs.get("region")}
Date : {obs.get("event_date")}
Latitude : {obs.get("latitude")}
Longitude : {obs.get("longitude")}

Météo :
Température moyenne : {weather.get("temperature_mean")} °C
Température minimale : {weather.get("temperature_min")} °C
Température maximale : {weather.get("temperature_max")} °C
Précipitations : {weather.get("precipitation")} mm
Vent maximal : {weather.get("wind_speed_max")} km/h
-------------------------
"""

    route = predicted_route or "Aucune route prédite."

    prompt = f"""
Tu es l'agent scientifique de BioMigrate AI.

Réponds à la question uniquement à partir des données
fournies ci-dessous.

QUESTION :
{question}

OBSERVATIONS GBIF ENRICHIES :
{data}

ROUTE PRÉDITE PAR LE MODÈLE ML :
{route}

RÈGLES :
- Ne fais aucune prédiction.
- Ne crée aucune nouvelle route.
- Ne crée aucune coordonnée.
- Explique uniquement les données disponibles.
- Distingue les faits des interprétations.
- Réponds en français.
- Sois court, clair et scientifique.

Réponds directement à la question.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text

def explain_with_gemini(question, xai_text):
    prompt = f"""
Tu es un assistant scientifique de BioMigrate AI.

Le modèle Random Forest a produit une prédiction.
Une analyse XAI avec SHAP a calculé l'influence
des différentes variables sur cette prédiction.

Question :
{question}

Résultats XAI :
{xai_text}

Explique simplement pourquoi le modèle a produit
cette prédiction.

IMPORTANT :
- Utilise uniquement les résultats XAI fournis.
- N'invente aucune donnée.
- Ne modifie pas la prédiction.
- Explique quelles variables ont le plus influencé
  la prédiction.
- Distingue l'influence du modèle d'une causalité
  biologique réelle.
- Réponds en français.
- Sois court et facile à comprendre.
"""

    response = client.models.generate_content(
        model="gemini-3.6-flash",
        contents=prompt
    )

    return response.text

