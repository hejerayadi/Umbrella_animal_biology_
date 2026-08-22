import os
import re
import pickle
from datetime import datetime

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "animal_migration"


# ============================================================
# CONNEXION À QDRANT
# ============================================================

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)


# ============================================================
# MODÈLE D'EMBEDDING
# ============================================================

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ============================================================
# CHARGEMENT DU MODÈLE MACHINE LEARNING
# ============================================================

# Patched by MigrationWorker adapter: resolve the model path relative
# to this file so imports work regardless of the caller's cwd. The rest
# of Miriam's file is unchanged from her upstream branch.
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "migration_model.pkl")

try:
    with open(MODEL_PATH, "rb") as f:
        saved_model = pickle.load(f)

    migration_model = saved_model["model"]
    species_mapping = saved_model["species_mapping"]

    print("[migration] modele charge :", MODEL_PATH)

except FileNotFoundError:
    migration_model = None
    species_mapping = {}

    print("[migration] fichier migration_model.pkl introuvable")


# ============================================================
# RECHERCHE DES OBSERVATIONS DANS QDRANT
# ============================================================

def process_agent_query(user_query: str, species_filter: str = None):

    """
    Recherche les observations pertinentes dans Qdrant.

    ``species_filter`` est le nom commun français tel qu'il est stocké
    dans le payload (ex: "cigogne blanche"). Sans lui, la recherche
    vectorielle pioche dans toute la collection et peut répondre à une
    question sur une espèce avec les observations d'une autre : la
    similarité d'embedding seule ne sépare pas les espèces.
    """

    query_vector = embedding_model.encode(
        user_query
    ).tolist()

    query_filter = None

    if species_filter:

        query_filter = Filter(
            must=[
                FieldCondition(
                    key="species",
                    match=MatchValue(value=species_filter)
                )
            ]
        )

    results = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=10,
        query_filter=query_filter
    )

    routes = []
    detected_species = "Animal Inconnu"

    for point in results.points:

        payload = point.payload

        text = payload.get("text", "")

        species = payload.get(
            "species",
            "animal"
        )

        if detected_species == "Animal Inconnu":
            detected_species = species

        # ----------------------------------------------------
        # Récupération des coordonnées
        # ----------------------------------------------------

        lat = payload.get("latitude")
        lon = payload.get("longitude")

        # Si les coordonnées ne sont pas dans le payload,
        # on les cherche dans le texte.
        if lat is None or lon is None:

            match = re.search(
                r"Coordonnées:\s*(-?\d+\.\d+),\s*(-?\d+\.\d+)",
                text
            )

            if match:
                lat = float(match.group(1))
                lon = float(match.group(2))

        if lat is not None and lon is not None:

            routes.append({
                "start_lat": float(lat),
                "start_lon": float(lon),

                # ------------------------------------------------
                # Pour l'affichage de la carte dans app.py
                # La destination réelle sera calculée par
                # predict_next_route() après le clic.
                # ------------------------------------------------
                "end_lat": float(lat),
                "end_lon": float(lon),

                "region": payload.get(
                    "region",
                    "Inconnue"
                ),

                "species": species,

                "event_date": payload.get(
                    "event_date",
                    ""
                )
            })

    return detected_species, routes


# ============================================================
# PRÉDICTION DE LA PROCHAINE POSITION
# ============================================================

def predict_next_route(
    species: str,
    current_lat: float,
    current_lon: float,
    event_date=None
):

    """
    Utilise le modèle Machine Learning.

    Le modèle a été entraîné avec 5 variables :

    1. espèce
    2. latitude
    3. longitude
    4. mois
    5. jour
    """

    # --------------------------------------------------------
    # Vérification du modèle
    # --------------------------------------------------------

    if migration_model is None:

        raise RuntimeError(
            "Le modèle migration_model.pkl "
            "n'est pas disponible."
        )

    # --------------------------------------------------------
    # Identification de l'espèce
    # --------------------------------------------------------

    species_lower = species.lower().strip()

    if species_lower not in species_mapping:

        raise ValueError(
            f"Espèce inconnue : {species}. "
            f"Espèces disponibles : "
            f"{list(species_mapping.keys())}"
        )

    species_value = species_mapping[
        species_lower
    ]

    # --------------------------------------------------------
    # Conversion de la date
    # --------------------------------------------------------

    if event_date:

        try:

            date_string = str(event_date)

            month = int(
                date_string[5:7]
            )

            day = int(
                date_string[8:10]
            )

        except Exception:

            month = 1
            day = 1

    else:

        now = datetime.now()

        month = now.month
        day = now.day

    # --------------------------------------------------------
    # LES 5 FEATURES
    # --------------------------------------------------------

    features = [[
        species_value,
        float(current_lat),
        float(current_lon),
        month,
        day
    ]]

    # Patched by MigrationWorker adapter: the debug print
    # ``print("Features envoyées au modèle :", features)`` used to fire
    # on every prediction, polluting the Streamlit terminal on each
    # click. Silenced here - the same information is on the ``features``
    # local variable if a caller wants to log it explicitly.

    # --------------------------------------------------------
    # PRÉDICTION
    # --------------------------------------------------------

    prediction = migration_model.predict(
        features
    )

    predicted_lat = float(
        prediction[0][0]
    )

    predicted_lon = float(
        prediction[0][1]
    )

    # --------------------------------------------------------
    # RÉSULTAT
    # --------------------------------------------------------

    return {
        "species": species,

        "start_lat": float(
            current_lat
        ),

        "start_lon": float(
            current_lon
        ),

        "predicted_lat": predicted_lat,

        "predicted_lon": predicted_lon,

        "month": month,

        "day": day
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print()
    print("=" * 60)
    print("TEST DU MODÈLE DE MIGRATION")
    print("=" * 60)

    tests = [

        {
            "species": "cigogne blanche",
            "lat": 45.15242,
            "lon": 38.58291,
            "date": "2026-01-02"
        },

        {
            "species": "baleine à bosse",
            "lat": 30.0,
            "lon": -20.0,
            "date": "2026-01-02"
        },

        {
            "species": "papillon monarque",
            "lat": 40.0,
            "lon": -100.0,
            "date": "2026-01-02"
        }

    ]

    for test in tests:

        print()
        print("-" * 60)

        resultat = predict_next_route(

            species=test["species"],

            current_lat=test["lat"],

            current_lon=test["lon"],

            event_date=test["date"]

        )

        print(
            "🐾 Espèce :",
            resultat["species"]
        )

        print(
            "📍 Position actuelle :",
            resultat["start_lat"],
            resultat["start_lon"]
        )

        print(
            "🔮 Position prédite :",
            resultat["predicted_lat"],
            resultat["predicted_lon"]
        )

    print()
    print("[migration] tous les tests sont termines")