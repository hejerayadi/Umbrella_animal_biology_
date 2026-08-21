"""
Récupération de données réelles via l'API GBIF (Global Biodiversity Information Facility).

Ce script correspond à la branche "For teams using APIs" du sprint :
1. Implement the retrieval pipeline -> appel à l'API GBIF
2. Normalize the retrieved data     -> transformation JSON -> texte structuré
3. Prepare it for indexing          -> même format que data.py, prêt pour ingest_documents.py

API publique, aucune clé nécessaire : https://www.gbif.org/developer/summary
"""
import requests

GBIF_API_URL = "https://api.gbif.org/v1/occurrence/search"

# Espèces migratrices qu'on veut interroger (nom scientifique = plus fiable pour l'API)
SPECIES_QUERIES = [
    {"scientificName": "Ciconia ciconia", "common_name": "cigogne blanche"},
    {"scientificName": "Megaptera novaeangliae", "common_name": "baleine à bosse"},
    {"scientificName": "Danaus plexippus", "common_name": "papillon monarque"},
]


def fetch_occurrences(scientific_name: str, limit: int = 100):
    """Récupère davantage d'observations GBIF avec coordonnées et dates."""

    params = {
        "scientificName": scientific_name,
        "limit": limit,
        "hasCoordinate": "true",
        "hasGeospatialIssue": "false",
    }

    response = requests.get(
        GBIF_API_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    return response.json().get("results", [])

def normalize_occurrence(raw_record: dict, common_name: str) -> dict:
    """Transforme une observation GBIF en données structurées."""

    country = raw_record.get("country", "pays inconnu")
    locality = raw_record.get("locality", "")
    event_date = raw_record.get("eventDate")

    lat = raw_record.get("decimalLatitude")
    lon = raw_record.get("decimalLongitude")

    # Ignorer les observations sans coordonnées
    if lat is None or lon is None:
        return None

    text = (
        f"Observation de {common_name} enregistrée en {country}"
        f"{f', près de {locality}' if locality else ''}, "
        f"le {event_date if event_date else 'date inconnue'}. "
        f"Coordonnées: {lat}, {lon}."
    )

    return {
        "text": text,
        "species": common_name,
        "region": country,
        "source": "GBIF API",

        # Données structurées pour l'analyse de migration
        "event_date": event_date,
        "latitude": float(lat),
        "longitude": float(lon),
    }


if __name__ == "__main__":
    all_documents = []

    for query in SPECIES_QUERIES:
        print(f"Récupération des données pour: {query['common_name']}...")
        raw_records = fetch_occurrences(query["scientificName"])

        for record in raw_records:
            doc = normalize_occurrence(record, query["common_name"])
            all_documents.append(doc)
            print(f"  -> {doc['text'][:80]}...")

    print(f"\n✅ {len(all_documents)} observations réelles récupérées et normalisées")

    # On sauvegarde dans un fichier Python réutilisable par ingest_documents.py
    with open("data_gbif.py", "w", encoding="utf-8") as f:
        f.write("# Données réelles récupérées depuis l'API GBIF\n")
        f.write("documents = [\n")
        for doc in all_documents:
            f.write(f"    {doc!r},\n")
        f.write("]\n")

    print("Sauvegardé dans data_gbif.py -- prêt pour l'ingestion")