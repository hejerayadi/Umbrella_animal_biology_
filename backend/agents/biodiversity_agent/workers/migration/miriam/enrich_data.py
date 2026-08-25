from data_gbif import documents
from weather import get_weather


def enrich_observation(observation):
    lat = observation.get("latitude")
    lon = observation.get("longitude")
    date = observation.get("event_date")

    if lat is None or lon is None or not date:
        return observation

    try:
        weather = get_weather(lat, lon, str(date)[:10])
    except Exception as e:
        print(f"⚠️ Erreur météo : {e}")
        weather = None

    enriched = observation.copy()
    enriched["weather"] = weather

    return enriched


def enrich_documents(documents, limit=5):
    return [
        enrich_observation(document)
        for document in documents[:limit]
    ]