import requests


# ============================================================
# CONFIGURATION
# ============================================================

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"


# ============================================================
# RÉCUPÉRATION DES DONNÉES MÉTÉOROLOGIQUES
# ============================================================

def get_weather(latitude, longitude, event_date):
    """
    Récupère les conditions météorologiques historiques
    correspondant à une observation GBIF.

    Paramètres :
        latitude : latitude de l'observation
        longitude : longitude de l'observation
        event_date : date de l'observation au format YYYY-MM-DD

    Retourne :
        dictionnaire contenant les données météo
    """

    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": event_date,
        "end_date": event_date,

        "daily": [
            "temperature_2m_mean",
            "temperature_2m_min",
            "temperature_2m_max",
            "precipitation_sum",
            "wind_speed_10m_max"
        ],

        "timezone": "auto"
    }

    response = requests.get(
        OPEN_METEO_URL,
        params=params,
        timeout=30
    )

    response.raise_for_status()

    data = response.json()

    daily = data.get("daily")

    if not daily:
        return None

    return {
        "date": daily["time"][0],

        "temperature_mean": daily[
            "temperature_2m_mean"
        ][0],

        "temperature_min": daily[
            "temperature_2m_min"
        ][0],

        "temperature_max": daily[
            "temperature_2m_max"
        ][0],

        "precipitation": daily[
            "precipitation_sum"
        ][0],

        "wind_speed_max": daily[
            "wind_speed_10m_max"
        ][0]
    }


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TEST OPEN-METEO")
    print("=" * 60)

    latitude = 45.15242
    longitude = 38.58291
    event_date = "2026-01-02"

    try:

        weather = get_weather(
            latitude,
            longitude,
            event_date
        )

        print()
        print("📍 Position :", latitude, longitude)
        print("📅 Date :", event_date)
        print()

        print("🌦️ Données météorologiques :")

        for key, value in weather.items():
            print(f"   {key} : {value}")

    except Exception as e:

        print()
        print("❌ Erreur :", e)