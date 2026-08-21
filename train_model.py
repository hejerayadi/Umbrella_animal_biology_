import pickle
from collections import defaultdict

from data_gbif import documents

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_absolute_error


# ============================================================
# 1. CONFIGURATION
# ============================================================

species_mapping = {
    "cigogne blanche": 0,
    "baleine à bosse": 1,
    "papillon monarque": 2
}


# ============================================================
# 2. REGROUPEMENT DES OBSERVATIONS PAR ESPÈCE
# ============================================================

observations = defaultdict(list)

for doc in documents:

    species = doc.get("species", "").lower().strip()

    lat = doc.get("latitude")
    lon = doc.get("longitude")
    event_date = doc.get("event_date")

    if lat is None or lon is None:
        continue

    if species not in species_mapping:
        continue

    try:
        date_string = str(event_date)

        year = int(date_string[:4])
        month = int(date_string[5:7])
        day = int(date_string[8:10])

    except Exception:
        continue

    observations[species].append({
        "latitude": float(lat),
        "longitude": float(lon),
        "year": year,
        "month": month,
        "day": day
    })


# ============================================================
# 3. TRI CHRONOLOGIQUE
# ============================================================

for species in observations:

    observations[species].sort(
        key=lambda x: (
            x["year"],
            x["month"],
            x["day"]
        )
    )


# ============================================================
# 4. CRÉATION DES DONNÉES D'ENTRAÎNEMENT
# ============================================================

X = []
y = []

for species, obs_list in observations.items():

    # Il faut au minimum 2 observations
    # pour créer une relation actuelle -> suivante.

    if len(obs_list) < 2:
        continue

    for i in range(len(obs_list) - 1):

        current = obs_list[i]
        next_obs = obs_list[i + 1]

        # ----------------------------------------------------
        # FEATURES
        # ----------------------------------------------------
        #
        # 1. espèce
        # 2. latitude actuelle
        # 3. longitude actuelle
        # 4. mois actuel
        # 5. jour actuel
        #

        X.append([
            species_mapping[species],
            current["latitude"],
            current["longitude"],
            current["month"],
            current["day"]
        ])

        # ----------------------------------------------------
        # CIBLE
        # ----------------------------------------------------
        #
        # La prochaine position réelle
        #

        y.append([
            next_obs["latitude"],
            next_obs["longitude"]
        ])


# ============================================================
# 5. VÉRIFICATION
# ============================================================

print(
    f"Nombre d'observations utilisables : {len(X)}"
)

if len(X) < 20:

    raise ValueError(
        "Pas assez de données pour entraîner le modèle."
    )


# ============================================================
# 6. SÉPARATION TRAIN / TEST
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


# ============================================================
# 7. CRÉATION DU MODÈLE
# ============================================================

model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
)


# ============================================================
# 8. ENTRAÎNEMENT
# ============================================================

print("Entraînement du modèle...")

model.fit(
    X_train,
    y_train
)

print("✅ Modèle entraîné")


# ============================================================
# 9. ÉVALUATION
# ============================================================

predictions = model.predict(X_test)

error = mean_absolute_error(
    y_test,
    predictions
)

print(
    f"Erreur moyenne : {error:.4f}"
)


# ============================================================
# 10. SAUVEGARDE
# ============================================================

model_path = "migration_model.pkl"

with open(model_path, "wb") as file:

    pickle.dump(
        {
            "model": model,
            "species_mapping": species_mapping
        },
        file
    )

print(
    f"✅ Modèle sauvegardé dans : {model_path}"
)