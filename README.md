# 🌿 BioMigrate AI

BioMigrate AI est une application d'analyse de la migration animale basée sur l'intelligence artificielle.

L'application combine :

* 🦅 des observations de biodiversité provenant de **GBIF** ;
* 🔎 **Qdrant** pour rechercher les observations pertinentes ;
* 🧠 **Sentence Transformers** pour la recherche sémantique ;
* 🤖 un modèle **Random Forest** pour prédire la prochaine position ;
* 🔍 **SHAP** pour expliquer les prédictions du modèle ;
* 💬 **Gemini** pour générer des explications en langage naturel ;
* 🌍 **Folium** pour visualiser les observations et les routes sur une carte interactive ;
* 🎨 **Streamlit** pour l'interface utilisateur.

---

## 📁 Structure du projet

```text
BioMigrate-AI/
│
├── app.py
├── predict.py
├── gemini_api.py
├── xai.py
├── enrich.py
│
├── migration_model.pkl
├── requirements.txt
├── .env
│
└── README.md
```

### Rôle des principaux fichiers

| Fichier               | Rôle                                                             |
| --------------------- | ---------------------------------------------------------------- |
| `app.py`              | Interface Streamlit, chatbot et carte interactive                |
| `predict.py`          | Recherche Qdrant et prédiction avec Random Forest                |
| `gemini_api.py`       | Communication avec Gemini                                        |
| `xai.py`              | Explication des prédictions avec SHAP                            |
| `enrich.py`           | Enrichissement des observations avec les données supplémentaires |
| `migration_model.pkl` | Modèle Machine Learning entraîné                                 |
| `.env`                | Clés API et configuration                                        |
| `requirements.txt`    | Bibliothèques Python nécessaires                                 |

---

# 1. Prérequis

Avant de commencer, installer :

* Python 3.10 ou version compatible ;
* Git ;
* un compte Google AI / une clé API Gemini ;
* un compte Qdrant Cloud si Qdrant Cloud est utilisé.

---

# 2. Télécharger le projet

Cloner le projet :

```bash
git clone URL_DU_REPOSITORY
```

Puis entrer dans le dossier :

```bash
cd BioMigrate-AI
```

---

# 3. Créer l'environnement virtuel

Sous Windows :

```bash
python -m venv venv
```

Activer l'environnement :

```bash
venv\Scripts\activate
```

Lorsque l'environnement est actif, le terminal doit afficher :

```text
(venv)
```

---

# 4. Installer les bibliothèques

Installer toutes les dépendances du projet :

```bash
pip install -r requirements.txt
```

Si SHAP n'est pas présent dans `requirements.txt` :

```bash
pip install shap
```

Puis mettre à jour le fichier :

```bash
pip freeze > requirements.txt
```

---

# 5. Configurer les variables d'environnement

Créer un fichier nommé :

```text
.env
```

à la racine du projet.

Exemple :

```text
MigrationAgent=VOTRE_CLE_GEMINI

QDRANT_URL=VOTRE_URL_QDRANT

QDRANT_API_KEY=VOTRE_CLE_QDRANT
```

⚠️ Ne jamais publier les clés API sur GitHub.

Ajouter `.env` dans `.gitignore` :

```text
.env
venv/
__pycache__/
```

---

# 6. Vérifier le modèle Machine Learning

Le fichier suivant doit être présent à la racine du projet :

```text
migration_model.pkl
```

Le modèle contient :

```text
MultiOutputRegressor
│
├── RandomForestRegressor → latitude
│
└── RandomForestRegressor → longitude
```

Le modèle utilise cinq variables :

```text
1. espèce
2. latitude
3. longitude
4. mois
5. jour
```

Il produit une nouvelle position composée de :

```text
latitude prédite
longitude prédite
```

---

# 7. Fonctionnement de la recherche

Lorsqu'un utilisateur pose une question dans l'application :

```text
Question utilisateur
        ↓
Sentence Transformer
        ↓
Embedding de la question
        ↓
Recherche dans Qdrant
        ↓
Observations GBIF pertinentes
        ↓
Affichage sur la carte
```

Les observations récupérées contiennent notamment :

* espèce ;
* latitude ;
* longitude ;
* région ;
* date d'observation.

---

# 8. Prédiction de migration

L'utilisateur peut cliquer sur une position affichée sur la carte.

L'application récupère :

```text
Espèce
Latitude
Longitude
Date
```

Puis construit les cinq variables nécessaires au modèle :

```text
[espèce, latitude, longitude, mois, jour]
```

Le Random Forest prédit ensuite :

```text
Latitude suivante
Longitude suivante
```

La route prédite est affichée en orange sur la carte.

La route observée est affichée en vert.

---

# 9. XAI avec SHAP

Le fichier `xai.py` permet d'analyser l'influence des variables utilisées par le Random Forest.

La chaîne XAI est :

```text
Random Forest
      ↓
Prédiction
      ↓
SHAP
      ↓
Importance des variables
```

Les variables analysées sont :

```text
espèce
latitude
longitude
mois
jour
```

Pour tester le XAI indépendamment de l'application :

```bash
python xai.py
```

Le terminal affiche les contributions des variables pour :

```text
Latitude prédite
Longitude prédite
```

---

# 10. Explication avec Gemini

Gemini ne réalise pas la prédiction.

La prédiction est réalisée par :

```text
Random Forest
```

SHAP explique ensuite l'influence des variables :

```text
Random Forest
      ↓
Prédiction
      ↓
SHAP
      ↓
Résultats XAI
      ↓
Gemini
      ↓
Explication en français
```

Gemini transforme les résultats techniques de SHAP en une explication compréhensible.

Il ne doit pas inventer une nouvelle prédiction.

---

# 11. Lancer l'application

Une fois l'environnement activé :

```bash
streamlit run app.py
```

Streamlit ouvre normalement l'application dans le navigateur.

Si elle ne s'ouvre pas automatiquement, utiliser l'adresse affichée dans le terminal, généralement :

```text
http://localhost:8501
```

---

# 12. Utilisation de l'application

### Étape 1 — Poser une question

Exemple :

```text
Où migre la cigogne blanche ?
```

L'application recherche les observations pertinentes.

---

### Étape 2 — Observer la carte

Les observations sont affichées sur la carte.

Les trajectoires observées sont représentées en vert.

---

### Étape 3 — Cliquer sur une position

Cliquer sur un point de départ.

Le modèle Random Forest calcule la prochaine position.

La route prédite apparaît en orange.

---

### Étape 4 — Poser une question à Gemini

Exemples :

```text
Pourquoi cette route a-t-elle été prédite ?
```

```text
Comment la cigogne blanche a-t-elle migré selon les observations ?
```

```text
Quelles variables ont influencé la prédiction ?
```

Gemini peut alors utiliser les données disponibles et, lorsque le module XAI est utilisé, les résultats SHAP.

---

# 13. Tester séparément les composants

Il est recommandé de tester les composants progressivement.

### Tester le modèle ML

```bash
python predict.py
```

### Tester le XAI

```bash
python xai.py
```

### Tester l'application complète

```bash
streamlit run app.py
```

---

# 14. Architecture générale

L'architecture globale de BioMigrate AI est :

```text
                    ┌─────────────────┐
                    │ Question        │
                    │ utilisateur     │
                    └────────┬────────┘
                             │
                             ↓
                    ┌─────────────────┐
                    │ Sentence        │
                    │ Transformer     │
                    └────────┬────────┘
                             │
                             ↓
                    ┌─────────────────┐
                    │ Qdrant          │
                    │ Observations    │
                    └────────┬────────┘
                             │
                             ↓
                    ┌─────────────────┐
                    │ GBIF            │
                    │ Observations    │
                    └────────┬────────┘
                             │
                             ↓
                    ┌─────────────────┐
                    │ Random Forest   │
                    │ Prédiction      │
                    └────────┬────────┘
                             │
                  ┌──────────┴──────────┐
                  ↓                     ↓
          ┌──────────────┐      ┌──────────────┐
          │ SHAP / XAI   │      │ Carte        │
          │ Explication  │      │ Folium       │
          └──────┬───────┘      └──────────────┘
                 │
                 ↓
          ┌──────────────┐
          │ Gemini       │
          │ Explication  │
          └──────────────┘
```

---

# 15. Résumé du rôle de chaque technologie

| Technologie           | Fonction                                   |
| --------------------- | ------------------------------------------ |
| GBIF                  | Fournit les observations de biodiversité   |
| Qdrant                | Stocke et recherche les observations       |
| Sentence Transformers | Transforme les questions en vecteurs       |
| Random Forest         | Prédit la prochaine position               |
| SHAP                  | Explique la prédiction du modèle           |
| Gemini                | Formule une explication en langage naturel |
| Folium                | Affiche les routes sur la carte            |
| Streamlit             | Fournit l'interface de l'application       |

---

# 16. Ordre recommandé pour tester le projet

```text
1. Activer venv
        ↓
2. Installer requirements.txt
        ↓
3. Configurer .env
        ↓
4. Vérifier migration_model.pkl
        ↓
5. Tester predict.py
        ↓
6. Tester xai.py
        ↓
7. Lancer app.py
```

Commandes principales :

```bash
venv\Scripts\activate
pip install -r requirements.txt
python predict.py
python xai.py
streamlit run app.py
```

---

# 17. Sécurité

Ne jamais publier :

```text
.env
QDRANT_API_KEY
MigrationAgent
autres clés API
```

Le fichier `.env` doit rester local.

---

# 👩‍💻 BioMigrate AI

Projet d'analyse et de prédiction de la migration animale combinant **données de biodiversité, recherche sémantique, Machine Learning, XAI et LLM**.

```
```
