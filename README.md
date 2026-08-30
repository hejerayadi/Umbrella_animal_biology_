# Umbrella Animal Biology — Scientific Analysis Agent

Base de connaissances Qdrant + pipelines d'ingestion/retrieval pour l'agent
Scientific Analysis (QA/RAG, Contradiction Detection, Gap Detection).

## 1. Installation

```bash
# Se placer dans le dossier du projet
cd umbrella-animal-biology

# Creer un environnement virtuel (recommande)
python -m venv venv
source venv/bin/activate        # Windows : venv\Scripts\activate

# Installer les dependances
pip install -r requirements.txt
```

## 2. Configuration

1. Cree un compte gratuit sur https://cloud.qdrant.io
2. Cree un cluster, recupere son URL et sa cle API
3. Copie `.env.example` en `.env` :

```bash
cp .env.example .env
```

4. Remplis `.env` avec tes vraies valeurs :

```
QDRANT_URL=https://xxxxx.qdrant.io
QDRANT_API_KEY=xxxxx
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
```

## 3. Premier test (avec les donnees d'exemple incluses)

Un petit echantillon (`data/pubmedqa_sample.json`, 3 documents) est deja
inclus pour tester immediatement, sans attendre le telechargement des
vrais corpus.

```bash
# Etape 1 : creer les collections + ingerer l'echantillon
python scripts/run_offline_ingestion.py

# Etape 2 : tester une requete de bout en bout
python scripts/run_search_agent.py "does the MSTN gene affect muscle growth in cattle"
```

Si tout fonctionne, tu dois voir les collections creees, le nombre de
points inseres, puis le contexte recupere pour ta question avec les
scores de similarite.

## 4. Lancer les tests

```bash
python -m pytest tests/test_retrieval.py -v
```

## 5. Ajouter les vrais datasets

Telecharge ces corpus et place-les dans `data/` :

- **PubMedQA** : https://github.com/pubmedqa/pubmedqa (`ori_pqal.json`)
- **SciFact** : https://github.com/allenai/scifact (`claims_train.jsonl`, `corpus.jsonl`)
- **ORKG** : export via l'API ORKG (https://orkg.org)

Puis decommente les lignes correspondantes dans
`scripts/run_offline_ingestion.py`.

## 6. Structure du projet

```
config.py                              -> configuration centrale (URL, modele, seuils)
knowledge_base/
    qdrant_client.py                   -> connexion + creation des collections
    embeddings.py                      -> generation des vecteurs (un seul endroit)
    ingestion.py                       -> chunking + upsert batch
    retrieval.py                       -> recherche + logique de fallback
pipelines/
    offline_pipeline.py                -> nettoyage avant ingestion des corpus statiques
    search_agent.py                    -> fallback online (APIs) si Qdrant insuffisant
data_sources/
    datasets_loader.py                 -> charge PubMedQA / SciFact / ORKG depuis data/
    api_clients.py                     -> PubMed, OpenAlex, Europe PMC
agents/scientific_analysis/
    qa_module.py                       -> QA / RAG
    contradiction_module.py            -> Contradiction Detection
    gap_detection_module.py            -> Gap Detection
scripts/
    run_offline_ingestion.py           -> a executer pour remplir Qdrant
    run_search_agent.py                -> a executer pour tester une question
tests/
    test_retrieval.py                  -> validation basique
```

## 7. Logique de fallback (important)

`retrieve_with_fallback()` dans `knowledge_base/retrieval.py` implemente
la cascade suivante :

1. Recherche d'abord dans Qdrant (local, rapide, gratuit).
2. Si le meilleur score est sous `SCORE_THRESHOLD` (dans `config.py`,
   0.65 par defaut) -> declenche le Search Agent (`pipelines/search_agent.py`),
   qui interroge PubMed en direct, indexe les resultats dans Qdrant, puis
   relance la recherche locale.

Ajuste `SCORE_THRESHOLD` selon tes tests : trop bas, le fallback ne se
declenche jamais ; trop haut, tu appelles les APIs externes en permanence.
