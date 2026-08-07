# Umbrella — Protein Structure Visualization Agent

> **Document d’implémentation contraignant pour un agent de code**
> Ce fichier définit le périmètre, les contrats, le workflow, les stockages, les tests et l’ordre d’exécution.
> L’agent de code doit implémenter uniquement ce qui est décrit ici et ne doit ajouter aucun framework, service, agent ou fonctionnalité non demandé.

---

## Amendements de sprint (2026-08-06)

Ces décisions remplacent le texte d’origine là où elles s’y opposent :

1. **Pas de `docker-compose.yml`.** Qdrant est un cluster managé déjà déployé ; l’URL et la clé viennent de `QDRANT_URL` / `QDRANT_API_KEY`. Le fichier compose a été supprimé, ainsi que les commandes `docker compose up`.
2. **PostgreSQL est hors périmètre de ce sprint.** `PERSISTENCE_ENABLED=false` par défaut : le service répond sans base relationnelle. Les modèles SQLAlchemy, les repositories et les migrations Alembic restent en place pour le sprint suivant. Les points de la §12 et de la §22 qui exigent PostgreSQL sont reportés.
3. **Enveloppe API uniforme.** Toutes les réponses `/api/v1` sont `{data, meta, error}` (§16 et `docs/api-contract.md`). `error` reste compatible RFC 9457 et ajoute un `code` stable.
4. **Logs structurés.** Le format JSON de la §19.1 est implémenté dans `app/observability/` : contexte propagé par `contextvars`, middleware de corrélation, helper `log_stage` pour `status` / `duration_ms` / `error_code`.
5. **Routes alignées sur la §16.1** : `/api/v1/protein-structure-analyses`, `/api/v1/knowledge/protein/ingestions`, `/api/v1/ready`.

---

## Sommaire

1. [Objectif](#1-objectif)
2. [Périmètre exact du sprint](#2-périmètre-exact-du-sprint)
3. [Hors périmètre](#3-hors-périmètre)
4. [Décisions techniques imposées](#4-décisions-techniques-imposées)
5. [Architecture cible](#5-architecture-cible)
6. [Sources biologiques](#6-sources-biologiques)
7. [Contrat avec le Grand Orchestrator](#7-contrat-avec-le-grand-orchestrator)
8. [Contrats de données internes](#8-contrats-de-données-internes)
9. [Sous-orchestrateur Protein et agents spécialisés](#9-sous-orchestrateur-protein-et-agents-spécialisés)
10. [Workflow LangGraph](#10-workflow-langgraph)
11. [Règles de décision](#11-règles-de-décision)
12. [Persistance PostgreSQL](#12-persistance-postgresql)
13. [Base vectorielle Qdrant](#13-base-vectorielle-qdrant)
14. [Pipeline d’ingestion](#14-pipeline-dingestion)
15. [Pipeline de retrieval](#15-pipeline-de-retrieval)
16. [API REST `/api/v1`](#16-api-rest-apiv1)
17. [Structure GitHub attendue](#17-structure-github-attendue)
18. [Configuration](#18-configuration)
19. [Observabilité et gestion des erreurs](#19-observabilité-et-gestion-des-erreurs)
20. [Tests obligatoires](#20-tests-obligatoires)
21. [Ordre d’implémentation](#21-ordre-dimplémentation)
22. [Definition of Done](#22-definition-of-done)
23. [Interdictions explicites](#23-interdictions-explicites)

---

## 1. Objectif

Implémenter le **Protein Structure Visualization Sub-Orchestrator** de la plateforme multi-agent Umbrella.

Le composant doit recevoir une tâche structurée provenant du **Grand Orchestrator**, exécuter un workflow LangGraph réel, interroger les sources biologiques réelles, sélectionner une structure protéique, récupérer du contexte dans Qdrant, enregistrer l’exécution dans PostgreSQL et retourner une réponse traçable.

Chaîne cible :

```text
Grand Orchestrator ou runner local
    → Protein Sub-Orchestrator
    → LangGraph
    → agents spécialisés et outils
    → UniProt / RCSB PDB / AlphaFold / InterPro / PDBe-SIFTS
    → Qdrant
    → sélection déterministe
    → configuration Mol*
    → explication fondée sur preuves
    → validation scientifique
    → résultat au Grand Orchestrator
```

Le LLM ne constitue jamais une source biologique. Les faits doivent provenir des APIs ou des documents indexés.

---

## 2. Périmètre exact du sprint

Le sprint doit livrer :

- un repository GitHub organisé ;
- un backend Python ;
- un endpoint recevant une tâche du Grand Orchestrator ;
- un `ProteinSubOrchestrator` réel avec LangGraph ;
- des contrats Pydantic à la frontière HTTP ;
- des dataclasses immuables dans le domaine ;
- des clients asynchrones pour les APIs biologiques ;
- des fournisseurs biologiques exclusivement réels en runtime ;
- un mode `real` activable par configuration ;
- PostgreSQL comme source de vérité ;
- Qdrant avec la collection `protein_knowledge` ;
- BGE-M3 pour les embeddings ;
- un pipeline d’ingestion ;
- un pipeline de retrieval sémantique avec filtres ;
- une sélection PDB-first avec fallback AlphaFold ;
- une configuration Mol* JSON, sans frontend ;
- une explication générée uniquement à partir d’un `EvidencePack` ;
- un Scientific Critic retournant `ACCEPT`, `REVISE` ou `ABSTAIN` ;
- des tests unitaires, d’intégration et end-to-end.

---

## 3. Hors périmètre

Ne pas implémenter dans ce sprint :

- le vrai Grand Orchestrator Umbrella ;
- le frontend Next.js ;
- l’intégration visuelle réelle de Mol* ;
- la reconstruction ADN ;
- la génération d’images d’espèces ;
- un moteur de prédiction AlphaFold local ;
- un système de publication scientifique ;
- une base graphe ;
- des agents Genome, Literature, Evolution ou Trait Discovery complets ;
- une authentification utilisateur complète ;
- Kubernetes ;
- Kafka ;
- Celery ;
- une architecture microservices ;
- Redis ou MinIO/S3 dans le MVP initial ;
- un système de billing ou multi-tenant.

Les références Ensembl doivent venir du Genome Agent dans l’architecture finale. Les publications doivent venir du Literature Agent. Le Protein Agent ne doit pas absorber ces responsabilités.

---

## 4. Décisions techniques imposées

| Élément | Choix obligatoire |
|---|---|
| Langage | Python 3.12 |
| API | FastAPI |
| Validation HTTP | Pydantic 2 |
| Modèles internes | `dataclasses` immuables avec `frozen=True` et `slots=True` |
| Orchestration | LangGraph |
| Intégration LLM | LangChain uniquement pour structured output et explication |
| LLM | Azure AI Foundry, configurable |
| Embeddings | BGE-M3 |
| Base relationnelle | PostgreSQL |
| ORM | SQLAlchemy 2 |
| Migrations | Alembic |
| Base vectorielle | Qdrant |
| Client HTTP | `httpx.AsyncClient` |
| Retries | Tenacity |
| Tests | pytest, pytest-asyncio, respx |
| Dépendances Python | `requirements.txt` |
| API versioning | toutes les routes sous `/api/v1` |

### Règle de gestion du projet Python

Utiliser :

```text
requirements.txt
requirements-dev.txt
```

Ne pas créer :

```text
pyproject.toml
uv.lock
Poetry
Pipenv
```

---

## 5. Architecture cible

```mermaid
flowchart TD
    CALLER[Grand Orchestrator / runner local] --> API[FastAPI /api/v1]
    API --> PSO[Protein Sub-Orchestrator]
    PSO --> GRAPH[LangGraph StateGraph]

    GRAPH --> ID[Protein Identity Capability]
    GRAPH --> EXP[Experimental Structure Capability]
    GRAPH --> PRED[Predicted Structure Capability]
    GRAPH --> ANN[Annotation Capability]
    GRAPH --> MAP[Residue Mapping Capability]
    GRAPH --> RET[Knowledge Retrieval Capability]
    GRAPH --> SEL[Deterministic Structure Selector]
    GRAPH --> SCENE[Mol* Scene Builder]
    GRAPH --> EXPLAIN[Evidence-Bound Explanation]
    GRAPH --> CRITIC[Scientific Critic]

    ID --> UNIPROT[UniProt API]
    EXP --> PDB[RCSB PDB API]
    PRED --> AF[AlphaFold DB API]
    ANN --> INTERPRO[InterPro API]
    MAP --> SIFTS[PDBe / SIFTS API]
    RET --> QDRANT[(Qdrant)]

    GRAPH --> POSTGRES[(PostgreSQL)]
    CRITIC --> API
    API --> CALLER
```

### Principe architectural

Les APIs externes sont des **tools déterministes**.
Les agents spécialisés représentent des **capacités biologiques**, pas un wrapper différent pour chaque endpoint HTTP.

---

## 6. Sources biologiques

| Source | Rôle | Accès | Priorité |
|---|---|---|---|
| UniProtKB | identité canonique, séquence, nom, organisme, annotations | REST API | obligatoire |
| RCSB PDB | structures expérimentales, méthode, résolution, chaînes | Search API + Data API + mmCIF | obligatoire |
| AlphaFold DB | structure prédite, pLDDT, PAE | REST API + fichiers | fallback obligatoire |
| InterPro | domaines, familles, sites et régions fonctionnelles | REST API | obligatoire pour annotations |
| PDBe / SIFTS | mapping UniProt vers chaîne/résidu PDB | REST API | obligatoire si résidu, mutation ou domaine ciblé |
| Qdrant | contexte sémantique interne | client Python | obligatoire dans le sprint |
| PostgreSQL | état, historique et décisions | SQLAlchemy | obligatoire |

### Sources indirectes

- Ensembl : fourni par le Genome Agent.
- PubMed / Europe PMC : fourni par le Literature Agent.
- Le Protein Agent peut retourner un statut `NEEDS_AGENT`, mais ne doit pas implémenter ces agents.

---

## 7. Contrat avec le Grand Orchestrator

### 7.1 Entrée orchestrée

Le Grand Orchestrator ou le runner local envoie une tâche structurée :

```json
{
  "task_id": "f0b9737f-a7f6-4aac-8af1-f7d196d16918",
  "trace_id": "7cce77eb-ec91-4353-9f29-2af0c55519b7",
  "source_agent": "umbrella-main-orchestrator",
  "target_capability": "protein-structure.orchestrate",
  "schema_version": "1.0",
  "idempotency_key": "protein-task-001",
  "input": {
    "resolved_gene_id": "TP53",
    "species": {
      "scientific_name": "Homo sapiens",
      "taxon_id": 9606
    },
    "uniprot_accession": "P04637",
    "protein_sequence": null,
    "requested_regions": [],
    "residue_position": null,
    "mutation": null,
    "preferred_source": "AUTO",
    "include_explanation": true
  }
}
```

### 7.2 Sortie attendue

```json
{
  "task_id": "f0b9737f-a7f6-4aac-8af1-f7d196d16918",
  "status": "COMPLETED",
  "validation_status": "ACCEPT",
  "protein": {
    "uniprot_accession": "P04637",
    "gene_symbol": "TP53",
    "scientific_name": "Homo sapiens",
    "taxon_id": 9606
  },
  "selected_structure": {
    "source": "RCSB_PDB",
    "external_id": "1TUP",
    "structure_type": "EXPERIMENTAL",
    "file_format": "MMCIF",
    "file_url": "https://files.rcsb.org/download/1TUP.cif"
  },
  "alternative_structures": [],
  "annotations": [],
  "residue_mappings": [],
  "molstar_config": {},
  "explanation": {
    "summary": "Explication générée uniquement depuis les preuves validées.",
    "limitations": []
  },
  "warnings": [],
  "evidence": []
}
```

### 7.3 Statuts autorisés

```text
RECEIVED
VALIDATING
RUNNING
NEEDS_CLARIFICATION
NEEDS_AGENT
PARTIAL
COMPLETED
NO_STRUCTURE_FOUND
FAILED
```

### 7.4 Validation scientifique

```text
ACCEPT
REVISE
ABSTAIN
```

- `ACCEPT` : identité, structure et preuves suffisantes.
- `REVISE` : résultat récupérable, mais mapping ou preuve manque.
- `ABSTAIN` : identité ambiguë, mapping critique impossible ou preuves insuffisantes.

---

## 8. Contrats de données internes

Créer des modèles Pydantic dans `app/contracts/` et des dataclasses dans `app/domain/models/`.

### 8.1 Modèles minimaux

```python
from dataclasses import dataclass, field
from typing import Literal
from uuid import UUID


@dataclass(frozen=True, slots=True)
class SpeciesRef:
    scientific_name: str
    taxon_id: int


@dataclass(frozen=True, slots=True)
class ProteinStructureRequest:
    task_id: UUID
    trace_id: UUID
    resolved_gene_id: str
    species: SpeciesRef
    uniprot_accession: str | None = None
    protein_sequence: str | None = None
    requested_regions: tuple[str, ...] = ()
    residue_position: int | None = None
    mutation: str | None = None
    preferred_source: Literal["AUTO", "PDB", "ALPHAFOLD"] = "AUTO"
    include_explanation: bool = True


@dataclass(frozen=True, slots=True)
class EvidenceRef:
    provider: str
    external_id: str
    retrieved_at: str
    source_url: str | None = None
    checksum: str | None = None


@dataclass(frozen=True, slots=True)
class StructureCandidate:
    source: Literal["RCSB_PDB", "ALPHAFOLD_DB"]
    external_id: str
    structure_type: Literal["EXPERIMENTAL", "PREDICTED"]
    chain_id: str | None
    experimental_method: str | None
    resolution_angstrom: float | None
    sequence_coverage: float
    mean_plddt: float | None
    file_format: Literal["MMCIF", "PDB"]
    file_url: str
    selection_score: float
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ResidueMapping:
    uniprot_accession: str
    uniprot_position: int
    pdb_id: str
    chain_id: str
    pdb_residue_number: str | None
    is_observed: bool
    mapping_source: Literal["SIFTS"] = "SIFTS"
```

### 8.2 État LangGraph

Créer un `TypedDict` ou un modèle Pydantic nommé `ProteinWorkflowState` avec au minimum :

```text
task
current_status
resolved_protein
uniprot_data
pdb_candidates
alphafold_candidate
annotations
residue_mappings
retrieved_documents
selected_structure
alternative_structures
evidence_pack
molstar_config
explanation
validation_status
warnings
errors
executed_nodes
retry_counts
```

---

## 9. Sous-orchestrateur Protein et agents spécialisés

### 9.1 ProteinSubOrchestrator

Responsabilités :

- construire le graphe ;
- initialiser l’état ;
- déclencher les nœuds ;
- appliquer les routes conditionnelles ;
- persister chaque étape ;
- agréger les résultats ;
- retourner un contrat stable.

### 9.2 Capacités internes

| Capacité | Mission |
|---|---|
| Input Validation | valider contrat, espèce et identifiant |
| Protein Identity | confirmer l’accession UniProt et la séquence |
| Experimental Structure Search | rechercher et normaliser les candidats RCSB PDB |
| Predicted Structure Search | récupérer AlphaFold si nécessaire |
| Annotation Retrieval | récupérer les domaines InterPro |
| Residue Mapping | mapper UniProt vers PDB avec SIFTS |
| Knowledge Retrieval | interroger Qdrant avec BGE-M3 |
| Structure Selection | classer les candidats avec règles déterministes |
| Mol* Scene Builder | produire une configuration JSON sûre |
| Evidence Explanation | produire une explication depuis l’EvidencePack |
| Scientific Critic | `ACCEPT`, `REVISE` ou `ABSTAIN` |

### 9.3 Interdiction

Ne pas créer un agent LLM séparé pour chaque API.
Les clients UniProt, PDB, AlphaFold, InterPro et SIFTS sont des outils sous les capacités ci-dessus.

---

## 10. Workflow LangGraph

```mermaid
flowchart TD
    START([START]) --> RECEIVE[receive_task]
    RECEIVE --> VALIDATE[validate_input]

    VALIDATE --> VALID{input valid?}
    VALID -->|no| CLARIFY[return_needs_clarification]
    VALID -->|yes| IDENTITY[resolve_protein_identity]

    IDENTITY --> ID_OK{identity confirmed?}
    ID_OK -->|no| ABSTAIN[scientific_abstain]
    ID_OK -->|yes| PARALLEL[parallel_retrieval]

    PARALLEL --> PDB[search_experimental_structures]
    PARALLEL --> INTERPRO[fetch_annotations]
    PARALLEL --> QDRANT[retrieve_knowledge]

    PDB --> EVALUATE_PDB[evaluate_pdb_results]
    EVALUATE_PDB --> PDB_OK{valid PDB candidate?}

    PDB_OK -->|yes| SELECT[select_structure]
    PDB_OK -->|no| AF[search_alphafold]
    AF --> SELECT

    INTERPRO --> JOIN[join_results]
    QDRANT --> JOIN
    SELECT --> JOIN

    JOIN --> NEED_MAPPING{residue/domain mapping required?}
    NEED_MAPPING -->|yes| SIFTS[map_residues_with_sifts]
    NEED_MAPPING -->|no| EVIDENCE[build_evidence_pack]
    SIFTS --> EVIDENCE

    EVIDENCE --> SCENE[build_molstar_scene]
    SCENE --> EXPLAIN{include explanation?}
    EXPLAIN -->|yes| LLM[generate_grounded_explanation]
    EXPLAIN -->|no| CRITIC[run_scientific_critic]
    LLM --> CRITIC

    CRITIC --> VERDICT{critic verdict}
    VERDICT -->|ACCEPT| COMPLETE[complete]
    VERDICT -->|REVISE| PARTIAL[return_partial]
    VERDICT -->|ABSTAIN| ABSTAIN

    CLARIFY --> END([END])
    COMPLETE --> END
    PARTIAL --> END
    ABSTAIN --> END
```

### 10.1 Où se manifeste l’agenticité

- choix du prochain nœud selon l’état ;
- exécution parallèle de PDB, InterPro et Qdrant ;
- fallback AlphaFold ;
- décision de demander un mapping SIFTS ;
- suspension avec `NEEDS_CLARIFICATION` ;
- critique scientifique ;
- arrêt avec `ABSTAIN` au lieu d’inventer ;
- retries limités sur erreurs temporaires.

Le workflow ne doit pas être une chaîne fixe.

---

## 11. Règles de décision

### 11.1 Identité

- Une accession UniProt valide et compatible avec l’espèce est l’ancre canonique.
- Si l’identité est ambiguë, retourner `NEEDS_CLARIFICATION` ou `ABSTAIN`.
- Ne jamais sélectionner une structure avant confirmation de l’identité protéique.

### 11.2 Sélection de structure

Ordre par défaut :

```text
PDB expérimental valide
    → PDB expérimental partiel mais couvrant la zone demandée
    → AlphaFold
    → NO_STRUCTURE_FOUND
```

PDB est sélectionné uniquement si :

- l’identité correspond ;
- la chaîne correspond à la protéine ;
- la couverture est suffisante ;
- la zone demandée est observable lorsque nécessaire ;
- les métadonnées sont disponibles.

AlphaFold doit être marqué explicitement `PREDICTED`.

### 11.3 Mapping de résidus

- Ne jamais supposer que la position UniProt est égale au numéro de résidu PDB.
- Utiliser PDBe/SIFTS.
- Si un résidu demandé ne peut pas être mappé, ne pas le surligner.
- Retourner un warning et laisser le Scientific Critic choisir `REVISE` ou `ABSTAIN`.

### 11.4 LLM

Le LLM peut :

- produire une explication ;
- reformuler les limitations ;
- générer un structured output validé.

Le LLM ne peut pas :

- choisir librement une structure en ignorant les scores ;
- inventer une fonction biologique ;
- déclarer une structure AlphaFold expérimentale ;
- créer une preuve absente ;
- inférer une causalité médicale.

---

## 12. Persistance PostgreSQL

PostgreSQL est la source de vérité.

### 12.1 Tables minimales

| Table | Contenu |
|---|---|
| `orchestrator_tasks` | tâche reçue, statut, trace, idempotency key |
| `agent_runs` | exécution de chaque capacité |
| `tool_calls` | appels API, durée, statut HTTP, erreurs |
| `protein_entities` | accession UniProt, gène, espèce, taxonomie |
| `structure_candidates` | tous les candidats PDB et AlphaFold |
| `structure_selections` | structure sélectionnée et raison |
| `protein_annotations` | domaines et sites InterPro |
| `residue_mappings` | mapping UniProt vers PDB |
| `retrieval_runs` | requête Qdrant, filtres, scores |
| `visualization_configs` | configuration Mol* |
| `evidence_refs` | provenance des preuves |
| `ingestion_documents` | état et hash des documents indexés |

### 12.2 Données interdites dans PostgreSQL

Ne pas stocker :

- les embeddings BGE-M3 ;
- les fichiers mmCIF complets ;
- les fichiers PDB complets ;
- les secrets ;
- les prompts contenant des clés ;
- des réponses API brutes illimitées.

Les réponses brutes peuvent être conservées en JSONB seulement si elles sont bornées, nettoyées et nécessaires à l’audit.

---

## 13. Base vectorielle Qdrant

### 13.1 Collection

```text
protein_knowledge
```

### 13.2 Contenu

Qdrant stocke :

- texte normalisé ;
- embedding BGE-M3 ;
- métadonnées ;
- référence de provenance.

Qdrant ne stocke pas :

- tâches ;
- utilisateurs ;
- statuts LangGraph ;
- fichiers mmCIF/PDB ;
- décisions métier ;
- secrets.

### 13.3 Payload obligatoire

```json
{
  "document_id": "uniprot-P04637-function-001",
  "text": "Normalized biological text",
  "domain": "protein",
  "source": "UniProt",
  "source_record_id": "P04637",
  "protein_id": "P04637",
  "gene_symbol": "TP53",
  "species_name": "Homo sapiens",
  "taxonomy_id": 9606,
  "document_type": "protein_function",
  "section": "function",
  "language": "en",
  "content_sha256": "sha256",
  "indexed_at": "ISO-8601"
}
```

### 13.4 Types de documents autorisés

```text
protein_function
protein_domain
experimental_structure_description
predicted_structure_description
scientific_abstract
curated_evidence
```

---

## 14. Pipeline d’ingestion

```mermaid
flowchart LR
    SOURCE[API or document] --> PARSE[parse]
    PARSE --> NORMALIZE[normalize]
    NORMALIZE --> VALIDATE[validate provenance]
    VALIDATE --> HASH[compute content hash]
    HASH --> EXISTS{already indexed?}
    EXISTS -->|yes and unchanged| SKIP[skip]
    EXISTS -->|no or changed| CHUNK[chunk]
    CHUNK --> META[attach metadata]
    META --> EMBED[BGE-M3 embedding]
    EMBED --> UPSERT[upsert Qdrant]
    UPSERT --> TRACK[record ingestion in PostgreSQL]
```

### 14.1 Contraintes

- chunking déterministe ;
- conserver les limites de sections ;
- ne pas mélanger plusieurs protéines dans un chunk ;
- conserver `protein_id`, `source`, `taxonomy_id` et `document_type` ;
- calculer un SHA-256 du contenu normalisé ;
- rendre l’ingestion idempotente ;
- réindexer uniquement si le hash change.

---

## 15. Pipeline de retrieval

Entrée :

```text
question + protein_id + taxonomy_id + document types
```

Étapes :

1. normaliser la question ;
2. générer l’embedding BGE-M3 ;
3. appliquer les filtres Qdrant ;
4. récupérer le top K ;
5. supprimer les doublons ;
6. rejeter les documents sans provenance ;
7. rejeter les documents d’une autre protéine lorsque le filtre est strict ;
8. construire un contexte borné ;
9. enregistrer le retrieval dans PostgreSQL.

Filtres minimaux :

```text
domain = protein
protein_id = accession UniProt canonique
taxonomy_id = taxonomie demandée
document_type in allowed types
```

Le retrieval ne doit jamais élargir silencieusement l’espèce ou la protéine.
Tout élargissement doit créer un warning explicite.

---

## 16. API REST `/api/v1`

### 16.1 Endpoints obligatoires

| Méthode | Route | Usage |
|---|---|---|
| `POST` | `/api/v1/protein-structure-analyses` | recevoir une tâche et lancer le workflow |
| `GET` | `/api/v1/protein-structure-analyses/{analysis_id}` | récupérer statut et résultat |
| `GET` | `/api/v1/health` | santé de l’application |
| `GET` | `/api/v1/ready` | état PostgreSQL, Qdrant et configuration |

### 16.2 Endpoint d’ingestion

| Méthode | Route | Usage |
|---|---|---|
| `POST` | `/api/v1/knowledge/protein/ingestions` | indexer un jeu de documents contrôlé |

Cet endpoint doit être protégé par une clé interne en environnement autre que local.

### 16.3 Réponse de création

```http
HTTP/1.1 202 Accepted
Content-Type: application/json
```

```json
{
  "analysis_id": "uuid",
  "task_id": "uuid",
  "status": "RECEIVED"
}
```

### 16.4 Erreurs

Utiliser un format de type RFC 9457 :

```json
{
  "type": "https://umbrella.local/problems/invalid-protein-request",
  "title": "Invalid protein request",
  "status": 422,
  "detail": "uniprot_accession or protein_sequence is required",
  "instance": "/api/v1/protein-structure-analyses",
  "trace_id": "uuid"
}
```

---

## 17. Structure GitHub attendue

```text
umbrella-protein-agent/
├── app/
│   ├── main.py
│   ├── api/
│   │   └── v1/
│   │       ├── routes/
│   │       │   ├── analyses.py
│   │       │   ├── knowledge.py
│   │       │   └── health.py
│   │       └── dependencies.py
│   ├── contracts/
│   │   ├── agent_task.py
│   │   ├── protein_request.py
│   │   ├── protein_response.py
│   │   └── problem_details.py
│   ├── domain/
│   │   ├── models/
│   │   ├── enums.py
│   │   └── exceptions.py
│   ├── orchestrators/
│   │   └── protein/
│   │       ├── orchestrator.py
│   │       ├── state.py
│   │       ├── graph.py
│   │       ├── routers.py
│   │       └── nodes/
│   ├── capabilities/
│   │   ├── identity/
│   │   ├── structures/
│   │   ├── annotations/
│   │   ├── residue_mapping/
│   │   ├── retrieval/
│   │   ├── visualization/
│   │   ├── explanation/
│   │   └── critic/
│   ├── tools/
│   │   ├── uniprot_client.py
│   │   ├── rcsb_client.py
│   │   ├── alphafold_client.py
│   │   ├── interpro_client.py
│   │   └── sifts_client.py
│   ├── persistence/
│   │   ├── db.py
│   │   ├── models/
│   │   └── repositories/
│   ├── knowledge_base/
│   │   ├── qdrant.py
│   │   ├── embeddings.py
│   │   ├── ingestion.py
│   │   ├── retrieval.py
│   │   └── schemas.py
│   ├── llm/
│   │   ├── azure_foundry.py
│   │   ├── prompts.py
│   │   └── schemas.py
│   ├── configuration/
│   │   └── settings.py
│   └── observability/
│       ├── logging.py
│       └── metrics.py
├── migrations/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── e2e/
│   └── fixtures/
├── scripts/
│   ├── run_orchestrator.py
│   ├── seed_qdrant.py
│   └── smoke_test.py
├── docs/
│   ├── agent-card.md
│   ├── api-contract.md
│   └── workflow.md
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
├── .gitignore
└── README.md
```

---

## 18. Configuration

### 18.1 `.env.example`

```env
APP_ENV=local
LOG_LEVEL=INFO

DATABASE_URL=postgresql+psycopg://umbrella:umbrella@localhost:5432/umbrella
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=
QDRANT_COLLECTION=protein_knowledge

EMBEDDING_MODEL=BAAI/bge-m3

AZURE_AI_ENDPOINT=
AZURE_AI_API_KEY=
AZURE_AI_DEPLOYMENT=
AZURE_AI_API_VERSION=

UNIPROT_BASE_URL=https://rest.uniprot.org
RCSB_SEARCH_BASE_URL=https://search.rcsb.org/rcsbsearch/v2
RCSB_DATA_BASE_URL=https://data.rcsb.org
RCSB_FILES_BASE_URL=https://files.rcsb.org
ALPHAFOLD_BASE_URL=https://alphafold.ebi.ac.uk/api
INTERPRO_BASE_URL=https://www.ebi.ac.uk/interpro/api
SIFTS_BASE_URL=https://www.ebi.ac.uk/pdbe/api

HTTP_TIMEOUT_SECONDS=15
HTTP_MAX_RETRIES=2
MAX_STRUCTURE_CANDIDATES=10
RETRIEVAL_TOP_K=5
INTERNAL_INGESTION_API_KEY=change-me
```

### 18.2 Sources de données

Le runtime utilise exclusivement les APIs réelles et BGE-M3. Aucun sélecteur de
fixtures n'est exposé par la configuration. Les doubles déterministes sont
strictement limités aux tests automatisés.

---

## 19. Observabilité et gestion des erreurs

### 19.1 Logs structurés

Chaque log doit contenir :

```text
timestamp
level
trace_id
task_id
analysis_id
node
capability
status
duration_ms
error_code
```

Exemple :

```text
[INFO] trace_id=... task_id=... node=search_pdb status=completed candidates=3 duration_ms=421
```

### 19.2 Retries

Retry uniquement pour :

- HTTP 429 ;
- HTTP 500, 502, 503, 504 ;
- timeout réseau ;
- erreur de connexion.

Ne pas retry pour :

- HTTP 400 ;
- HTTP 401 ;
- HTTP 403 ;
- HTTP 404 biologique attendu ;
- validation métier ;
- ambiguïté d’identité.

### 19.3 Dégradation contrôlée

- PDB échoue, AlphaFold réussit → `PARTIAL` ou `COMPLETED` avec warning.
- InterPro échoue → structure encore utilisable, warning.
- Qdrant échoue → structure encore utilisable, explication limitée.
- UniProt échoue ou identité ambiguë → arrêt.
- SIFTS échoue pour une mutation demandée → ne pas surligner, `REVISE` ou `ABSTAIN`.
- LLM échoue → retourner les preuves structurées sans explication générative.

---

## 20. Tests obligatoires

### 20.1 Tests unitaires

Tester :

- validation des contrats ;
- parsing des réponses UniProt ;
- normalisation PDB ;
- normalisation AlphaFold ;
- scoring des candidats ;
- règles de sélection ;
- filtres Qdrant ;
- construction de l’EvidencePack ;
- construction de la configuration Mol* ;
- verdict du Scientific Critic ;
- chaque route conditionnelle LangGraph.

### 20.2 Tests d’intégration

Tester :

- FastAPI → ProteinSubOrchestrator ;
- orchestrateur → LangGraph ;
- LangGraph → fournisseurs biologiques réels ;
- LangGraph → PostgreSQL ;
- LangGraph → Qdrant ;
- ingestion → embedding → Qdrant ;
- retrieval → filtres → documents validés.

### 20.3 Scénarios end-to-end obligatoires

#### E2E-01 — PDB disponible

Entrée :

```text
TP53 / P04637 / Homo sapiens
```

Attendu :

```text
RCSB_PDB sélectionné
status = COMPLETED
validation_status = ACCEPT
```

#### E2E-02 — Fallback AlphaFold

```text
PDB retourne zéro candidat valide
AlphaFold retourne un modèle
```

Attendu :

```text
ALPHAFOLD_DB sélectionné
structure_type = PREDICTED
warning explicite
```

#### E2E-03 — Identité ambiguë

Attendu :

```text
NEEDS_CLARIFICATION ou ABSTAIN
aucune structure sélectionnée
```

#### E2E-04 — Timeout PDB

```text
PDB timeout
AlphaFold disponible
```

Attendu :

```text
résultat utilisable
warning PDB_TIMEOUT
aucun crash
```

#### E2E-05 — Aucune structure

```text
PDB vide
AlphaFold vide
```

Attendu :

```text
NO_STRUCTURE_FOUND
```

#### E2E-06 — Mapping résidu impossible

```text
residue_position demandé
SIFTS ne fournit pas de mapping
```

Attendu :

```text
résidu non surligné
validation_status = REVISE ou ABSTAIN
warning explicite
```

#### E2E-07 — Qdrant indisponible

Attendu :

```text
structure retournée
explication limitée
warning RETRIEVAL_UNAVAILABLE
```

#### E2E-08 — Idempotence

Deux appels avec la même `idempotency_key`.

Attendu :

```text
une seule tâche métier
aucune double exécution destructive
même analysis_id ou réponse de réutilisation
```

---

## 21. Ordre d’implémentation

L’agent de code doit respecter cet ordre.

### Phase 0 — Initialisation

- créer l’arborescence ;
- créer `requirements.txt` ;
- créer `.env.example` ;
- créer Docker Compose avec PostgreSQL et Qdrant ;
- configurer FastAPI ;
- configurer pytest.

### Phase 1 — Contrats et domaine

- Pydantic HTTP ;
- dataclasses métier ;
- enums ;
- exceptions ;
- tests de validation.

### Phase 2 — PostgreSQL

- modèles SQLAlchemy ;
- repositories ;
- migration Alembic ;
- tests repository.

### Phase 3 — Outils biologiques réels

- interfaces des clients ;
- client UniProt réel ;
- recherche et Data API RCSB PDB réelles ;
- client AlphaFold réel ;
- client InterPro réel ;
- client PDBe SIFTS réel ;
- tests.

### Phase 4 — LangGraph

- state ;
- nodes ;
- routers ;
- parallélisation ;
- fallbacks ;
- critic ;
- tests du graphe.

### Phase 5 — API REST

- POST analysis ;
- GET analysis ;
- health ;
- readiness ;
- script local `run_orchestrator.py` utilisant les fournisseurs réels.

### Phase 6 — Qdrant et BGE-M3

- collection ;
- payload schema ;
- ingestion ;
- retrieval ;
- filtres ;
- tests.

### Phase 7 — LLM Azure Foundry

- provider configurable ;
- structured output ;
- prompt evidence-only ;
- fallback sans LLM ;
- tests avec doubles déterministes isolés du runtime.

### Phase 8 — APIs réelles

- implémenter les vrais clients sous les interfaces existantes ;
- préserver les modèles normalisés ;
- timeouts ;
- retries ;
- tests contractuels non bloquants.

### Phase 9 — End-to-end

- exécuter les huit scénarios obligatoires ;
- documenter les résultats ;
- corriger les écarts ;
- ne pas ajouter de nouvelle fonctionnalité.

Ne pas commencer une phase tant que les tests de la phase précédente échouent.

---

## 22. Definition of Done

Le sprint est terminé uniquement si :

- [ ] `docker compose up -d` démarre PostgreSQL et Qdrant ;
- [ ] `pip install -r requirements.txt` fonctionne ;
- [ ] `alembic upgrade head` fonctionne ;
- [ ] FastAPI démarre ;
- [ ] le runner local envoie une tâche réelle ;
- [ ] LangGraph exécute plusieurs nœuds ;
- [ ] la route PDB-first fonctionne ;
- [ ] le fallback AlphaFold fonctionne ;
- [ ] Qdrant contient des points BGE-M3 ;
- [ ] le retrieval applique des filtres de métadonnées ;
- [ ] PostgreSQL contient la tâche, les runs, les tool calls et la sélection ;
- [ ] la réponse contient les preuves et warnings ;
- [ ] une prédiction AlphaFold est clairement marquée comme prédite ;
- [ ] le mapping SIFTS empêche les faux surlignages ;
- [ ] le Scientific Critic retourne un verdict ;
- [ ] tous les tests unitaires passent ;
- [ ] tous les tests d’intégration passent ;
- [ ] les huit scénarios end-to-end passent ;
- [ ] aucun secret n’est commité ;
- [ ] le README décrit les commandes d’exécution ;
- [ ] le code est typé, formaté et sans duplication évidente.

---

## 23. Interdictions explicites

L’agent de code ne doit pas :

1. remplacer LangGraph par CrewAI ;
2. ajouter ChromaDB, MongoDB, Zep ou Neo4j ;
3. remplacer PostgreSQL par une autre base ;
4. remplacer Qdrant par pgvector ;
5. créer `pyproject.toml` ;
6. ajouter Redis, Celery, Kafka ou MinIO dans ce sprint ;
7. implémenter un frontend ;
8. appeler Ensembl directement dans le workflow Protein final ;
9. appeler PubMed ou Europe PMC directement ;
10. mettre des fichiers mmCIF/PDB dans Qdrant ;
11. mettre les embeddings dans PostgreSQL ;
12. laisser le LLM choisir une structure sans règle déterministe ;
13. présenter AlphaFold comme expérimental ;
14. inventer une fonction biologique ;
15. surligner un résidu sans mapping SIFTS validé ;
16. ignorer une ambiguïté d’espèce ou d’identité ;
17. masquer silencieusement une erreur d’API ;
18. supprimer les warnings de provenance ;
19. introduire une fonctionnalité non décrite ;
20. modifier les contrats sans mettre à jour les tests.

---

## Commandes attendues

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt

docker compose up -d postgres qdrant

alembic upgrade head

uvicorn app.main:app --reload

python scripts/seed_qdrant.py
python -m scripts.run_orchestrator

pytest -q
```

Sous Windows PowerShell :

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -r requirements-dev.txt
docker compose up -d postgres qdrant
alembic upgrade head
uvicorn app.main:app --reload
```

---

## Résultat démontrable attendu

```text
1. Le Grand Orchestrator ou le runner local envoie une tâche P04637.
2. FastAPI valide le contrat.
3. PostgreSQL enregistre la tâche.
4. LangGraph confirme l’identité UniProt.
5. PDB, InterPro et Qdrant sont interrogés en parallèle.
6. PDB est évalué.
7. AlphaFold est appelé uniquement si nécessaire.
8. La meilleure structure est sélectionnée par règles déterministes.
9. SIFTS mappe les résidus lorsque nécessaire.
10. Un EvidencePack est constitué.
11. Une configuration Mol* est générée.
12. Le LLM explique uniquement les preuves.
13. Le Scientific Critic valide ou refuse.
14. PostgreSQL enregistre le résultat.
15. Le Grand Orchestrator ou le runner local reçoit la réponse finale.
```
