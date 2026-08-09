# Recognition Agent — Décisions validées

**Projet :** IEEE Umbrella  
**Agent :** Multimodal Species Recognition Agent  
**Référence arrêtée le :** 6 août 2026

## 1. Mission de Recognition

- Recognition identifie une espèce animale à partir d’une image.
- Une seule image est traitée par requête.
- L’agent ne réalise pas de comparaison entre deux images.
- L’agent ne traite pas de vidéo ni d’audio.
- L’image constitue la preuve scientifique principale.
- Une instruction textuelle peut accompagner l’image, mais elle ne peut pas créer un candidat absent des résultats du retrieval.

## 2. Architecture retenue

- **GPT-5 mini** est le cerveau intelligent et autonome de Recognition.
- **LangGraph** contrôle l’exécution et les étapes autorisées.
- **BioCLIP-2** transforme l’image en représentation vectorielle.
- **Qdrant** recherche les espèces visuellement proches.
- **GBIF et NCBI** valident et normalisent la taxonomie.
- Une **logique déterministe** calcule la confiance scientifique.

L’agent est défini par l’ensemble suivant :

```text
Recognition
= GPT-5 mini pour comprendre, planifier et expliquer
+ LangGraph pour contrôler le workflow
+ BioCLIP-2 et Qdrant pour produire les preuves de reconnaissance
+ GBIF et NCBI pour la validation et la normalisation taxonomiques
+ logique déterministe pour la décision de confiance
```

## 3. Rôle et limites de GPT-5 mini

GPT-5 mini peut :

- comprendre la demande ;
- planifier l’exécution interne de Recognition ;
- exploiter les indices textuels sûrs ;
- analyser les preuves structurées produites par le workflow ;
- produire une explication finale fondée sur ces preuves.

GPT-5 mini ne peut pas :

- inventer une espèce ;
- ajouter ou supprimer un candidat retourné par Qdrant ;
- modifier les scores ;
- transformer un score de similarité en probabilité ;
- inventer un identifiant GBIF ou NCBI ;
- appeler directement un autre agent ;
- contourner les validations ou les étapes obligatoires.

Le nombre maximal d’appels LLM par requête est de deux :

1. un appel de planification ;
2. un appel d’explication finale fondée sur les preuves.

Si GPT-5 mini échoue ou est indisponible, Recognition utilise un fallback déterministe.

## 4. Workflow cible

```text
Image + instruction facultative
→ validation de l’entrée
→ planification par GPT-5 mini
→ BioCLIP-2
→ recherche Qdrant Top-K
→ agrégation des candidats
→ validation et normalisation par GBIF et NCBI
→ calcul déterministe de la confiance
→ explication fondée par GPT-5 mini
→ AgentResult
```

## 5. Fonctions acceptées

- Retourner les candidats visuellement proches **Top-K**.
- Retourner une décision parmi :
  - `identified` ;
  - `uncertain` ;
  - `not_identified`.
- Demander une meilleure image lorsque la preuve visuelle est insuffisante.
- Retourner `needs_agent` lorsqu’une expertise biologique complémentaire est nécessaire.
- Indiquer, si nécessaire, la capacité ou l’agent recommandé au Global Orchestrator.

La demande d’une meilleure image n’est pas une route générale de clarification. Elle intervient seulement après un résultat visuel insuffisant.

Recognition ne contacte jamais directement l’agent recommandé. Le Global Orchestrator décide et effectue l’appel éventuel.

## 6. Composants retenus pour le Sprint 2

| Composant | Décision validée |
|---|---|
| GPT-5 mini | Fake pour les tests, puis test réel contrôlé |
| LangGraph | Réel |
| BioCLIP-2 | Mock |
| Qdrant | Mock pour les tests unitaires, puis réel minimal pour l’intégration et la démonstration |
| GBIF | Mock |
| NCBI | Mock |
| Confiance | Réelle et déterministe |
| API Recognition | Réelle sur le port `8005` |

Les composants BioCLIP-2, GBIF et NCBI réels seront intégrés dans des sprints ultérieurs.

Le gate d’intégration avec le Qdrant réel reste à réaliser avec les éléments fournis par Chahd. Il n’est pas considéré comme validé à ce stade.

## 7. Comportement des mocks GBIF et NCBI

Les mocks GBIF et NCBI sont présents dès le Sprint 2 pour tester le workflow sans appeler les services externes réels.

Ils simulent les situations suivantes :

- GBIF et NCBI disponibles ;
- GBIF disponible et NCBI absent ;
- candidat absent des fixtures taxonomiques ;
- timeout ou indisponibilité simulée.

GBIF et NCBI reçoivent un candidat provenant de Qdrant. Ils peuvent :

- vérifier le nom scientifique ;
- résoudre un synonyme vers un nom accepté ;
- normaliser le rang et la classification ;
- retourner les identifiants disponibles ;
- signaler un résultat partiel ou non vérifié.

GBIF et NCBI ne choisissent pas l’espèce et ne peuvent pas introduire une espèce absente des candidats Qdrant.

## 8. Communication avec le Global Orchestrator

- Le Global Orchestrator appelle Recognition avec `POST /execute`.
- Recognition répond avec le contrat commun `AgentResult`.
- Recognition peut inclure `needs_agent` dans sa réponse.
- Recognition n’importe, ne lance et n’appelle directement aucun autre agent.
- Le profil biologique complet est assemblé par l’écosystème multi-agent, et non directement par Recognition.

## 9. Contenu attendu du résultat

Selon le cas traité, le résultat peut contenir :

- l’espèce identifiée ;
- le nom scientifique normalisé ;
- le score de confiance ;
- la décision scientifique ;
- les candidats Top-K ;
- les identifiants GBIF et NCBI disponibles ;
- une demande éventuelle de meilleure image ;
- un signal `needs_agent` éventuel ;
- la capacité ou l’agent recommandé, si nécessaire ;
- la provenance précisant les composants réels et mockés ;
- une explication fondée uniquement sur les preuves disponibles.

## 10. Règles scientifiques et de sécurité

- Aucune espèce ne peut provenir uniquement du LLM ou du texte utilisateur.
- Les candidats scientifiques doivent provenir du retrieval Qdrant.
- Le texte ne peut pas remplacer la preuve visuelle.
- Les scores et la confiance ne sont pas décidés ni modifiés par GPT-5 mini.
- Les identifiants taxonomiques ne sont pas inventés.
- Les données brutes de l’image, le Base64, les vecteurs et les credentials ne sont pas envoyés au LLM.
- Une preuve insuffisante conduit à `uncertain` ou `not_identified`.

## 11. Ordre des prochaines étapes convenu

1. Utiliser ce document comme référence officielle des décisions de Recognition.
2. Préparer une instruction d’implémentation limitée au gate GPT-5 mini + LangGraph en environnement mock.
3. Implémenter ce gate avec un faux GPT-5 mini, un Mock BioCLIP-2, un Mock Qdrant, un Mock GBIF et un Mock NCBI.
4. Exécuter et vérifier les tests de ce gate avant d’autoriser la suite.
5. Après validation du gate mock, intégrer le Qdrant réel avec les éléments fournis par Chahd.
6. Continuer ensuite les tests d’intégration, l’évaluation, le service Recognition, la communication avec le Global Orchestrator et la démonstration finale.

La première instruction d’implémentation ne doit pas inclure les phases suivantes ni déclarer le gate Qdrant réel comme validé.

## 12. Priorité de cette référence

Ce document conserve uniquement les décisions validées pour Recognition. Lorsqu’une ancienne description du projet contredit ces décisions, la présente référence est celle à appliquer pour la suite du travail sur l’agent.
