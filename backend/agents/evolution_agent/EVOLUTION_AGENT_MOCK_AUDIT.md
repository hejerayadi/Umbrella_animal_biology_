# EVOLUTION AGENT — MOCK-PHASE AUDIT

**Branche** `group_d_evolution_agent` · **HEAD** `49540bd` · **Base** `main` (`ff10927`)
**Périmètre** — providers bioinformatiques volontairement mockés ; l'audit porte sur la **sélection de branche**, la séparation des sous-agents et la qualité des mocks.
**Ce rapport remplace l'audit précédent.**

> Aucun fichier de production modifié. Aucun commit, aucun push, aucun changement de branche.
> Ajouts limités au dossier de tests : `tests/test_branch_acceptance.py`, `tests/test_mock_quality_audit.py`.

---

## 1. Architecture cible vs implémentation réelle

```
CIBLE                                    RÉEL (49540bd)

User request                             User request
    ↓                                        ↓
LLM intent classification                LLM intent classification   ✅ correcte
    ↓                                        ↓
Evolution Orchestrator                   Evolution Orchestrator
    ↓                                        ↓
UNE SEULE branche                        asyncio.gather(MC, PHYLO)   ❌ les deux, toujours
    ↓                                        ↓
similarity_network                       similarity_network
   OU phylogenetic_tree                     ET phylogenetic_tree
```

Le classifieur d'intention produit bien `molecular_comparison` / `phylogenetic_tree` / `None`.
`_plan_node` ([evolution_orchestrator.py:178-189](orchestrator/evolution_orchestrator.py:178)) valide cette valeur puis l'enregistre dans `planned_feature`. **Aucun nœud du graphe ne relit jamais `planned_feature`.**
`_dispatch_node` ([:231-247](orchestrator/evolution_orchestrator.py:231)) lance systématiquement les deux workers :

```python
mc_result, phylo_result = await asyncio.gather(
    call(self._mc_worker),
    call(self._phylo_worker),
)
```

Le module qui implémenterait la sélection, [orchestrator/router.py](orchestrator/router.py) (`Router.pick` / `pick_many`), **n'est importé par aucun fichier**. Idem pour [orchestrator/aggregator.py](orchestrator/aggregator.py).

---

## 2. Smoke test live — classification d'intention (3 appels Azure GPT-5-mini)

| Prompt | Branche attendue | `feature` retourné | Espèces extraites | Durée |
|---|---|---|---|---|
| « Compare these species and build a similarity network: human, chimp, mouse. » | `similarity_network` | **`molecular_comparison`** ✅ | `[Homo sapiens, Pan troglodytes, Mus musculus]` | 5.70 s |
| « Build a phylogenetic tree for human, mouse and chicken. » | `phylogenetic_tree` | **`phylogenetic_tree`** ✅ | `[Homo sapiens, Mus musculus, Gallus gallus]` | 2.03 s |
| « Tell me something about evolution. » | clarification | **`None`** ✅ | `[]` | 1.72 s |

**La couche d'intention fonctionne.** Elle produit exactement la bonne décision dans les trois cas. Cette décision est ensuite jetée par l'orchestrateur. Le défaut est donc localisé, isolé et corrigeable sans toucher au LLM.

*Défaut de diagnostic* : quand le modèle répond correctement `feature: null`, [intent.py:153](intent.py:153) étiquette le résultat `source="unparsable"`. Un refus légitime et une panne de parsing deviennent indiscernables dans les logs.

---

## 3. Exécution des tests

### 3.1 Suite existante

```bash
python -m pytest backend/agents/evolution_agent/tests -v
```
**75 collected · 75 passed · 0 failed · 0 skipped · 0.36 s**

**Trois de ces tests passent parce que le bug est présent** et ne peuvent pas être comptés comme des succès fonctionnels :

| Test | Ligne | Ce qu'il verrouille |
|---|---|---|
| `test_both_workers_are_called_in_parallel` | [test_orchestrator_pipeline.py:139](tests/test_orchestrator_pipeline.py:139) | `assert sorted(calls) == ["mc", "phylo"]` — « Both workers must be dispatched » |
| `test_phylo_runs_without_mc_alignment` | [:185](tests/test_orchestrator_pipeline.py:185) | « Phylo must not receive MC's alignment » |
| `test_source_agents_lists_both_workers` | [:107](tests/test_orchestrator_pipeline.py:107) | les deux sous-agents toujours cités, sans feature demandée |

**Tests fonctionnels réellement valides : 72 / 75.**

### 3.2 Suites ajoutées

```bash
python -m pytest backend/agents/evolution_agent/tests -q
```
**145 collected · 128 passed · 17 failed · 2.13 s**

| Suite | Tests | Passed | Failed |
|---|---|---|---|
| `test_orchestrator_pipeline.py` + `test_workers.py` + `test_adapter.py` (existants) | 75 | 75 | 0 |
| `test_mock_quality_audit.py` (ajouté — observation) | 33 | 33 | 0 |
| `test_branch_acceptance.py` (ajouté — acceptation) | 37 | 20 | **17** |

Les 17 échecs sont écrits contre le **contrat**, pas contre le code : ils constituent la preuve exécutable des défauts.

---

## 4. Les 17 échecs d'acceptation

| # | Test | Message |
|---|---|---|
| 1 | `test_similarity_request_never_calls_phylogenetic_reconstruction` | `Phylogenetic Reconstruction ran 1 times for a similarity-network request` |
| 2 | `test_similarity_result_carries_no_tree_fields` | `tree fields leaked: ['newick_tree','tree_url','bootstrap_support','confidence_values','model']` |
| 3 | `test_similarity_network_works_with_only_two_species` | `assert phylo.count == 0` → 1 |
| 4 | `test_two_species_similarity_request_with_the_real_mock_workers` | `returns failed: "Phylogenetic reconstruction requires at least 3 species; got 2"` |
| 5 | `test_tree_request_does_not_build_a_similarity_network` | `Molecular Comparison ran 1 times for a tree-only request` |
| 6 | `test_tree_result_carries_no_network_fields` | `network fields leaked: ['similarity_network','similarity_scores','species_groups','alignment_url']` |
| 7 | `test_two_intents_produce_different_worker_call_patterns` | `network branch called (mc=1, phylo=1)` |
| 8 | `test_two_intents_do_not_produce_identical_json` | `a similarity-network request and a tree request returned identical JSON` |
| 9 | `test_ambiguous_request_returns_a_structured_clarification` | `produced only a prose string (status=FAILED, output_type=str)` |
| 10 | `test_full_analysis_is_not_advertised_as_the_ambiguity_fallback` | le prompt contient « Use this when in doubt » |
| 11 | `test_missing_feature_does_not_silently_become_full_analysis` | `an unspecified feature silently ran both branches` |
| 12 | `test_tree_with_fewer_than_three_species_fails_in_its_own_branch` | `the unselected MC worker ran anyway` |
| 13 | `test_malformed_context_is_handled_without_crashing` | `TypeError: 'int' object is not iterable` — [orchestrator_adapter.py:57](orchestrator_adapter.py:57) |
| 14 | `test_failure_of_an_unselected_worker_cannot_affect_the_request` | `an unselected worker's failure broke the request` |
| 15 | `test_exception_of_an_unselected_worker_cannot_affect_the_request` | `RuntimeError: mock IQ-TREE segfault` remonte |
| 16 | `test_worker_exception_becomes_a_failed_result_not_a_crash` | `worker exception escaped the orchestrator` |
| 17 | `test_exactly_two_sub_agent_worker_packages_are_shipped` | `['divergence_time','molecular_comparison','phylogenetic_tree']` |

**Cas #4 — bloquant pour la démo.** « How similar are humans and chimpanzees at the molecular level? » est la question canonique de démonstration. Le LLM classe correctement `molecular_comparison` et extrait 2 espèces. Le mock de comparaison moléculaire accepte 2 espèces. Mais le mock phylogénétique, lancé sans avoir été demandé, exige 3 taxa, échoue, et **fait échouer toute la requête**. Ce comportement a également été reproduit en HTTP live sur le port 8002.

---

## 5. Qualité des mocks — 33/33

| Critère demandé | Verdict | Preuve |
|---|---|---|
| Clairement identifiés | **PASS** | classes `*Mock`, modules `mock.py`, docstrings nommant NCBI/ESM/MAFFT/IQ-TREE/UFBoot, `score_is_mock: true` dans la sortie publique, URLs sur hôte `.local` |
| Déterministes | **PASS** | même entrée → même sortie ; indépendants de l'ordre des espèces ; entrées différentes → sorties différentes |
| Reçoivent les bonnes entrées | **PASS** | noms normalisés (`strip().lower()`), lus depuis `species_list` / `context["species_list"]` / `context["species"]` ; l'orchestrateur transmet les noms scientifiques canoniques résolus |
| Appelés uniquement dans la bonne branche | **FAIL** | échecs #1, #5, #7 |
| Conformes aux schémas | **PASS** | `MolecularComparisonResult` / `PhylogeneticResult` complets ; `n(n-1)/2` scores ; réseau symétrique ; Newick terminé par `;` ; bootstrap entiers ; confiance ∈ [0,1] ; sortie JSON-sérialisable |
| Simulent succès et erreurs | **PASS** | MC refuse < 2 espèces et liste vide ; Phylo refuse < 3 espèces ; les deux refusent une espèce hors catalogue ; messages nommant le sous-agent fautif ; `AgentResult` conforme en cas d'échec |

Le **contenu** des mocks est de bonne facture. C'est leur **orchestration** qui est défaillante.

---

## 6. Défauts hors contrat de branche

| Sévérité | Défaut | Preuve |
|---|---|---|
| **High** | Une exception de worker traverse l'orchestrateur **et** l'adapter ; seul `api.py:139` l'arrête | échecs #15, #16 |
| **High** | `context["species_list"]` non-itérable → `TypeError` non gérée dans `resolve_species` | échec #13 ; [orchestrator_adapter.py:57](orchestrator_adapter.py:57) |
| **Medium** | `workers/divergence_time/` : 3ᵉ paquet worker, jamais câblé, et **cassé** — passe `divergence_times=` à `AgentResult` qui n'a plus ce champ → `TypeError` | échec #17 ; `test_DEFECT_divergence_time_...` ; [workers/divergence_time/mock.py:101](workers/divergence_time/mock.py:101) |
| **Medium** | `router.py` et `aggregator.py` : code mort, jamais importés | `test_DEFECT_router_and_aggregator_modules_are_never_imported` |
| **Medium** | `to_platform_result` non idempotent : réappliqué, il transforme tout le payload en chaîne dans `explanation` | `test_DEFECT_to_platform_result_is_not_idempotent` |
| **Medium** | Sans `EVOLUTION_AGENT_IMPL=orchestrator`, le service sert `mock.py` (escalade Genome → Literature) | `test_default_implementation_without_the_env_var_is_the_naive_mock` |
| **Low** | Une réponse LLM correcte `feature: null` est étiquetée `source="unparsable"` | [intent.py:153](intent.py:153) |
| **Low** | Les logs `[Intent]` ne sortent pas sous uvicorn — décisions de routing non traçables | log serveur du smoke test |

**Points positifs confirmés** : le contrat HTTP tient (`/execute` renvoie 200 + `status: failed` pour toute erreur métier, jamais 500) ; aucune fuite de credential dans les réponses ; un crash de worker est contenu par la frontière HTTP sans traceback exposée ; l'échec d'un worker **sélectionné** est rapporté proprement en nommant le sous-agent fautif ; timeout LLM et sortie LLM invalide ne déclenchent aucune branche.

---

## 7. Séparation des sous-agents

| Critère | Verdict | Preuve |
|---|---|---|
| Modules / classes distincts | **PASS** | `MolecularComparisonMock` et `PhylogeneticTreeMock`, modules séparés, `run.__code__` distincts |
| Responsabilités distinctes | **PASS** | fetch/embed/score/cluster/network vs align/model/tree/bootstrap |
| Schémas de sortie distincts | **PASS** | intersection des champs de `MolecularComparisonResult` et `PhylogeneticResult` = ∅ |
| Tests indépendants | **PASS** | chaque worker exécutable seul, sans orchestrateur |
| Aucun import/appel direct entre eux | **PASS** | vérifié statiquement dans les deux sens |
| Orchestration uniquement par le parent | **PASS** | aucun module hors du parent n'instancie un worker |
| Absence de 3ᵉ sous-agent actif | **FAIL** | `workers/divergence_time/` présent (mort et cassé) |

Séparation **structurelle** correcte. Ce qui manque n'est pas la séparation, c'est le **contrôle sélectif** exercé sur elle.

---

## 8. Autonomie avec les mocks

L'absence de providers réels n'est pas pénalisée.

| Capacité | Verdict | Commentaire |
|---|---|---|
| Compréhension de l'intention | **PASS** | 3/3 en live |
| Choix de la branche | **FAIL** | la classification n'est jamais relue |
| Sélection du bon worker | **FAIL** | `asyncio.gather` inconditionnel |
| Non-exécution du worker inutile | **FAIL** | le worker non sélectionné s'exécute et peut faire échouer la requête |
| Gestion des préconditions | **PASS** | feature invalide rejetée avant tout worker ; espèce non résolue → arrêt net avant tout worker ; < 2 espèces rejeté |
| Clarification | **FAIL** | `status: failed` + prose ; aucun champ structuré ; `card.json` annonce `clarification_required` |
| Gestion des erreurs | **PARTIAL** | échecs de worker propres ; exceptions non capturées ; erreur d'un worker non sélectionné contaminante |
| Conformité de la réponse | **PARTIAL** | contrat plateforme respecté ; mais le contenu ne correspond pas à la question posée |

**Score d'autonomie : 2 / 5** — « routing limité, faible réaction aux erreurs ». Les préconditions sont réellement validées et le workflow s'arrête proprement quand elles échouent ; mais aucune décision de branche n'est prise, et un sous-agent non demandé peut détruire une requête valide.

---

## 9. VERDICT

| # | Question | Verdict | Preuve |
|---|---|---|---|
| 1 | L'intention utilisateur sélectionne-t-elle réellement une seule branche ? | **FAIL** | échecs #7, #8, #11 ; `_plan_node` calcule `planned_feature`, aucun nœud ne le relit |
| 2 | Une demande similarity network lance-t-elle uniquement Molecular Comparison ? | **FAIL** | échec #1 : `phylo.count == 1` ; échec #2 : `newick_tree`/`tree_url`/`bootstrap_support` dans la réponse |
| 3 | Une demande phylogenetic tree lance-t-elle uniquement Phylogenetic Reconstruction ? | **FAIL** | échec #5 : `mc.count == 1` ; échec #6 : `similarity_network`/`similarity_scores` dans la réponse |
| 4 | Les deux sous-agents sont-ils réellement séparés ? | **PARTIAL** | 6/7 critères PASS ; un 3ᵉ paquet worker mort et cassé subsiste (échec #17) |
| 5 | L'orchestrateur contrôle-t-il leur exécution ? | **PARTIAL** | il possède le cycle de vie, l'état et l'agrégation ; il n'exerce **aucune sélection** |
| 6 | Les mocks sont-ils correctement utilisés ? | **PARTIAL** | 5/6 critères PASS (identification, déterminisme, entrées, schémas, erreurs) ; « bonne branche » FAIL |
| 7 | L'agent est-il autonome avec les mocks ? | **FAIL** — **2/5** | intention ✅, préconditions ✅ ; branche ❌, worker inutile ❌, clarification ❌ |
| 8 | Le workflow mock est-il prêt pour la démonstration Sprint 3 ? | **FAIL** | échec #4 : la question canonique de démo renvoie `failed` |

**Providers bioinformatiques réels** : **MOCK_EXPECTED** — hors périmètre, non comptés comme défaut.
**Appel LLM live** : **PASS** — classification correcte 3/3.

---

## 10. Synthèse

Le socle est meilleur que ne le suggère le verdict. Le classifieur d'intention prend la bonne décision à chaque fois, les deux sous-agents sont structurellement bien séparés, les mocks sont identifiés, déterministes, conformes aux schémas et simulent correctement leurs erreurs, et le contrat HTTP tient sous toutes les entrées testées. Un seul maillon manque : entre la décision et l'exécution.

Le commit `49540bd`, qui a remplacé le pipeline séquentiel par un fan-out parallèle, a supprimé le point où cette décision aurait pu être appliquée. Conséquence directe : chaque requête exécute les deux branches, chaque réponse contient les deux résultats, deux questions opposées renvoient un JSON identique, et un sous-agent que personne n'a demandé peut faire échouer une requête parfaitement valide — ce qui est exactement ce qui se produit sur la question de démonstration la plus probable.

**Trois corrections prioritaires** (non implémentées, conformément au périmètre) :

1. Faire relire `state.planned_feature` par `_dispatch_node` et n'appeler que le worker correspondant — `full_analysis` restant le seul cas à deux branches. `router.py` existe déjà pour ça.
2. Cesser d'utiliser `full_analysis` comme repli d'ambiguïté : retirer « Use this when in doubt » du prompt, retirer les `or "full_analysis"` de [orchestrator_adapter.py:68](orchestrator_adapter.py:68) et [evolution_orchestrator.py:189](orchestrator/evolution_orchestrator.py:189), et renvoyer une clarification structurée.
3. Envelopper le dispatch dans un `try/except` convertissant toute exception en `AgentResult(FAILED)`, et durcir `resolve_species` contre les types inattendus.
