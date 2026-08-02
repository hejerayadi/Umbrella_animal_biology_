| Field | Value |
|---|---|
| name | Trait Discovery Agent |
| type | orchestrator |
| description | Given a biological trait and species, identifies the responsible genes, functional evidence (pathways + proteins), and supporting literature; updates the knowledge graph and returns a plain-language explanation. |
| reports_to | Global Scientific Orchestrator |
| may_need | Genome Agent; Literature Agent |
| managed_agents | Gene Mapper; Functional Evidence; Literature Support |
| managed_services | — none |
| capabilities | Trait-to-gene routing; Result aggregation; Knowledge graph update; Plain-language explanation generation |
| input | trait_name: str; species_name: str |
| output | go_annotations: list[GOAnnotation]; pathway_data: list[PathwayEntry]; protein_data: list[ProteinEntry]; evidence: list[LiteratureRecord]; explanation: str; status / target_agent / prompt_to_target_agent |