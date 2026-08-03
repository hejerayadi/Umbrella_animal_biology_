| Field | Value |
|---|---|
| name | Functional Evidence |
| type | sub-orchestrator |
| description | Cross-references KEGG pathway membership and UniProt protein function over the same candidate gene set, in parallel. |
| reports_to | Trait Discovery Agent |
| may_need | Gene Mapper — data dependency only: supplies gene_list and go_annotations upstream |
| managed_agents | Pathways; Protein Data |
| managed_services | — none |
| capabilities | Parallel evidence retrieval; Cross-referencing gene function across two sources |
| input | gene_list: list[str] (from Gene Mapper, via orchestrator); go_annotations: list[GOAnnotation] |
| output | pathway_data: list[PathwayEntry]; protein_data: list[ProteinEntry]; status / target_agent / prompt_to_target_agent |