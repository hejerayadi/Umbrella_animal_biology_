| Field | Value |
|---|---|
| name | Gene Mapper |
| type | worker |
| description | Enriches a gene list received from the Genome Agent with trait-relevant Gene Ontology annotations. |
| reports_to | Trait Discovery Agent |
| may_need | Genome Agent — data dependency only: supplies gene_list upstream, orchestrator makes the cross-agent call |
| managed_agents | — none |
| managed_services | Gene Ontology REST API (QuickGO / AmiGO) |
| capabilities | GO term lookup by gene symbol; Flagging genes with no GO match |
| input | trait_name: str; gene_list: list[str] (from Genome Agent); species_name: str |
| output | go_annotations: list[GOAnnotation]; unmatched_genes: list[str]; status / target_agent / prompt_to_target_agent |