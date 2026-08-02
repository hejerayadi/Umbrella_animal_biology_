| Field | Value |
|---|---|
| name | Protein Data |
| type | worker |
| description | For a list of candidate genes, retrieves the corresponding reviewed UniProt protein entries with name and function summary. |
| reports_to | Functional Evidence |
| may_need | Gene Mapper — data dependency only: same gene_list as Pathways, passed down via Functional Evidence |
| managed_agents | — none |
| managed_services | UniProt REST API (one query per gene symbol) |
| capabilities | Reviewed protein entry lookup; Missing-entry flagging |
| input | gene_list: list[str] (from Gene Mapper, same list as Pathways) |
| output | proteins: list[ProteinEntry]; missing_genes: list[str]; status / target_agent / prompt_to_target_agent |