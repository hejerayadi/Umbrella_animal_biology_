| Field | Value |
|---|---|
| name | Pathways |
| type | worker |
| description | For a list of candidate genes, retrieves all associated KEGG pathways and returns pathway IDs and names. |
| reports_to | Functional Evidence |
| may_need | Gene Mapper — data dependency only: gene_list originates here, passed down via Functional Evidence |
| managed_agents | — none |
| managed_services | KEGG REST API (gene-to-pathway link endpoint) |
| capabilities | Pathway lookup by gene list; Malformed-ID validation |
| input | gene_list: list[str] (from Gene Mapper, via Functional Evidence) |
| output | pathways: list[PathwayEntry]; malformed_ids: list[str]; status / target_agent / prompt_to_target_agent |