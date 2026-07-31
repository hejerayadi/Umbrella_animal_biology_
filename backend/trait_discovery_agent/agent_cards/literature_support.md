| Field | Value |
|---|---|
| name | Literature Support |
| type | worker |
| description | Requests supporting literature evidence for a trait + gene pair from the Literature Agent, validates the response, and writes the trait→gene relationship (backed by pmid) to the Neo4j knowledge graph. Does not query PubMed directly. |
| reports_to | Trait Discovery Agent |
| may_need | Gene Mapper — data dependency only: supplies gene_list upstream; Literature Agent — direct cross-agent call for evidence |
| managed_agents | — none |
| managed_services | — none |
| capabilities | Evidence request delegation; Result validation & deduplication by pmid; Knowledge graph relationship write |
| input | trait_name: str; gene_list: list[str] (from Gene Mapper) |
| output | evidence: list[LiteratureRecord]; status / target_agent / prompt_to_target_agent |