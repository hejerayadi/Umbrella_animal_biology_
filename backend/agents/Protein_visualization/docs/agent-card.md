# Protein Structure Visualization Agent Card

- **Name:** `umbrella-protein-agent`
- **Version:** `1.1.0`
- **Protocol:** JSON over HTTP, `{data, meta, error}` envelope on every `/api/v1` response
- **Capability:** `protein-structure.orchestrate`
- **Input:** an `AgentTask` from the Grand Orchestrator (`task_id`, `trace_id`, `idempotency_key`, and an `input` block carrying `resolved_gene_id`, `species`, optional `uniprot_accession`, `protein_sequence`, `requested_regions`, `residue_position`, `mutation`, `preferred_source`)
- **Output:** an `AgentResult` routing contract (`completed`, `continue`, `needs_agent`, `failed`) whose `output` contains canonical identity, selected structure with alternatives, InterPro annotations, SIFTS residue mappings, knowledge evidence, Mol* configuration, grounded explanation, and a `validation_status` of `ACCEPT`, `REVISE`, or `ABSTAIN`
- **Delegation:** unresolved canonical identity targets `genome_agent`; absence of supportable structural evidence targets `literature_agent`; transient provider failures remain with this agent as retryable `continue`
- **Evidence sources:** UniProt, RCSB PDB, AlphaFold DB, InterPro, PDBe SIFTS, and the `protein_knowledge` Qdrant collection
- **Correlation:** `X-Trace-Id` and `X-Request-Id` are accepted from the caller, echoed on the response, and present on every log line
- **Failure policy:** identity failure stops the workflow; optional capability failures are returned as warnings with `PARTIAL` status
- **Safety:** generated explanations must remain grounded in returned evidence, must not highlight a residue without a validated SIFTS mapping, and must clearly distinguish predicted from experimental structures
