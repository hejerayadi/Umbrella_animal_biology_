# Genome Agent

## Objective
Retrieve, explain, and visualize genomic information about animal species.

## Main Tasks
- Show genomes and genome sizes for animal species.
- Report chromosome counts and other core genomic facts.
- Explain gene annotations and related biological context.
- Support researchers exploring animal genomics through clear, concise answers.

## Example Questions
- Show me the genome of the woolly mammoth.
- What is the genome size of the tiger?
- How many chromosomes does the Asian elephant have?
- Show gene annotations for the polar bear.

## Two agent cards, and why

This agent has its card in two places. They are read by different code and
both are required.

- `card.json` (this folder) is the **platform** card. `backend/registry.py`
  loads one of these per agent at import time to build `AGENT_CARDS`, which is
  what the Global Orchestrator's planner and capability resolver choose
  between. Its fields are the shared format every agent uses - do not add
  fields here. Deleting this file stops the whole orchestrator from starting,
  not just this agent.

- `agent_cards/genome_agent.json` is this agent's entry in its **own** local
  catalog, read by `workflows/agent_catalog.py` for the internal capability
  resolver. That folder also holds cards for the four internal subagents and
  for the external agents this one may escalate to. It supports extra fields
  the shared format does not (`role`, `call_when`, `cannot_help_with`).

The two copies describe the same agent and must be kept in step by hand - JSON
carries no comments, so there is nowhere to say this in the files themselves.
They had already drifted once: `may_need` listed one agent in `card.json` and
three in `agent_cards/genome_agent.json`. If you change one, change the other.
