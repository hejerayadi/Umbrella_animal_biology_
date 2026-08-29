# Test prompts — through the Global Orchestrator

Prompts to type into the frontend (or `POST /api/v1/workspace/chat`) to exercise
this agent end to end. Every prompt below was run against the live planner; the
routing column is measured, not assumed.

Both agents must be running the current code — `python -m backend.run_agents --reload`
from the repository root. Without `--reload` an agent serves whatever was on disk
when it started, which is how a fixed bug can appear to persist for hours.

## Genome → Reconstruction

The normal chain. The Genome Agent resolves the species, sees a `Scaffold` or
`Contig` assembly, locates the gap coordinates, and hands off.

> Reconstruct the unresolved regions of the polar bear genome

> Reconstruct the unresolved regions of the wombat genome

> Fill the gaps in the polar bear genome

> Are there unresolved regions in the polar bear genome, and can they be reconstructed?

> Reconstruct the missing DNA in the wombat genome

**Expect**, for the polar bear: assembly `GCF_017311325.1`, scaffold
`NW_024426341.1`, ten gaps of 5–10 bp, `PARTIALLY_COMPLETED`, roughly one filled
(`ACCTTAC`, from *Ursus arctos*). About four minutes.

## Reconstruction → Genome → Reconstruction

The reverse chain, and worth testing separately: these phrasings reach this agent
*first*, with a species but no sequence. It cannot pick a target on its own, so it
returns `NEEDS_AGENT` for the Genome Agent, carrying a `reconstruction_request`
key so the orchestrator's loop guard sees the context change.

> The polar bear genome assembly is incomplete — can you fill in the gaps?

> Repair the gaps in polar bear scaffold NW_024426341.1

## Controls — these must NOT reach this agent

> Show me the genome of the tiger

> How big is the polar bear genome?

Chromosome-level assemblies have no unresolved regions to escalate, so the Genome
Agent answers alone.

> Reconstruct the woolly mammoth genome

No assembly exists. The Genome Agent fails with "no genome has been assembled and
deposited for it" — a data gap, not a spelling mistake.

## Which species trigger a handoff

| Species | Assembly | Level | Result |
|---|---|---|---|
| *Ursus maritimus* (polar bear) | `GCF_017311325.1` | Scaffold | Handoff, 10 gaps, ~1 resolved |
| *Vombatus ursinus* (wombat) | `GCF_900497805.2` | Scaffold | Handoff with real gaps |
| *Panthera tigris*, *Mus musculus*, *Loxodonta africana*, *Gorilla gorilla*, *Ambystoma mexicanum* | — | Chromosome | No handoff |
| *Mammuthus primigenius* | none | — | Genome Agent fails |
| *Okapia johnstoni*, *Chrysocyon brachyurus*, *Tapirus indicus* | `GCA_…` | Scaffold / Contig | Escalates with **no gaps** — see below |

Most well-known species are chromosome-level now, so the polar bear is the
dependable trigger. GenBank-only (`GCA_`) assemblies escalate but carry an empty
`target_gaps`: nuccore indexes its `[Assembly]` field for RefSeq only, and the WGS
fallback reaches just the project's master record. See
`genome_agent/docs/genome_agent_integration.md` §9a.

## Reading the result

A healthy run shows, in the `run_agents` terminal:

```
[find_target_gaps] 10 of 30 gaps in NW_024426341.1 (shortest 5 bp)
```

and in the orchestrator terminal:

```
[Genome] needs_agent -> 'Reconstruct the selected unresolved regions.'
[Reconstruction] completed -> {...}
[Responder] synthesized final answer from 2 findings
```

Two signals that something is stale rather than broken:

- **Gaps of 537–2,784 bp.** The Genome Agent now sends the ten *shortest* gaps,
  because long runs have no gap-spanning homolog and exceed what Evo 2 covers.
  Long gaps mean it is serving pre-fix code.
- **"Genome asked for help again without receiving anything new."** The
  escalation-loop breaker in `orchestrator_adapter.OrchestratorGenomeAgent.run`
  did not load.

## Expectations

Roughly one gap in ten resolves for the polar bear, and that is a data limit
rather than a defect. NCBI's searchable nucleotide collection holds 236,298
Ursidae mRNA transcripts and **zero** genomic records above 100 kb for the entire
family — the bear genomes live in the WGS division, outside `core_nt` and `nt`.
A spliced transcript cannot align across a genomic gap, so the homology search
returns hits on one flank or the other and none crossing the middle. That is what
`INSUFFICIENT_GAP_SPANNING_HOMOLOGS` reports, and refusing to fill is correct
behaviour, not a failure.

The planner is an LLM, so routing is not deterministic. "Reconstruct the
unresolved regions of X" measured stable across repeated runs; "repair the gaps in
X" pulls toward reaching this agent first. Rephrase toward the former if a prompt
lands on the wrong agent.
