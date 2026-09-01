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

## Direct entry — a pasted sequence

The card is explicit that this agent will not pick a target on its own, so the
planner routes here directly only when the message already carries the sequence
or the accession. The orchestrator's extractor finds a pasted run of nucleotides
by regex (`extractor._find_sequence`, 20 bases minimum) and seeds it as
`facts["sequence"]`.

This is the fastest way to exercise the agent alone, with no Genome Agent in the
path. The sequence below is real: the 5 bp gap at 337,487–337,491 of
`NW_024426341.1` with its true flanks, lifted from
`artifacts/reconstruction_handoff_payload.json`.

> Reconstruct the missing bases in this sequence:
> TTGTGTATAGAGTTGCAATCAAATTTGTATATTGACTTTATATCCTGCTGNNNNNGAATTCATTAGTTATAACAGATTTTTTCTTTTAAACAATTCATTGAATTA

Verified: the extractor detects that sequence bare, prefixed with a request, and
split across lines under a FASTA header — 105 bases in all three, with the N-run
intact.

## Controls — these must NOT reach this agent

> Show me the genome of the tiger

The tiger is the control case, and it stays one: `GCF_018350195.1` is
chromosome-level and reports 10,100 unresolved bases in 2.4 Gb — 0.0004%, far
under the escalation floor.

Note that escalation is decided by the **assembly**, not by the question:
`get_genome_metadata_node` sets `reconstruction_need` whenever the assembly
qualifies, even when the user only asked for the genome size. Any polar bear
question therefore escalates, which is why "how big is the polar bear genome?" is
not a control.

> Reconstruct the woolly mammoth genome

No assembly exists. The Genome Agent fails with "no genome has been assembled and
deposited for it" — a data gap, not a spelling mistake.

## Which species trigger a handoff

Escalation now has two triggers, not one. An assembly qualifies on its **level**
(`Scaffold` / `Contig`, as before) **or** on the share of it that is unresolved —
`total_length - ungapped_length`, against `_MIN_GAP_FRACTION = 0.001` in
`genome_agent/workflows/nodes/genome_data_nodes.py`. Chromosome-level is no
longer a guarantee of no handoff, and the table below is measured against NCBI's
own stats rather than assumed:

| Species | Assembly | Level | Unresolved | Handoff |
|---|---|---|---|---|
| *Ursus maritimus* (polar bear) | `GCF_017311325.1` | Scaffold | 0.4040% | Yes — 10 gaps, ~1 resolved |
| *Vombatus ursinus* (wombat) | `GCF_900497805.2` | Scaffold | 0.3275% | Yes, real gaps |
| *Okapia johnstoni* | `GCA_001660835.1` | Scaffold | 4.8472% | Yes, but **no gaps** — see below |
| *Ailuropoda melanoleuca* (giant panda) | `GCF_002007445.2` | Chromosome | 1.9133% | Yes — **new**, level alone said no |
| *Mus musculus* (house mouse) | `GCF_000001635.27` | Chromosome | 2.6977% | Yes — **new** |
| *Loxodonta africana* (African elephant) | `GCF_030014295.1` | Chromosome | 0.1042% | Yes — **new**, and only just: 1.04x the floor |
| *Gorilla gorilla* | `GCF_029281585.2` | Chromosome | 0.0564% | No |
| *Ambystoma mexicanum* (axolotl) | `GCF_040938575.1` | Chromosome | 0.0014% | No |
| *Panthera tigris* (tiger) | `GCF_018350195.1` | Chromosome | 0.0004% | No — the control |
| *Mammuthus primigenius* | none | — | — | Genome Agent fails |

The polar bear remains the dependable trigger because it escalates on level and
has real short gaps. The mouse and the panda are the ones to run if you want to
see the gap-share trigger fire — neither was reachable before. The African
elephant sits just over the floor, so it is the case to watch if the threshold is
ever retuned. GenBank-only (`GCA_`) assemblies escalate but carry an empty
`target_gaps`: nuccore indexes its `[Assembly]` field for RefSeq only, and the WGS
fallback reaches just the project's master record. See
`genome_agent/docs/genome_agent_integration.md` §9a.

## Reading the result

A healthy run shows, in the `run_agents` terminal:

```
[get_genome_metadata] assembly GCF_017311325.1 is incomplete (level=Scaffold, gap_bases=9414993, fraction=0.4040%) - flagging for reconstruction
[find_target_gaps] 10 of 23 gaps in NW_024426341.1 (30 found, shortest 5 bp)
```

The handoff context carries eleven keys now, not five. Alongside `target_gaps`
it reports what was filtered out to produce them - `gaps_found`,
`gaps_over_floor`, `gaps_selected`, `selection_policy` - plus
`assembly_gap_bases_bp` and `assembly_gap_fraction`, the evidence behind the
escalation. A payload with only the original five keys is stale code.

and in the orchestrator terminal:

```
[Genome] needs_agent -> 'Reconstruct the selected unresolved regions.'
[Reconstruction] completed -> {...}
[Responder] synthesized final answer from 2 findings
```

Two signals that something is stale rather than broken:

- **A five-key handoff context.** The selection counts above are missing, so the
  Reconstruction Agent is being told it received ten gaps rather than ten of
  thirty.
- **`reconstruction_sequence` in the response.** That key is now
  `reconstruction_best_fill`, and it is an object carrying `start`, `end` and
  `length_bp` rather than a bare string.
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
