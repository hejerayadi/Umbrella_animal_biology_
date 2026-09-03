# Prompt test bed

A scratchpad for trying prompts against every agent on the platform and writing
down what actually happened. Each section has starter prompts and an empty
results table — fill the table in as you test, so the next person does not have
to re-run the same prompt to find out it fails.

Two agents already have their own measured prompt files. Keep those as the
source of truth for their numbers; this file is for exploring:

- [evolution_agent/TESTING_PROMPTS.md](backend/agents/evolution_agent/TESTING_PROMPTS.md)
- [reconstruction_agent/TESTING_PROMPTS.md](backend/agents/reconstruction_agent/TESTING_PROMPTS.md)

---

## How to run a prompt

Start everything from the repository root:

```
python -m backend.run_agents --reload
```

Without `--reload` an agent serves whatever was on disk when it started.

### Through the orchestrator (what the frontend does)

`POST http://localhost:8000/api/v1/workspace/chat` — requires a logged-in
session cookie + CSRF header, so the easiest route is the frontend chat box.

```jsonc
{ "query": "<the prompt>", "context": {} }
```

Response: `answer`, `execution_history` (this is the routing trace — read it),
`context`, `image_url`.

### Straight at one agent (skips routing, fastest loop)

`POST http://localhost:<port>/execute`, no auth:

```jsonc
{ "instruction": "<the prompt>", "context": {} }
```

| Agent | Port | Folder |
|---|---|---|
| Genome | 8001 | [genome_agent](backend/agents/genome_agent/) |
| Evolution | 8002 | [evolution_agent](backend/agents/evolution_agent/) |
| Biodiversity | 8003 | [biodiversity_agent](backend/agents/biodiversity_agent/) |
| Literature | 8004 | [Literature_Agent](backend/agents/Literature_Agent/) |
| Multimodal | 8005 | [multimodal_recognition_agent](backend/agents/multimodal_recognition_agent/) |
| Reconstruction | 8006 | [reconstruction_agent](backend/agents/reconstruction_agent/) |
| Trait | 8007 | [trait_discovery_agent](backend/agents/trait_discovery_agent/) |
| Protein | 8008 | [Protein_visualization](backend/agents/Protein_visualization/) |
| ImageGeneration | 8009 | [image_generation_agent](backend/agents/image_generation_agent/) |

Ports and names come from [registry.py](backend/registry.py); every URL can be
overridden with `<AGENTNAME>_AGENT_URL`.

### One-liner

```powershell
curl.exe -s -X POST http://localhost:8002/execute -H "Content-Type: application/json" -d '{\"instruction\":\"How similar are human and chimp?\",\"context\":{}}'
```

### What to record

`status` is one of `completed`, `continue`, `needs_agent`, `failed`. A `continue`
is the agent asking a question, not a failure. A `needs_agent` names the agent it
wants next — that is the interesting part of a routing test.

---

## Template — copy this per prompt you try

```
### <short name>

> <the prompt, verbatim>

- Entry: orchestrator | direct :<port>
- Routed to:
- Status:
- Time:
- Output looked like:
- Verdict: works / wrong route / wrong answer / crashed
```

---

## 1. Genome Agent — :8001

Species → assembly → genome size, chromosome count, assembly level, gene table.
Live NCBI, so answers change and calls are slow.

Starter prompts:

> What is the genome size of the tiger?

> How many chromosomes does the Asian elephant have?

> Show gene annotations for the polar bear.

> Show me the genome of the woolly mammoth.

> Compare the genome size of the blue whale and the house mouse.

Probes worth trying:
- a species with no assembly at all (does it say so, or invent one?)
- a common name that maps to several species ("bear")
- a bare accession, e.g. `GCF_017311325.1`, with no species name

| Prompt | Route | Status | Time | Notes |
|---|---|---|---|---|
|  |  |  |  |  |

## 2. Evolution Agent — :8002

Molecular comparison from protein embeddings, and MAFFT + IQ-TREE phylogeny.

**Only five species resolve** (offline catalogue in
`orchestrator/services/species_resolver.py`): *Homo sapiens*, *Pan troglodytes*,
*Mus musculus*, *Gallus gallus*, *Danio rerio*, plus their common aliases.
Anything else is rejected before a worker runs. Deterministic guards: comparison
needs ≥2 species, a tree needs ≥3, and both branches run only if the prompt asks
for both. The first molecular call downloads ~150 MB of ESM-2.

Starter prompts:

> How similar are human, chimp and mouse at the molecular level?

> Build a phylogenetic tree for humans, the common chimpanzee, house mouse, red junglefowl and zebra danio.

> Give me both a similarity network and a phylogenetic tree for human, chimp, mouse and chicken.

> Tell me about evolution.  ← should ask which species, 1 LLM call

> Build a phylogenetic tree for human and chimp.  ← should refuse: needs 3

> Compare the axolotl and the platypus.  ← unsupported species, check the wording of the refusal

Full measured numbers: [TESTING_PROMPTS.md](backend/agents/evolution_agent/TESTING_PROMPTS.md).

| Prompt | Route | Status | Time | Notes |
|---|---|---|---|---|
|  |  |  |  |  |

## 3. Biodiversity Agent — :8003

A domain orchestrator over four workers: species distribution (GBIF), habitat
visualization (GBIF+IUCN+WorldClim), hotspots (GBIF+DBSCAN), migration
(Movebank+GBIF). Each prompt below targets one worker — check the trace picked
the one you meant.

Starter prompts:

> Map the global distribution of the snow leopard.

> Show me the habitat of the African elephant and its conservation status.

> Where are the biodiversity hotspots in the Mediterranean basin?

> Analyse the seasonal migration routes of the Arctic tern.

> Which threatened bird species occur in Tunisia?

Probes:
- a region with almost no GBIF records (empty result vs. error?)
- two skills in one prompt: "map the distribution of the snow leopard and find the hotspots in its range"
- a gene question, which should escalate `NEEDS_AGENT` → Trait Discovery

| Prompt | Worker | Status | Time | Notes |
|---|---|---|---|---|
|  |  |  |  |  |

## 4. Literature Agent — :8004

Retrieval + summarization + scientific writing (abstracts, intros, related work,
citations, venue suggestions).

Starter prompts:

> What does the published evidence say about CRISPR off-target effects in mammals?

> Find recent papers on polar bear population genomics and summarise the consensus.

> Draft an abstract for a study on gene-trait association in the woolly mammoth.

> Write a related-work section on phylogenetic reconstruction from protein embeddings.

> Give me the DOIs for the three most cited papers on Movebank migration data.

> Which journal should I submit an animal-genomics methods paper to?

Probes:
- a topic where the literature disagrees — does it flag the contradiction?
- ask for a citation that does not exist; check it does not fabricate a DOI

There are interactive manual harnesses in this agent's folder
(`manual_test_interactive*.py`) if you want to drive the writing paths directly.

| Prompt | Sub-skill | Status | Time | Notes |
|---|---|---|---|---|
|  |  |  |  |  |

## 5. Multimodal Recognition Agent — :8005

**Needs both an image and a non-empty instruction.** Image-only and text-only
requests are invalid. The image goes in `context.recognition_image` as a
`data_url` — no other context key is scanned, and remote URLs and file paths are
never fetched. JPEG/PNG/WEBP.

Through the frontend: upload first (`POST /api/v1/workspace/uploads`), then send
the returned `image_id` under the `context_key` it hands back.

```jsonc
{
  "instruction": "Identify this animal and explain the result.",
  "context": {
    "recognition_image": { "data_url": "data:image/jpeg;base64,<BASE64>", "filename": "observation.jpg" }
  }
}
```

Starter prompts (each paired with a photo):

> Identify this animal and explain the result.

> What species is this? Give me the top 5 candidates with confidence.

> Is this a red fox or a fennec fox?

> Identify this animal, then tell me its genome size.  ← should chain to Genome

Probes — the confidence gate returns `identified` / `uncertain` / `not_identified`:
- a clear, well-lit single animal → `identified`
- a blurry or distant animal → `uncertain`
- a photo with no animal in it → `not_identified`
- text with no image at all → should be rejected, not guessed

Fixtures live in [fixtures](backend/agents/multimodal_recognition_agent/fixtures/).

| Prompt + image | Gate | Top-1 | Status | Notes |
|---|---|---|---|---|
|  |  |  |  |  |

## 6. Reconstruction Agent — :8006

Fills unresolved regions (N-runs) in incomplete assemblies. Will not pick a
target on its own — it needs a sequence or an accession, otherwise it returns
`NEEDS_AGENT` for the Genome Agent. Runs take minutes, not seconds.

Starter prompts:

> Reconstruct the unresolved regions of the polar bear genome

> Fill the gaps in the polar bear genome

> The polar bear genome assembly is incomplete — can you fill in the gaps?  ← enters here first, bounces to Genome

> Reconstruct the missing bases in this sequence:
> TTGTGTATAGAGTTGCAATCAAATTTGTATATTGACTTTATATCCTGCTGNNNNNGAATTCATTAGTTATAACAGATTTTTTCTTTTAAACAATTCATTGAATTA

The pasted-sequence form is the fastest path — no Genome Agent in the chain.
Note this agent is time-starved: a BLAST step can exceed the slice budget and
come back empty, which is a budget problem and not a misconfiguration.

Full measured numbers: [TESTING_PROMPTS.md](backend/agents/reconstruction_agent/TESTING_PROMPTS.md).

| Prompt | Route | Status | Time | Notes |
|---|---|---|---|---|
|  |  |  |  |  |

## 7. Trait Discovery Agent — :8007

Traits ↔ genes, via Gene Ontology, UniProt, pathways. Covers morphology,
physiology, behaviour, life history, and the genes behind them.

Starter prompts:

> Which genes are associated with fur colour in the arctic fox?

> Why does the woolly mammoth have thick fur?

> What traits characterise the naked mole rat, and which genes control them?

> Compare the lifespan-related genes of the bowhead whale and the house mouse.

> What is the Gene Ontology evidence linking MC1R to coat colour?

Probes:
- a trait no ontology covers ("intelligence") — does it hedge or invent?
- a species with thin UniProt coverage
- a gene symbol with no trait in the prompt

| Prompt | Status | Genes returned | Evidence source | Notes |
|---|---|---|---|---|
|  |  |  |  |  |

## 8. 3D Protein Structure Agent — :8008

Gene/accession → canonical protein → best 3D structure (experimental or
predicted) → annotations mapped onto residues → viewer-ready scene. This is the
3D agent; the Image Generation agent is 2D and predicts nothing.

Starter prompts:

> Show me the 3D structure of human insulin.

> What does the protein encoded by TP53 look like, and where are its domains?

> Visualise P69905 and highlight the haem-binding site.

> Where does the mutation at residue 280 of TP53 sit in the folded structure?

> Is there an experimental structure for this protein, or only a predicted model?

Probes:
- a gene with no PDB entry (falls back to a predicted model?)
- an accession that does not exist
- a residue number past the end of the sequence

| Prompt | Identity resolved | Structure source | Status | Notes |
|---|---|---|---|---|
|  |  |  |  |  |

## 9. Image Generation Agent — :8009

2D scientific illustration via FLUX.2-pro, built from trait context. Needs a
subject — a species or a protein identifier. When traits are missing it routes
to Trait Discovery first, so a bare "draw X" is also a routing test.

Starter prompts:

> Draw a woolly mammoth

> Create a 2D illustration of woolly mammoth morphological traits

> Draw the structural traits of human insulin

> Generate a scientific image for the species described in context

> Illustrate the wing structure of the Arctic tern

Probes:
- no subject at all ("draw something scientific") — should ask, not guess
- a 3D-shaped request ("predict the fold of TP53") — should defer to the Protein agent
- run the same prompt twice and compare, since generation is not deterministic

| Prompt | Traits fetched from | Status | Time | Notes |
|---|---|---|---|---|
|  |  |  |  |  |

---

## 10. Cross-agent chains

The interesting failures live here. Run these through the orchestrator only, and
read `execution_history` rather than the final prose.

> Are there unresolved regions in the polar bear genome, and can they be reconstructed?
— Genome → Reconstruction

> Identify this animal and draw an illustration of its main traits.
— Multimodal → Trait → ImageGeneration (needs an image attached)

> What genes give the polar bear its white coat, and show me where they sit in the protein structure?
— Trait → Protein

> Map the snow leopard's range and summarise what the literature says about its decline.
— Biodiversity → Literature

> Build a phylogenetic tree for human, chimp and mouse and cite the papers behind the method.
— Evolution → Literature

Watch for: loop guards firing, an agent being called twice with the same
context, context keys not carrying over between hops, and the final answer
dropping a result an earlier hop produced.

| Chain | Hops taken | Hops expected | Status | Notes |
|---|---|---|---|---|
|  |  |  |  |  |

---

## 11. Routing-only prompts

Deliberately ambiguous. The point is which agent gets picked, not the answer.

> Tell me about the polar bear.

> What do we know about mammoths?

> Show me something about the tiger.

> Analyse this species.

> Genes.

> Draw the genome of the elephant.  ← nonsense on purpose: which agent claims it?

| Prompt | Agent picked | Reasonable? | Notes |
|---|---|---|---|
|  |  |  |  |

---

## Log

Newest first. One line per session: date, what you were testing, what broke.

| Date | Focus | Finding |
|---|---|---|
| 2026-09-02 | file created | — |
