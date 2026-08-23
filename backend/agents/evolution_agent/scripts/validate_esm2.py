"""Standalone validation: confirm ESM-2 loads and check embedding shape.

Not wired into the pipeline yet — just checking model behavior in isolation
before touching mock.py.
"""

import torch
import esm
import requests

SPECIES_TAXON_IDS = {
    "homo_sapiens": 9606,
    "pan_troglodytes": 9598,
    "mus_musculus": 10090,
    "gallus_gallus": 9031,
    "danio_rerio": 7955,
}

def fetch_uniprot_sequence(gene_names: list[str], taxon_id: int) -> str:
    for gene_name in gene_names:
        url = "https://rest.uniprot.org/uniprotkb/search"
        params = {
            "query": f"gene:{gene_name} AND organism_id:{taxon_id} AND reviewed:true",
            "format": "fasta",
            "size": 1,
        }
        resp = requests.get(url, params=params, timeout=30)
        resp.raise_for_status()
        fasta = resp.text.strip()
        if fasta:
            lines = fasta.split("\n")
            return "".join(line for line in lines if not line.startswith(">"))
    raise ValueError(f"No reviewed UniProt hit for gene in {gene_names}, taxon={taxon_id}")

data = []
for species, taxon_id in SPECIES_TAXON_IDS.items():
    seq = fetch_uniprot_sequence(["CYCS", "CYC"], taxon_id)
    data.append((species, seq))
    print(f"{species}: fetched {len(seq)} aa")

# Load esm2_t12_35M_UR50D — the model the team locked in.
model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
model.eval()
batch_converter = alphabet.get_batch_converter()

batch_labels, batch_strs, batch_tokens = batch_converter(data)

with torch.no_grad():
    results = model(batch_tokens, repr_layers=[12], return_contacts=False)

token_representations = results["representations"][12]

sequence_embeddings = []
for i, (_, seq) in enumerate(data):
    tokens_len = len(seq) + 2
    pooled = token_representations[i, 1:tokens_len - 1].mean(0)
    sequence_embeddings.append(pooled)

# Raw cosine similarity matrix.
print("\nRaw cosine similarity matrix:")
print(f"{'':>16}", *[f"{d[0]:>16}" for d in data])
for i in range(len(data)):
    row = [
        torch.nn.functional.cosine_similarity(
            sequence_embeddings[i].unsqueeze(0), sequence_embeddings[j].unsqueeze(0)
        ).item()
        for j in range(len(data))
    ]
    print(f"{data[i][0]:>16}", *[f"{v:>16.4f}" for v in row])

# Mean-center to counteract anisotropy before computing cosine similarity.
embedding_matrix = torch.stack(sequence_embeddings)          # [N, 480]
centered = embedding_matrix - embedding_matrix.mean(dim=0)   # subtract batch mean

print("\nCentered cosine similarity matrix:")
print(f"{'':>16}", *[f"{d[0]:>16}" for d in data])
for i in range(len(data)):
    row = [
        torch.nn.functional.cosine_similarity(
            centered[i].unsqueeze(0), centered[j].unsqueeze(0)
        ).item()
        for j in range(len(data))
    ]
    print(f"{data[i][0]:>16}", *[f"{v:>16.4f}" for v in row])