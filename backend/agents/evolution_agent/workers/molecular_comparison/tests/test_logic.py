import torch
from ..logic import MolecularComparisonAgent
from ....schema import AgentRequest, AgentStatus

def fake_fetch(species: str, gene_candidates: list[str]) -> str:
    return "MTNIRKSHPLFKIINHSFIDLPAPSNISSWWNFGSLLGACLILQITTGLFLAMHYTSDTT"

def fake_embed(sequences: dict) -> dict:
    # deterministic fake vectors, one per species
    return {sp: torch.ones(480) * (i + 1) for i, sp in enumerate(sequences)}

def test_completed_with_two_species():
    agent = MolecularComparisonAgent(fetch_fn=fake_fetch, embed_fn=fake_embed)
    req = AgentRequest(instruction="test", species_list=["homo sapiens", "pan troglodytes"])
    result = agent.run(req)
    assert result.status == AgentStatus.COMPLETED
    assert result.output.species_list == ["homo sapiens", "pan troglodytes"]

def test_fails_with_no_species():
    agent = MolecularComparisonAgent(fetch_fn=fake_fetch, embed_fn=fake_embed)
    req = AgentRequest(instruction="test", species_list=[])
    result = agent.run(req)
    assert result.status == AgentStatus.FAILED

def test_fails_with_one_species():
    agent = MolecularComparisonAgent(fetch_fn=fake_fetch, embed_fn=fake_embed)
    req = AgentRequest(instruction="test", species_list=["homo sapiens"])
    result = agent.run(req)
    assert result.status == AgentStatus.FAILED

def test_fetch_failure_returns_failed():
    def broken_fetch(species, genes):
        raise ValueError("no sequence found")
    agent = MolecularComparisonAgent(fetch_fn=broken_fetch, embed_fn=fake_embed)
    req = AgentRequest(instruction="test", species_list=["a", "b"])
    result = agent.run(req)
    assert result.status == AgentStatus.FAILED

def test_species_groups_and_similarity_network_shape():
    agent = MolecularComparisonAgent(fetch_fn=fake_fetch, embed_fn=fake_embed)
    req = AgentRequest(
        instruction="test",
        species_list=["homo sapiens", "pan troglodytes", "mus musculus"],
    )
    result = agent.run(req)
    assert result.status == AgentStatus.COMPLETED

    mc_result = result.output
    assert len(mc_result.similarity_scores) == 3  # 3 choose 2 pairs

    groups = mc_result.species_groups
    assert isinstance(groups, list)
    assert sum(len(g.species) for g in groups) == 3
    for group in groups:
        assert isinstance(group.group_id, int)
        assert isinstance(group.mean_score, float)

    network = mc_result.similarity_network
    assert set(network.keys()) >= {"nodes", "edges"}
    assert {n["id"] for n in network["nodes"]} == {
        "homo sapiens", "pan troglodytes", "mus musculus",
    }
    assert len(network["edges"]) == 3
    for edge in network["edges"]:
        assert "source" in edge and "target" in edge and "score" in edge