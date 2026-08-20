"""Contracts around the external phylogenetic tools.

Offline: no binary is executed, no network is touched. Only the wiring is
under test — binary discovery, taxon-identity round-tripping, the confidence
contract and the tree-URL policy.

The real-binary end-to-end runs live outside the repository; see the audit
report. What is pinned here is behaviour that must not silently regress.
"""

from __future__ import annotations

import re

import pytest

from backend.agents.evolution_agent.schema import (
    AgentRequest,
    AgentStatus,
    PhylogeneticResult,
)
from backend.agents.evolution_agent.tools import binaries, taxon_ids
from backend.agents.evolution_agent.tools import iqtree as iq
from backend.agents.evolution_agent.tools import mafft as mf
from backend.agents.evolution_agent.workers.phylogenetic_tree import (
    worker as worker_mod,
)


# ===========================================================================
# 1. BINARY DISCOVERY
# ===========================================================================

def test_explicit_env_path_wins(monkeypatch, tmp_path) -> None:
    exe = tmp_path / "mafft.bat"
    exe.write_text("")
    monkeypatch.setenv("MAFFT_BINARY", str(exe))
    assert binaries.resolve_binary(
        "MAFFT_BINARY", ("mafft.bat",), None) == str(exe)


def test_explicit_env_command_is_resolved_on_path(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("IQTREE_BINARY", "some-command")
    monkeypatch.setattr(binaries.shutil, "which",
                        lambda c: r"C:\bin\some-command.exe" if c == "some-command" else None)
    assert binaries.resolve_binary(
        "IQTREE_BINARY", ("iqtree3",), None) == r"C:\bin\some-command.exe"


def test_explicit_but_unresolvable_env_does_not_fall_through(
    monkeypatch, tmp_path,
) -> None:
    """A wrong explicit setting is a configuration error, not a fallback."""
    legacy = tmp_path / "legacy.exe"
    legacy.write_text("")
    monkeypatch.setenv("MAFFT_BINARY", str(tmp_path / "nope.bat"))
    monkeypatch.setattr(binaries.shutil, "which", lambda c: None)
    assert binaries.resolve_binary(
        "MAFFT_BINARY", ("mafft.bat",), str(legacy)) is None


def test_path_candidates_are_tried_in_order(monkeypatch) -> None:
    monkeypatch.delenv("IQTREE_BINARY", raising=False)
    seen: list[str] = []

    def fake_which(cmd):
        seen.append(cmd)
        return r"C:\bin\iqtree2.exe" if cmd == "iqtree2.exe" else None

    monkeypatch.setattr(binaries.shutil, "which", fake_which)
    got = binaries.resolve_binary(
        "IQTREE_BINARY", binaries.IQTREE_CANDIDATES, None)
    assert got == r"C:\bin\iqtree2.exe"
    assert seen[:3] == ["iqtree3.exe", "iqtree3", "iqtree2.exe"]


def test_legacy_bundled_path_is_the_last_resort(monkeypatch, tmp_path) -> None:
    legacy = tmp_path / "bundled.bat"
    legacy.write_text("")
    monkeypatch.delenv("MAFFT_BINARY", raising=False)
    monkeypatch.setattr(binaries.shutil, "which", lambda c: None)
    assert binaries.resolve_binary(
        "MAFFT_BINARY", ("mafft.bat",), str(legacy)) == str(legacy)


def test_nothing_found_returns_none(monkeypatch) -> None:
    monkeypatch.delenv("MAFFT_BINARY", raising=False)
    monkeypatch.setattr(binaries.shutil, "which", lambda c: None)
    assert binaries.resolve_binary("MAFFT_BINARY", ("mafft",), None) is None


def test_candidate_lists_cover_every_required_name() -> None:
    assert binaries.MAFFT_CANDIDATES == ("mafft.bat", "mafft")
    assert binaries.IQTREE_CANDIDATES == (
        "iqtree3.exe", "iqtree3", "iqtree2.exe", "iqtree2", "iqtree")


def test_env_keys_are_documented_without_a_local_path() -> None:
    from pathlib import Path
    example = (Path(__file__).resolve().parent.parent / ".env.example"
               ).read_text(encoding="utf-8")
    assert "MAFFT_BINARY=" in example
    assert "IQTREE_BINARY=" in example
    for line in example.splitlines():
        if line.startswith(("MAFFT_BINARY=", "IQTREE_BINARY=")):
            assert line.split("=", 1)[1].strip() == "", "a real path leaked"


def test_missing_binary_error_names_everything_that_was_tried(monkeypatch) -> None:
    monkeypatch.delenv("IQTREE_BINARY", raising=False)
    monkeypatch.setattr(binaries.shutil, "which", lambda c: None)
    monkeypatch.setattr(iq, "_LOCAL_IQTREE", r"C:\nowhere\iqtree2.exe")
    with pytest.raises(iq.IQTreeError) as err:
        iq.build_tree(">a\nMK\n>b\nMV\n>c\nMW\n")
    msg = str(err.value)
    assert "IQTREE_BINARY" in msg and "iqtree3.exe" in msg


# ===========================================================================
# 2. TAXON IDENTITY
# ===========================================================================

def test_safe_ids_are_whitespace_free_and_sequential() -> None:
    name_to_id, id_to_name = taxon_ids.make_mapping(
        ["Homo sapiens", "Pan troglodytes"])
    assert list(name_to_id.values()) == ["taxon_0001", "taxon_0002"]
    assert all(" " not in i for i in id_to_name)
    assert id_to_name["taxon_0001"] == "Homo sapiens"


def test_same_genus_different_species_never_collide() -> None:
    """The exact failure this correction exists for."""
    names = ["Homo sapiens", "Homo erectus", "Homo neanderthalensis"]
    name_to_id, id_to_name = taxon_ids.make_mapping(names)

    assert len(set(name_to_id.values())) == 3, "two taxa share an id"
    assert len(id_to_name) == 3

    newick = "((taxon_0001:0.1,taxon_0002:0.2)95:0.3,taxon_0003:0.4);"
    restored = taxon_ids.restore_newick(newick, id_to_name)
    for n in names:
        assert f"'{n}'" in restored, f"{n} lost in the Newick"
    assert "Homo:" not in restored, "a genus-only label survived"


def test_truncation_at_the_first_space_can_no_longer_happen() -> None:
    """Old behaviour: '>Homo sapiens' -> tool emits 'Homo'."""
    sequences = {"Homo sapiens": "MK", "Homo erectus": "MV"}
    name_to_id, _ = taxon_ids.make_mapping(list(sequences))
    safe = taxon_ids.to_safe_sequences(sequences, name_to_id)
    fasta = mf._dict_to_fasta(safe)

    headers = [l[1:] for l in fasta.splitlines() if l.startswith(">")]
    assert all(" " not in h for h in headers)
    # each header survives the tools' "first token" rule intact
    assert [h.split()[0] for h in headers] == headers
    assert len(set(headers)) == 2


def test_newick_labels_with_spaces_are_quoted() -> None:
    _, id_to_name = taxon_ids.make_mapping(["Homo sapiens"])
    out = taxon_ids.restore_newick("(taxon_0001:0.1);", id_to_name)
    assert out == "('Homo sapiens':0.1);"


def test_newick_quoting_escapes_embedded_single_quotes() -> None:
    assert taxon_ids.quote_newick_label("O'hara species") == "'O''hara species'"


def test_label_without_special_characters_is_not_quoted() -> None:
    assert taxon_ids.quote_newick_label("Ecoli") == "Ecoli"


def test_alignment_names_are_restored() -> None:
    _, id_to_name = taxon_ids.make_mapping(["Homo sapiens", "Homo erectus"])
    aligned = ">taxon_0001\nMK-\n>taxon_0002\nMV-\n"
    restored = taxon_ids.restore_fasta(aligned, id_to_name)
    assert ">Homo sapiens" in restored
    assert ">Homo erectus" in restored
    assert "taxon_0001" not in restored


def test_parse_fasta_takes_the_first_token_as_identifier() -> None:
    parsed = taxon_ids.parse_fasta(">taxon_0001 extra words\nMK\n")
    assert list(parsed) == ["taxon_0001"]


def test_worker_sends_safe_ids_and_returns_full_names(monkeypatch) -> None:
    """End-to-end through the worker with both tools stubbed."""
    seen: dict = {}

    def fake_mafft(sequences, *a, **k):
        seen["mafft_keys"] = list(sequences)
        return "".join(f">{k}\nMTNIRK-\n" for k in sequences)

    def fake_iqtree(alignment, model="MFP", bootstrap=1000, **k):
        seen["iqtree_keys"] = list(alignment)
        ids = list(alignment)
        nk = f"(({ids[0]}:0.1,{ids[1]}:0.1)95:0.2,{ids[2]}:0.3);"
        return iq.PhyloResult(nk, "LG+G4", {"node_0": 95}, {"node_0": 0.95},
                              0.95, ufboot_run=True)

    monkeypatch.setattr(worker_mod, "mafft_align", fake_mafft)
    monkeypatch.setattr(worker_mod, "iqtree_build", fake_iqtree)

    r = worker_mod.PhylogeneticTreeWorker().run(AgentRequest(
        instruction="tree", context={},
        species_list=["homo sapiens", "pan troglodytes", "mus musculus"]))

    assert r.status is AgentStatus.COMPLETED
    assert all(re.fullmatch(r"taxon_\d{4}", k) for k in seen["mafft_keys"])
    assert all(re.fullmatch(r"taxon_\d{4}", k) for k in seen["iqtree_keys"])

    nk = r.output.newick_tree
    assert "taxon_" not in nk, "a safe id leaked into the public Newick"
    for s in ("homo sapiens", "pan troglodytes", "mus musculus"):
        assert f"'{s}'" in nk
    assert ">homo sapiens" in r.output.aligned_fasta


def test_worker_fails_loudly_if_a_taxon_is_lost(monkeypatch) -> None:
    monkeypatch.setattr(worker_mod, "mafft_align",
                        lambda seqs, *a, **k: ">taxon_0001\nMK\n")
    monkeypatch.setattr(worker_mod, "iqtree_build",
                        lambda **k: pytest.fail("IQ-TREE must not be reached"))

    r = worker_mod.PhylogeneticTreeWorker().run(AgentRequest(
        instruction="tree", context={},
        species_list=["homo sapiens", "pan troglodytes", "mus musculus"]))
    assert r.status is AgentStatus.FAILED
    assert "did not return every input taxon" in str(r.output)


# ===========================================================================
# 3-5. CONFIDENCE CONTRACT AND UFBOOT
# ===========================================================================

def test_phylo_result_without_bootstrap_reports_nothing(monkeypatch) -> None:
    r = iq.PhyloResult("(a,b,c);", "LG", {}, {}, None, ufboot_run=False)
    assert r.bootstrap_support == {}
    assert r.confidence_values == {}
    assert r.overall_confidence is None
    assert r.ufboot_run is False


def test_no_fabricated_default_confidence_remains_in_the_source() -> None:
    from pathlib import Path
    src = (Path(iq.__file__)).read_text(encoding="utf-8")
    assert "else 0.95" not in src
    assert "0.95" not in src, "a hardcoded confidence default is back"


def test_confidence_is_computed_only_from_real_supports() -> None:
    support = {"node_0": 90, "node_1": 80}
    expected = round((0.90 + 0.80) / 2, 4)
    conf = {k: v / 100.0 for k, v in support.items()}
    assert round(sum(conf.values()) / len(conf), 4) == expected


def test_bootstrap_keys_are_nodes_not_species() -> None:
    parsed = iq._parse_bootstrap("((a:0.1,b:0.1)95:0.2,c:0.3);")
    assert list(parsed) == ["node_0"]
    assert all(k.startswith("node_") for k in parsed)
    assert "a" not in parsed and "b" not in parsed


def _flat(text: str | None) -> str:
    """Collapse whitespace so a wrapped docstring still matches."""
    return " ".join((text or "").split())


def test_docstrings_state_the_node_based_contract() -> None:
    module_doc = _flat(iq.__doc__)
    result_doc = _flat(iq.PhyloResult.__doc__)
    worker_doc = _flat(worker_mod.__doc__)

    assert "keyed by INTERNAL NODE" in module_doc
    assert "not by species" in module_doc
    assert "never of a single leaf" in module_doc
    assert "never per-species confidence" in result_doc
    assert "per-INTERNAL-NODE" in worker_doc


def test_worker_docstring_no_longer_claims_per_leaf_confidence() -> None:
    """The old text promised '{species: float}', which was never true."""
    worker_doc = _flat(worker_mod.__doc__)
    assert "per-leaf confidence" not in worker_doc
    assert "{species: float}" not in worker_doc


def test_ufboot_skipped_below_four_taxa_is_reported(monkeypatch) -> None:
    captured: dict = {}

    def fake_iqtree(alignment, model="MFP", bootstrap=1000, **k):
        captured["bootstrap"] = bootstrap
        ids = list(alignment)
        nk = f"({ids[0]}:0.1,{ids[1]}:0.1,{ids[2]}:0.2);"
        return iq.PhyloResult(nk, "LG+G4", {}, {}, None, ufboot_run=False)

    monkeypatch.setattr(worker_mod, "mafft_align",
                        lambda s, *a, **k: "".join(f">{i}\nMTNIRK\n" for i in s))
    monkeypatch.setattr(worker_mod, "iqtree_build", fake_iqtree)

    r = worker_mod.PhylogeneticTreeWorker().run(AgentRequest(
        instruction="tree", context={},
        species_list=["homo sapiens", "pan troglodytes", "mus musculus"]))

    assert captured["bootstrap"] == 0, "UFBoot minimum-taxa rule changed"
    assert r.status is AgentStatus.COMPLETED
    assert "ufboot_not_run" in r.warnings
    assert r.output.overall_confidence is None
    assert r.output.bootstrap_support == {}
    assert r.output.confidence_values == {}
    assert r.confidence is None


def test_ufboot_runs_from_four_taxa_and_no_warning_is_raised(monkeypatch) -> None:
    captured: dict = {}

    def fake_iqtree(alignment, model="MFP", bootstrap=1000, **k):
        captured["bootstrap"] = bootstrap
        ids = list(alignment)
        nk = f"(({ids[0]}:0.1,{ids[1]}:0.1)95:0.2,{ids[2]}:0.3,{ids[3]}:0.4);"
        return iq.PhyloResult(nk, "LG+G4", {"node_0": 95}, {"node_0": 0.95},
                              0.95, ufboot_run=True)

    monkeypatch.setattr(worker_mod, "mafft_align",
                        lambda s, *a, **k: "".join(f">{i}\nMTNIRK\n" for i in s))
    monkeypatch.setattr(worker_mod, "iqtree_build", fake_iqtree)

    r = worker_mod.PhylogeneticTreeWorker().run(AgentRequest(
        instruction="tree", context={},
        species_list=["homo sapiens", "pan troglodytes", "mus musculus",
                      "gallus gallus"]))

    assert captured["bootstrap"] == 1000
    assert "ufboot_not_run" not in r.warnings
    assert r.output.overall_confidence == 0.95


# ===========================================================================
# 7. TREE URL
# ===========================================================================

def test_worker_never_fabricates_a_tree_url(monkeypatch) -> None:
    monkeypatch.setattr(worker_mod, "mafft_align",
                        lambda s, *a, **k: "".join(f">{i}\nMTNIRK\n" for i in s))
    monkeypatch.setattr(
        worker_mod, "iqtree_build",
        lambda alignment, **k: iq.PhyloResult(
            "(" + ",".join(f"{i}:0.1" for i in alignment) + ");",
            "LG+G4", {}, {}, None, ufboot_run=False))

    r = worker_mod.PhylogeneticTreeWorker().run(AgentRequest(
        instruction="tree", context={},
        species_list=["homo sapiens", "pan troglodytes", "mus musculus"]))

    assert r.tree_url is None
    assert r.output.tree_url is None


def test_no_placeholder_tree_host_remains_in_the_worker() -> None:
    from pathlib import Path
    src = Path(worker_mod.__file__).read_text(encoding="utf-8")
    assert "evolution.umbrella.local/tree" not in src
