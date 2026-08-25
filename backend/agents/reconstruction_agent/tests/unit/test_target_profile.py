"""The taxonomy prior behind the database choice.

The agent's central defect was a hardcoded `em_std_vrt`. EMBL divisions are
mutually exclusive, so a *vertebrate* database excludes mammals outright and no
mammal run could ever find its own homologues. Measured on the polar bear
mitogenome, same query and parameters: `em_std_vrt` returned 7 gap-crossing hits
(all fish, ~67% identity) against `em_std_mam`'s 50 at 95.7%, which recovered
every masked base exactly.

These tests pin the classification that replaced it.
"""
from __future__ import annotations

import pytest

from domain.services.target_profile import (
    Division,
    Molecule,
    TargetProfile,
    candidate_databases,
    classify_division,
    classify_molecule,
    organism_from_description,
)
from tools.blast.catalogue import TOO_SLOW
from tools.blast.ena_snapshot import SNAPSHOT

#: The real NCBI lineage for the polar bear, as `TaxonomyService` returns it.
URSUS = (
    "Eukaryota", "Metazoa", "Chordata", "Craniata", "Vertebrata",
    "Euteleostomi", "Mammalia", "Eutheria", "Carnivora", "Caniformia",
    "Ursidae", "Ursus", "Ursus maritimus",
)


class TestDivision:
    @pytest.mark.parametrize(
        ("lineage", "expected"),
        [
            (URSUS, Division.MAM),
            (("Eukaryota", "Metazoa", "Chordata", "Mammalia", "Primates", "Homo",
              "Homo sapiens"), Division.HUM),
            (("Eukaryota", "Metazoa", "Chordata", "Mammalia", "Rodentia", "Mus",
              "Mus musculus"), Division.MUS),
            (("Eukaryota", "Metazoa", "Chordata", "Mammalia", "Rodentia",
              "Rattus", "Rattus norvegicus"), Division.ROD),
            (("Eukaryota", "Metazoa", "Chordata", "Vertebrata", "Actinopteri",
              "Danio", "Danio rerio"), Division.VRT),
            (("Eukaryota", "Metazoa", "Arthropoda", "Insecta", "Drosophila",
              "Drosophila melanogaster"), Division.INV),
            (("Eukaryota", "Viridiplantae", "Streptophyta", "Arabidopsis",
              "Arabidopsis thaliana"), Division.PLN),
            (("Bacteria", "Pseudomonadota", "Escherichia", "Escherichia coli"),
             Division.PRO),
            ((), None),
        ],
    )
    def test_lineages_land_in_the_right_division(self, lineage, expected) -> None:
        assert classify_division(lineage) is expected

    def test_a_mammal_is_never_classified_vertebrate(self) -> None:
        """The whole defect, as a test.

        `Vertebrata` is in the polar bear's lineage, so a rule that tested it
        before `Mammalia` would send every mammal to the one division that by
        construction cannot contain it.
        """
        assert classify_division(URSUS) is not Division.VRT

    def test_archaic_humans_are_mammals_not_human_division(self) -> None:
        """`Homo neanderthalensis` is filed MAM; HUM is *Homo sapiens* alone.

        And the divisions do not nest, so guessing HUM has no sideways recovery
        - the records simply are not there.
        """
        neanderthal = (*URSUS[:6], "Mammalia", "Primates", "Homo",
                       "Homo neanderthalensis")
        assert classify_division(neanderthal) is Division.MAM


class TestMolecule:
    def test_the_real_scaffold_defline_reads_as_nuclear(self) -> None:
        """Verbatim from `NW_007907101`, the record the Genome Agent hands over."""
        assert classify_molecule(
            description="Ursus maritimus isolate Baiyulong unplaced genomic scaffold",
            length=15_920_966,
        ) is Molecule.NUCLEAR

    def test_the_real_mitogenome_defline_reads_as_organellar(self) -> None:
        assert classify_molecule(
            description="Ursus maritimus mitochondrion, complete genome", length=17_017
        ) is Molecule.ORGANELLAR

    def test_assembly_level_alone_is_enough(self) -> None:
        assert classify_molecule(assembly_level="Scaffold") is Molecule.NUCLEAR

    def test_complete_genome_is_not_read_as_nuclear(self) -> None:
        """NCBI reports it for finished mitogenomes too, so it must not decide."""
        assert classify_molecule(assembly_level="Complete Genome") is not Molecule.NUCLEAR

    def test_length_settles_it_when_nothing_else_does(self) -> None:
        """Nothing organellar reaches a megabase."""
        assert classify_molecule(length=2_000_000) is Molecule.NUCLEAR
        assert classify_molecule(length=16_500) is Molecule.UNKNOWN


class TestCandidateDatabases:
    def test_a_mammal_mitogenome_looks_in_finished_mammal_records(self) -> None:
        profile = TargetProfile(Division.MAM, Molecule.ORGANELLAR, "lineage")
        assert candidate_databases(profile)[0] == "em_std_mam"

    def test_a_nuclear_scaffold_opens_on_the_whole_division(self) -> None:
        """Draft scaffolds live in WGS/CON/HTG, which only `em_<div>` carries."""
        profile = TargetProfile(Division.MAM, Molecule.NUCLEAR, "lineage")
        assert candidate_databases(profile)[0] == "em_mam"

    def test_an_unplaceable_organism_proposes_nothing(self) -> None:
        """Better an empty prior than a division picked at random."""
        assert candidate_databases(TargetProfile(None, Molecule.UNKNOWN)) == ()

    @pytest.mark.parametrize("division", list(Division))
    @pytest.mark.parametrize("molecule", list(Molecule))
    def test_every_proposable_code_really_exists(self, division, molecule) -> None:
        """EBI 400s on an unknown code - how `em_rel_vrt` broke every retry."""
        for code in candidate_databases(TargetProfile(division, molecule)):
            assert code in SNAPSHOT, f"{code} is not an ENA database"

    @pytest.mark.parametrize("division", list(Division))
    def test_the_databases_measured_too_slow_are_never_proposed(self, division) -> None:
        """`em_all` took 749 s and `em_std` never finished, against a 600 s slice."""
        for molecule in Molecule:
            proposed = set(candidate_databases(TargetProfile(division, molecule)))
            assert not proposed & set(TOO_SLOW)


class TestOrganismFromDescription:
    @pytest.mark.parametrize(
        ("description", "expected"),
        [
            ("Ursus maritimus isolate PB18-N26025 mitochondrion, complete genome.",
             "Ursus maritimus"),
            ("UNVERIFIED: Ursus maritimus isolate ma-USMA-5 mitochondrion",
             "Ursus maritimus"),
            ("UNVERIFIED_ORG: Acanthochromis polyacanthus mitochondrion",
             "Acanthochromis polyacanthus"),
            ("Ursus maritimus isolate Baiyulong unplaced genomic scaffold",
             "Ursus maritimus"),
            ("Uncultured bacterium clone 12", None),
            ("", None),
            (None, None),
        ],
    )
    def test_real_deflines(self, description, expected) -> None:
        """Every string here is verbatim from a live BLAST result."""
        assert organism_from_description(description) == expected
