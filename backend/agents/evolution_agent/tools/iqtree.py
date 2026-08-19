"""IQ-TREE phylogenetic tree builder wrapper.

Calls the IQ-TREE 2 CLI via subprocess.
Input: aligned sequences (FASTA or PHYLIP format)
Output: Newick tree, bootstrap support values, substitution model

Pipeline position:
    Sequences → MAFFT → aligned sequences → IQ-TREE → tree + support

IQ-TREE steps:
    1. ModelFinder (-m MFP) — picks best substitution model
    2. UFBoot2 (-bb 1000) — ultrafast bootstrap with 1000 replicates
    3. Output: .treefile (Newick), .log (model, support values)
"""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
from pathlib import Path

# Default IQ-TREE path (Windows install)
_PROJECT_ROOT = Path(__file__).resolve().parents[4]
_IQTREE_DEFAULT = str(_PROJECT_ROOT / "iqtree" / "iqtree-2.3.6-Windows" / "bin" / "iqtree2.exe")


class IQTreeError(Exception):
    """Raised when IQ-TREE fails."""


def build_tree(
    alignment: str | dict[str, str],
    iqtree_path: str | None = None,
    model: str = "MFP",
    bootstrap: int = 1000,
    timeout: int = 600,
) -> PhyloResult:
    """Build a phylogenetic tree using IQ-TREE.

    Parameters
    ----------
    alignment : str or dict[str, str]
        Either a FASTA string (aligned) or a dict of {species: aligned_sequence}.
    iqtree_path : str, optional
        Path to iqtree2.exe. Defaults to bundled install.
    model : str
        Substitution model or "MFP" for ModelFinder.
    bootstrap : int
        Number of UFBoot replicates (0 to disable).
        Automatically disabled if fewer than 4 sequences.
    timeout : int
        Max seconds to wait for IQ-TREE.

    Returns
    -------
    PhyloResult
        Newick tree, model, bootstrap support, confidence values.

    Raises
    ------
    IQTreeError
        If IQ-TREE fails or times out.
    """
    iqtree = iqtree_path or _IQTREE_DEFAULT
    if not os.path.exists(iqtree):
        raise IQTreeError(f"IQ-TREE not found at: {iqtree}")

    # Prepare alignment file
    if isinstance(alignment, dict):
        alignment_str = _dict_to_fasta(alignment)
        n_sequences = len(alignment)
    else:
        alignment_str = alignment
        n_sequences = alignment_str.count(">")

    # IQ-TREE requires >=4 sequences for bootstrap
    if n_sequences < 4:
        bootstrap = 0

    with tempfile.TemporaryDirectory() as tmpdir:
        aln_path = os.path.join(tmpdir, "alignment.fasta")
        with open(aln_path, "w", encoding="utf-8") as f:
            f.write(alignment_str)

        # Build IQ-TREE command
        cmd = [
            iqtree,
            "-s", aln_path,
            "-m", model,
            "-bb", str(bootstrap),
            "-nt", "AUTO",
            "-quiet",
            "--prefix", os.path.join(tmpdir, "output"),
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
        except subprocess.TimeoutExpired:
            raise IQTreeError(f"IQ-TREE timed out after {timeout}s.")

        # Parse outputs
        treefile = os.path.join(tmpdir, "output.treefile")
        logfile = os.path.join(tmpdir, "output.log")

        if not os.path.exists(treefile):
            raise IQTreeError(
                f"IQ-TREE did not produce a treefile.\n"
                f"stderr: {result.stderr[:500]}\n"
                f"stdout: {result.stdout[:500]}"
            )

        # Read Newick tree
        with open(treefile, "r", encoding="utf-8") as f:
            newick = f.read().strip()

        # Parse log for model and bootstrap values
        model_name = "unknown"
        bootstrap_support = {}
        if os.path.exists(logfile):
            with open(logfile, "r", encoding="utf-8") as f:
                log_content = f.read()
            model_name = _parse_model(log_content)
            bootstrap_support = _parse_bootstrap(newick)

        # Compute confidence values
        confidence_values = _compute_confidence(bootstrap_support)
        overall_confidence = (
            sum(confidence_values.values()) / len(confidence_values)
            if confidence_values
            else 0.0
        )

        return PhyloResult(
            newick_tree=newick,
            model=model_name,
            bootstrap_support=bootstrap_support,
            confidence_values=confidence_values,
            overall_confidence=round(overall_confidence, 4),
        )


def _parse_model(log_content: str) -> str:
    """Extract the best model from IQ-TREE log."""
    # Look for "Best-fit model:" line
    match = re.search(r"Best-fit model:\s*(\S+)", log_content)
    if match:
        return match.group(1)
    # Fallback: look for model name in log
    match = re.search(r"Model found:\s*(\S+)", log_content)
    if match:
        return match.group(1)
    return "unknown"


def _parse_bootstrap(newick: str) -> dict[str, int]:
    """Parse UFBoot support values from Newick string.

    IQ-TREE 2 annotates internal nodes as )bootstrap_value:branch_length
    Example: ((A:0.1,B:0.1)100:0.5,C:0.2)95:0.3;
    Also handles {value} format from older versions.
    """
    support = {}

    # IQ-TREE 2 format: )number:  (bootstrap support after closing paren)
    matches_v2 = re.findall(r"\)(\d+)(?=:)", newick)
    for i, value in enumerate(matches_v2):
        val = int(value)
        if 0 <= val <= 100:
            support[f"node_{i+1}"] = val

    # Legacy format: {number}
    if not support:
        matches_legacy = re.findall(r"\{(\d+)\}", newick)
        for i, value in enumerate(matches_legacy):
            support[f"node_{i+1}"] = int(value)

    return support


def _compute_confidence(bootstrap: dict[str, int]) -> dict[str, float]:
    """Convert bootstrap percentages to confidence scores (0-1)."""
    return {node: value / 100.0 for node, value in bootstrap.items()}


def _dict_to_fasta(sequences: dict[str, str]) -> str:
    """Convert dict to FASTA format."""
    lines = []
    for name, seq in sequences.items():
        lines.append(f">{name}")
        for i in range(0, len(seq), 60):
            lines.append(seq[i : i + 60])
    return "\n".join(lines) + "\n"


class PhyloResult:
    """Result from IQ-TREE."""
    def __init__(
        self,
        newick_tree: str,
        model: str,
        bootstrap_support: dict[str, int],
        confidence_values: dict[str, float],
        overall_confidence: float,
    ):
        self.newick_tree = newick_tree
        self.model = model
        self.bootstrap_support = bootstrap_support
        self.confidence_values = confidence_values
        self.overall_confidence = overall_confidence
