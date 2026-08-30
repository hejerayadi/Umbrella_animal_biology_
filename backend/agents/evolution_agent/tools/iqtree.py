"""IQ-TREE client — local binary.

Uses the locally installed IQ-TREE binary for tree building.
Supports ModelFinder (MFP) and UFBoot2.

The binary is located by ``binaries.resolve_binary``: ``$IQTREE_BINARY``
first, then ``iqtree3(.exe)`` / ``iqtree2(.exe)`` / ``iqtree`` on PATH, then
the legacy bundled path kept below for backward compatibility.

Support semantics
-----------------
``bootstrap_support`` and ``confidence_values`` are keyed by INTERNAL NODE
(``node_0``, ``node_1``, …), not by species: UFBoot measures how well a
branch is supported, which is a property of a split, never of a single leaf.
When UFBoot does not run, all three confidence fields stay empty / ``None``
— no value is invented.

Callers must pass FASTA identifiers without whitespace; see
``tools.taxon_ids``.
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
import tempfile
from pathlib import Path

from langsmith import traceable

from .binaries import (
    IQTREE_CANDIDATES,
    IQTREE_ENV_VAR,
    describe_search,
    resolve_binary,
)

_logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent

# Legacy bundled location — last resort, kept so existing setups keep working.
_LOCAL_IQTREE = str(_PROJECT_ROOT / "iqtree" / "iqtree-2.3.6-Windows" / "bin" / "iqtree2.exe")


def resolve_iqtree() -> str | None:
    """Path to a usable IQ-TREE executable, or ``None`` if none is installed."""
    return resolve_binary(IQTREE_ENV_VAR, IQTREE_CANDIDATES, _LOCAL_IQTREE)


class IQTreeError(Exception):
    """Raised when the IQ-TREE service fails or returns invalid data."""


@traceable(name="IQ-TREE build", run_type="tool")
def build_tree(
    alignment: str | dict[str, str],
    model: str = "MFP",
    bootstrap: int = 1000,
    timeout: int = 600,
    poll_interval: int = 10,
) -> "PhyloResult":
    """Build a tree using local IQ-TREE binary.

    Parameters
    ----------
    alignment : str or dict
        Aligned FASTA string or dict of {name: aligned_sequence}.
    model : str
        Substitution model (MFP = ModelFinder, LG+G4, etc.)
    bootstrap : int
        Number of UFBoot replicates (0 = no bootstrap).
    timeout : int
        Max seconds to wait for tree building.

    Returns
    -------
    PhyloResult
        Parsed tree result with newick, model, support values.
    """
    alignment_text = (
        _dict_to_fasta(alignment) if isinstance(alignment, dict) else alignment
    )
    if not alignment_text.strip():
        raise IQTreeError("An aligned FASTA input is required.")

    binary = resolve_iqtree()
    if not binary:
        raise IQTreeError(
            "No IQ-TREE executable was found. Searched: "
            + describe_search(IQTREE_ENV_VAR, IQTREE_CANDIDATES, _LOCAL_IQTREE)
        )

    return _run_local_iqtree(alignment_text, model, bootstrap, timeout, binary)


@traceable(name="IQ-TREE local binary run", run_type="tool")
def _run_local_iqtree(
    alignment_text: str,
    model: str,
    bootstrap: int,
    timeout: int,
    binary: str | None = None,
) -> "PhyloResult":
    """Run IQ-TREE locally."""
    executable = binary or resolve_iqtree()
    if not executable:
        raise IQTreeError(
            "No IQ-TREE executable was found. Searched: "
            + describe_search(IQTREE_ENV_VAR, IQTREE_CANDIDATES, _LOCAL_IQTREE)
        )

    with tempfile.TemporaryDirectory() as tmpdir:
        aln_path = os.path.join(tmpdir, "alignment.fasta")
        with open(aln_path, "w", encoding="utf-8") as f:
            f.write(alignment_text)

        # Build command
        cmd = [
            executable,
            "-s", aln_path,
            "-m", model,
            "-pre", os.path.join(tmpdir, "output"),
            "-quiet",
        ]
        if bootstrap > 0:
            cmd.extend(["-bb", str(bootstrap)])

        _logger.info("[IQ-TREE] running: %s", " ".join(cmd[:5]))

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            raise IQTreeError(f"IQ-TREE timed out after {timeout}s")

        if result.returncode != 0:
            stderr = result.stderr[:500] if result.stderr else "no error output"
            raise IQTreeError(f"IQ-TREE failed (exit {result.returncode}): {stderr}")

        # Read the treefile
        treefile = os.path.join(tmpdir, "output.treefile")
        if not os.path.exists(treefile):
            raise IQTreeError("IQ-TREE did not produce a treefile")

        with open(treefile, "r", encoding="utf-8") as f:
            newick = f.read().strip()

        if not newick:
            raise IQTreeError("IQ-TREE treefile is empty")

        # Read the log for model info
        logfile = os.path.join(tmpdir, "output.log")
        model_used = model
        if os.path.exists(logfile):
            with open(logfile, "r", encoding="utf-8") as f:
                log_content = f.read()
            # Best model line: "Best-fit model: LG+G4..."
            m = re.search(r"Best-fit model:\s*(\S+)", log_content)
            if m:
                model_used = m.group(1)

        # Parse per-internal-node bootstrap support from the Newick string.
        bootstrap_support = _parse_bootstrap(newick) if bootstrap > 0 else {}

        # Confidence is derived ONLY from real UFBoot values. With no
        # bootstrap there is nothing to derive, so nothing is reported —
        # an invented default would misrepresent an unsupported tree.
        if bootstrap_support:
            confidence_values = {
                node: val / 100.0 for node, val in bootstrap_support.items()
            }
            overall_confidence = round(
                sum(confidence_values.values()) / len(confidence_values), 4
            )
            ufboot_run = True
        else:
            confidence_values = {}
            overall_confidence = None
            ufboot_run = False

        _logger.info(
            "[IQ-TREE] tree built: model=%s, ufboot_run=%s, bootstrap_nodes=%d",
            model_used, ufboot_run, len(bootstrap_support),
        )

        return PhyloResult(
            newick_tree=newick,
            model=model_used,
            bootstrap_support=bootstrap_support,
            confidence_values=confidence_values,
            overall_confidence=overall_confidence,
            ufboot_run=ufboot_run,
        )


def _parse_bootstrap(newick: str) -> dict[str, int]:
    """Parse bootstrap support values from Newick string.

    IQ-TREE 2 Newick bootstrap format: )value:branch_length
    """
    support = {}
    # Pattern: )number: (IQ-TREE bootstrap format)
    for match in re.finditer(r"\)(\d+(?:\.\d+)?)\s*:", newick):
        val = float(match.group(1))
        if val <= 100:
            support[f"node_{len(support)}"] = int(val)
    return support


def _dict_to_fasta(sequences: dict[str, str]) -> str:
    lines: list[str] = []
    for name, sequence in sequences.items():
        lines.append(f">{name}")
        for index in range(0, len(sequence), 60):
            lines.append(sequence[index : index + 60])
    return "\n".join(lines) + "\n"


class PhyloResult:
    """Normalized result returned by the IQ-TREE client.

    ``bootstrap_support`` and ``confidence_values`` are keyed by internal
    node (``node_0``, ``node_1``, …). They express branch support, never
    per-species confidence. Both are empty and ``overall_confidence`` is
    ``None`` when UFBoot did not run — check ``ufboot_run``.
    """

    def __init__(
        self,
        newick_tree: str,
        model: str,
        bootstrap_support: dict[str, int],
        confidence_values: dict[str, float],
        overall_confidence: float | None,
        ufboot_run: bool = False,
    ) -> None:
        self.newick_tree = newick_tree
        self.model = model
        self.bootstrap_support = bootstrap_support
        self.confidence_values = confidence_values
        self.overall_confidence = overall_confidence
        self.ufboot_run = ufboot_run
