"""
Evaluation pipeline for DNA gap prediction models (Post-Sprint 1).

Computes per-model, per-sequence-type metrics (normalized edit distance and
DNA-adapted BLEU score) over a set of GapPredictions and their ground-truth
reference sequences.

Mammuthus primigenius is excluded from all evaluation sets per Req 9.4.

Requirements covered: 9.1, 9.2, 9.3, 9.4
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from ...schema import GapPrediction

# ---------------------------------------------------------------------------
# Mammoth exclusion (Req 9.4)
# ---------------------------------------------------------------------------

MAMMOTH_SPECIES_ID = "Mammuthus primigenius"


def is_mammoth(species_id: str) -> bool:
    """Return ``True`` if *species_id* refers to Mammuthus primigenius.

    Case-insensitive substring match.  Spaces and underscores in
    *species_id* are treated as equivalent so that variants like
    ``"mammuthus_primigenius"`` or ``"Mammuthus primigenius"`` both match.

    Parameters
    ----------
    species_id : str
        Species identifier string to check.

    Returns
    -------
    bool
    """
    # Normalise both strings: lowercase, replace underscores with spaces
    normalised = species_id.lower().replace("_", " ").strip()
    return "mammuthus primigenius" in normalised


def filter_training_data(
    items: list,
    get_species_id: Callable[[object], str],
) -> list:
    """Remove mammoth entries from a list of training/validation items.

    Parameters
    ----------
    items : list
        Any collection of data items.
    get_species_id : Callable[[object], str]
        Extracts the species ID string from a single item.

    Returns
    -------
    list
        Filtered list with Mammuthus primigenius entries removed (Req 9.4).
    """
    return [item for item in items if not is_mammoth(get_species_id(item))]


# ---------------------------------------------------------------------------
# Sequence similarity metrics (Req 9.3)
# ---------------------------------------------------------------------------

def normalized_edit_distance(predicted: str, reference: str) -> float:
    """Compute Levenshtein edit distance normalized by the max sequence length.

    Returns 0.0 for identical sequences and 1.0 for maximally different ones.

    Implementation uses the standard DP recurrence without external libraries.

    Parameters
    ----------
    predicted : str
        Predicted DNA sequence.
    reference : str
        Ground-truth (reference) sequence.

    Returns
    -------
    float
        Value in [0.0, 1.0].
    """
    if not predicted and not reference:
        return 0.0
    max_len = max(len(predicted), len(reference))
    if max_len == 0:
        return 0.0

    # Standard DP edit distance (O(m*n) time, O(n) space)
    m, n = len(predicted), len(reference)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            if predicted[i - 1] == reference[j - 1]:
                curr[j] = prev[j - 1]
            else:
                curr[j] = 1 + min(prev[j], curr[j - 1], prev[j - 1])
        prev = curr

    return prev[n] / max_len


def dna_bleu_score(predicted: str, reference: str, max_n: int = 4) -> float:
    """Compute a character-level BLEU score adapted for DNA sequences.

    Calculates the geometric mean of n-gram precisions for n = 1 … max_n
    with add-1 smoothing to avoid zero scores on short sequences.

    Parameters
    ----------
    predicted : str
        Predicted DNA sequence.
    reference : str
        Ground-truth (reference) sequence.
    max_n : int
        Maximum n-gram order (default 4).

    Returns
    -------
    float
        BLEU-like score in [0.0, 1.0].
    """
    if not predicted or not reference:
        return 0.0

    log_avg = 0.0
    for n in range(1, max_n + 1):
        pred_ngrams = Counter(predicted[i: i + n] for i in range(len(predicted) - n + 1))
        ref_ngrams = Counter(reference[i: i + n] for i in range(len(reference) - n + 1))

        # Clipped count: min of pred count and ref count per n-gram
        clipped = sum(min(cnt, ref_ngrams[ngram]) for ngram, cnt in pred_ngrams.items())
        total_pred = max(len(predicted) - n + 1, 0)

        # Add-1 smoothing to handle zero counts gracefully
        precision = (clipped + 1) / (total_pred + 1)
        log_avg += math.log(precision)

    bleu = math.exp(log_avg / max_n)

    # Brevity penalty (BP)
    if len(predicted) < len(reference):
        bp = math.exp(1.0 - len(reference) / max(len(predicted), 1))
    else:
        bp = 1.0

    return max(0.0, min(1.0, bp * bleu))


# ---------------------------------------------------------------------------
# Data containers (Req 9.1, 9.2)
# ---------------------------------------------------------------------------

@dataclass
class ModelMetrics:
    """Aggregate metrics for a single (model_name, sequence_type) group.

    Attributes
    ----------
    model_name : str
        Canonical model name, e.g. ``"DNABERT"``.
    sequence_type : str
        ``"nuclear"`` or ``"mitochondrial"``.
    count : int
        Number of predictions in this group.
    avg_edit_distance : float
        Mean normalized edit distance (lower is better).
    avg_bleu_score : float
        Mean DNA BLEU score (higher is better).
    avg_confidence : float
        Mean model confidence score.
    """

    model_name: str
    sequence_type: str
    count: int
    avg_edit_distance: float
    avg_bleu_score: float
    avg_confidence: float


@dataclass
class EvaluationReport:
    """Complete evaluation results across all models and sequence types.

    Attributes
    ----------
    metrics : list[ModelMetrics]
        One entry per (model_name, sequence_type) pair.
    total_predictions : int
        Total number of predictions evaluated.
    excluded_mammoth_count : int
        Number of predictions excluded due to Mammuthus primigenius filter.
    """

    metrics: list[ModelMetrics] = field(default_factory=list)
    total_predictions: int = 0
    excluded_mammoth_count: int = 0

    def summary(self) -> str:
        """Return a human-readable text summary of the evaluation report."""
        lines = [
            f"Evaluation Report",
            f"  Total predictions : {self.total_predictions}",
            f"  Mammoth excluded  : {self.excluded_mammoth_count}",
            "",
        ]
        for m in sorted(self.metrics, key=lambda x: (x.model_name, x.sequence_type)):
            lines.append(
                f"  [{m.model_name} / {m.sequence_type}]  n={m.count}"
                f"  edit_dist={m.avg_edit_distance:.4f}"
                f"  bleu={m.avg_bleu_score:.4f}"
                f"  confidence={m.avg_confidence:.4f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main entry point (Req 9.1, 9.2, 9.3, 9.4)
# ---------------------------------------------------------------------------

def compute_metrics(
    predictions: list["GapPrediction"],
    references: list[str],
    species_ids: list[str] | None = None,
) -> EvaluationReport:
    """Compute per-model, per-sequence-type evaluation metrics.

    Models are reported **separately** — results are never averaged across
    model boundaries (Req 9.1).  Within each model, metrics are stratified
    by ``sequence_type`` (Req 9.2).  Mammuthus primigenius entries are
    excluded when ``species_ids`` is provided (Req 9.4).

    Parameters
    ----------
    predictions : list[GapPrediction]
        Predictions produced by any model in the cascade.
    references : list[str]
        Ground-truth sequences aligned to *predictions* by index.
    species_ids : list[str] | None
        Optional species identifier per prediction for mammoth filtering.

    Returns
    -------
    EvaluationReport
        Grouped metrics with one ``ModelMetrics`` per
        (model_name, sequence_type) pair.
    """
    assert len(predictions) == len(references), (
        f"predictions and references must be the same length "
        f"({len(predictions)} != {len(references)})"
    )
    if species_ids is not None:
        assert len(species_ids) == len(predictions), (
            f"species_ids length must match predictions ({len(species_ids)} != {len(predictions)})"
        )

    # Group by (model_name, sequence_type)
    # Each group accumulates (edit_distances, bleu_scores, confidences)
    groups: dict[
        tuple[str, str],
        dict[str, list[float]],
    ] = defaultdict(lambda: {"edit": [], "bleu": [], "conf": []})

    excluded_count = 0
    evaluated_count = 0

    for i, (pred, ref) in enumerate(zip(predictions, references)):
        # Mammoth exclusion (Req 9.4)
        if species_ids is not None and is_mammoth(species_ids[i]):
            excluded_count += 1
            continue

        key = (pred.model_used, pred.gap.sequence_type)
        groups[key]["edit"].append(normalized_edit_distance(pred.predicted_sequence, ref))
        groups[key]["bleu"].append(dna_bleu_score(pred.predicted_sequence, ref))
        groups[key]["conf"].append(pred.confidence)
        evaluated_count += 1

    # Build ModelMetrics per group
    metrics: list[ModelMetrics] = []
    for (model_name, seq_type), vals in groups.items():
        n = len(vals["edit"])
        metrics.append(
            ModelMetrics(
                model_name=model_name,
                sequence_type=seq_type,
                count=n,
                avg_edit_distance=sum(vals["edit"]) / n,
                avg_bleu_score=sum(vals["bleu"]) / n,
                avg_confidence=sum(vals["conf"]) / n,
            )
        )

    return EvaluationReport(
        metrics=metrics,
        total_predictions=evaluated_count,
        excluded_mammoth_count=excluded_count,
    )
