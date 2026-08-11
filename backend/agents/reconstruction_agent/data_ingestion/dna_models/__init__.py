"""
dna_models — Deep Learning DNA predictors for the reconstruction_agent.

Public API
----------
DNAModelRouter
    Singleton cascade router: DNABERT → NucleotideTransformer → [Seq2Seq] → AzureGPT.

CANONICAL_ORDER
    Ordered list of model names used by DNAModelRouter.

FlanksWindow
    Context window dataclass (left_context, gap_length, right_context,
    sequence_type, total_length property).

DNABERTPredictor
    Zero-shot MLM predictor backed by ``zhihan1996/DNA_bert_6``.

NucleotideTransformerPredictor
    Zero-shot MLM predictor using InstaDeep's Nucleotide Transformer family.

_select_nt_checkpoint
    Select the NT checkpoint based on available VRAM.

_resolve_device
    Detect or force the compute device (reads ``DNA_MODEL_DEVICE`` env var).

_pick_dtype
    Select float32 / float16 based on available VRAM.

_get_free_vram_gb
    Query free VRAM on the default CUDA device.

_normalize_length
    Truncate or pad a predicted sequence to an exact target length.

_symmetric_truncate
    Symmetrically trim left/right token flanks to fit a model context window.

--- Post-Sprint 1 Seq2Seq API ---

Seq2SeqPredictor
    Custom encoder-decoder Transformer for DNA gap prediction, trained
    from scratch on windowing.py triplets.  Uses beam search (beam=4)
    or greedy fallback depending on available VRAM.

VOCAB
    Token vocabulary dict {A, C, G, T, N, <BOS>, <EOS>, <PAD>} → int IDs.

--- Post-Sprint 1 evaluation API ---

normalized_edit_distance
    Levenshtein edit distance normalized to [0.0, 1.0].

dna_bleu_score
    DNA-adapted BLEU score (n-gram precision on nucleotide characters).

ModelMetrics
    Per-model metrics broken down by sequence_type.

EvaluationReport
    Complete evaluation results across all models and sequence types.

compute_metrics
    Main entry point: compute per-model, per-sequence_type metrics from
    GapPrediction / reference pairs.

is_mammoth
    Return True if a species_id refers to Mammuthus primigenius.

filter_training_data
    Filter out mammoth entries from any training/validation dataset.
"""

from .dnabert_predictor import DNABERTPredictor
from .evaluation import (
    EvaluationReport,
    ModelMetrics,
    compute_metrics,
    dna_bleu_score,
    filter_training_data,
    is_mammoth,
    normalized_edit_distance,
)
from .nucleotide_transformer_predictor import (
    NucleotideTransformerPredictor,
    _select_nt_checkpoint,
)
from .router import CANONICAL_ORDER, DNAModelRouter
from .seq2seq_predictor import VOCAB, Seq2SeqPredictor
from .utils import (
    FlanksWindow,
    _get_free_vram_gb,
    _normalize_length,
    _pick_dtype,
    _resolve_device,
    _symmetric_truncate,
)

__all__ = [
    # Sprint 1 — cascade & predictors
    "CANONICAL_ORDER",
    "DNABERTPredictor",
    "DNAModelRouter",
    "FlanksWindow",
    "NucleotideTransformerPredictor",
    "_get_free_vram_gb",
    "_normalize_length",
    "_pick_dtype",
    "_resolve_device",
    "_select_nt_checkpoint",
    "_symmetric_truncate",
    # Post-Sprint 1 — Seq2Seq predictor
    "Seq2SeqPredictor",
    "VOCAB",
    # Post-Sprint 1 — evaluation pipeline
    "EvaluationReport",
    "ModelMetrics",
    "compute_metrics",
    "dna_bleu_score",
    "filter_training_data",
    "is_mammoth",
    "normalized_edit_distance",
]
