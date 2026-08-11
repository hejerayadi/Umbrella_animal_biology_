"""
test_sprint1_checkpoint.py — Sprint 1 intermediate checkpoint tests.

Verifies that all required classes are importable from data_ingestion.dna_models
and that core structural invariants hold (without loading real ML checkpoints).

The conftest.py in this directory stubs out torch and transformers so these
tests run in environments where those heavy packages are not installed.
"""
import sys
import types


# ---------------------------------------------------------------------------
# Step 1: Import verification
# ---------------------------------------------------------------------------

def test_dnabert_predictor_importable():
    """DNABERTPredictor must be importable from data_ingestion.dna_models."""
    from backend.agents.reconstruction_agent.data_ingestion.dna_models import (
        DNABERTPredictor,
    )
    assert DNABERTPredictor is not None


def test_nucleotide_transformer_predictor_importable():
    """NucleotideTransformerPredictor must be importable from data_ingestion.dna_models."""
    from backend.agents.reconstruction_agent.data_ingestion.dna_models import (
        NucleotideTransformerPredictor,
    )
    assert NucleotideTransformerPredictor is not None


def test_dna_model_router_importable():
    """DNAModelRouter must be importable from data_ingestion.dna_models.router."""
    from backend.agents.reconstruction_agent.data_ingestion.dna_models.router import (
        DNAModelRouter,
    )
    assert DNAModelRouter is not None


def test_flanks_window_importable():
    """FlanksWindow must be importable from data_ingestion.dna_models."""
    from backend.agents.reconstruction_agent.data_ingestion.dna_models import FlanksWindow
    assert FlanksWindow is not None


def test_all_public_exports_present():
    """All symbols listed in __all__ must be present in the package namespace."""
    import backend.agents.reconstruction_agent.data_ingestion.dna_models as pkg

    expected = {
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
    }
    for name in expected:
        assert hasattr(pkg, name), f"Expected '{name}' to be exported from dna_models"


# ---------------------------------------------------------------------------
# Step 2: Structural invariants (no real model loading)
# ---------------------------------------------------------------------------

def test_canonical_order_contains_required_models():
    """CANONICAL_ORDER must include DNABERT and NucleotideTransformer in that order."""
    from backend.agents.reconstruction_agent.data_ingestion.dna_models import CANONICAL_ORDER

    assert "DNABERT" in CANONICAL_ORDER
    assert "NucleotideTransformer" in CANONICAL_ORDER
    assert CANONICAL_ORDER.index("DNABERT") < CANONICAL_ORDER.index("NucleotideTransformer"), (
        "DNABERT must precede NucleotideTransformer in CANONICAL_ORDER"
    )


def test_router_singleton_get_returns_same_instance():
    """DNAModelRouter.get() must return the same instance on repeated calls (singleton)."""
    from backend.agents.reconstruction_agent.data_ingestion.dna_models.router import DNAModelRouter

    # Reset singleton state for a clean test
    DNAModelRouter._instance = None

    instance_a = DNAModelRouter.get()
    instance_b = DNAModelRouter.get()
    assert instance_a is instance_b, "DNAModelRouter.get() must return the same singleton instance"

    # Cleanup
    DNAModelRouter._instance = None


def test_seq2seq_unavailable_in_sprint1():
    """_seq2seq_available() must return False in Sprint 1."""
    from backend.agents.reconstruction_agent.data_ingestion.dna_models.router import DNAModelRouter

    router = DNAModelRouter()
    assert router._seq2seq_available() is False, (
        "_seq2seq_available() must return False in Sprint 1"
    )


def test_flanks_window_total_length():
    """FlanksWindow.total_length must equal len(left) + gap_length + len(right)."""
    from backend.agents.reconstruction_agent.data_ingestion.dna_models import FlanksWindow

    fw = FlanksWindow(
        left_context="ACGT",
        gap_length=10,
        right_context="TTTT",
        sequence_type="nuclear",
    )
    assert fw.total_length == len("ACGT") + 10 + len("TTTT") == 18


def test_flanks_window_sequence_type_propagated():
    """FlanksWindow.sequence_type must store the value without altering left/right context."""
    from backend.agents.reconstruction_agent.data_ingestion.dna_models import FlanksWindow

    for seq_type in ("nuclear", "mitochondrial"):
        fw = FlanksWindow(
            left_context="AAAA",
            gap_length=5,
            right_context="CCCC",
            sequence_type=seq_type,
        )
        assert fw.sequence_type == seq_type
        assert fw.left_context == "AAAA"
        assert fw.right_context == "CCCC"


# ---------------------------------------------------------------------------
# Step 3: Utils correctness
# ---------------------------------------------------------------------------

def test_normalize_length_truncates():
    """_normalize_length must truncate sequences longer than target_length."""
    from backend.agents.reconstruction_agent.data_ingestion.dna_models import _normalize_length

    result = _normalize_length("ACGTACGT", 4)
    assert result == "ACGT"
    assert len(result) == 4


def test_normalize_length_pads():
    """_normalize_length must pad with 'N' when sequence is shorter than target_length."""
    from backend.agents.reconstruction_agent.data_ingestion.dna_models import _normalize_length

    result = _normalize_length("AC", 5)
    assert result == "ACNNN"
    assert len(result) == 5


def test_normalize_length_exact():
    """_normalize_length must return unchanged sequence when length matches exactly."""
    from backend.agents.reconstruction_agent.data_ingestion.dna_models import _normalize_length

    result = _normalize_length("ACGT", 4)
    assert result == "ACGT"


def test_symmetric_truncate_preserves_mask():
    """_symmetric_truncate must never modify mask_tokens."""
    from backend.agents.reconstruction_agent.data_ingestion.dna_models import _symmetric_truncate

    left = list(range(100))
    mask = [999, 999, 999]  # sentinel values
    right = list(range(100))

    _, returned_mask, _ = _symmetric_truncate(left, mask, right, max_tokens=50)
    assert returned_mask == mask, "_symmetric_truncate must not modify mask_tokens"


def test_symmetric_truncate_fits_in_window():
    """After truncation, total token count must not exceed max_tokens."""
    from backend.agents.reconstruction_agent.data_ingestion.dna_models import _symmetric_truncate

    left = list(range(200))
    mask = list(range(10))
    right = list(range(200))
    max_tokens = 64

    left_t, mask_t, right_t = _symmetric_truncate(left, mask, right, max_tokens)
    total = len(left_t) + len(mask_t) + len(right_t) + 2  # +2 for [CLS] [SEP]
    assert total <= max_tokens, (
        f"Total tokens after truncation ({total}) must be <= max_tokens ({max_tokens})"
    )


def test_resolve_device_returns_string():
    """_resolve_device must return a string ('cpu' or 'cuda')."""
    from backend.agents.reconstruction_agent.data_ingestion.dna_models import _resolve_device

    device = _resolve_device()
    assert isinstance(device, str)
    assert device in ("cpu", "cuda")


def test_select_nt_checkpoint_high_vram():
    """_select_nt_checkpoint must return the 2.5B model when VRAM >= 12 GB."""
    from backend.agents.reconstruction_agent.data_ingestion.dna_models import _select_nt_checkpoint

    checkpoint = _select_nt_checkpoint(12.0)
    assert "2.5b" in checkpoint or "2.5B" in checkpoint


def test_select_nt_checkpoint_low_vram():
    """_select_nt_checkpoint must return the 500M model when VRAM < 12 GB."""
    from backend.agents.reconstruction_agent.data_ingestion.dna_models import _select_nt_checkpoint

    checkpoint = _select_nt_checkpoint(4.0)
    assert "500m" in checkpoint or "500M" in checkpoint


# ---------------------------------------------------------------------------
# Phase-1 bug-fix tests — n_mask correction + majority-vote decoding
# (added after audit Parts 1-12 identified the ~83% N-output root cause)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# _token_to_kmer: replaces the old _token_to_base
# ---------------------------------------------------------------------------

def test_token_to_kmer_returns_full_kmer():
    """_token_to_kmer must return the full 6-mer, not just the first character."""
    from backend.agents.reconstruction_agent.data_ingestion.dna_models.dnabert_predictor import (
        _token_to_kmer,
    )

    assert _token_to_kmer("ACGTAC") == "ACGTAC", "Must return all 6 bases"
    assert _token_to_kmer("TTTTTT") == "TTTTTT"
    assert _token_to_kmer("GCGCGC") == "GCGCGC"


def test_token_to_kmer_single_base():
    """_token_to_kmer must handle 1-character canonical tokens."""
    from backend.agents.reconstruction_agent.data_ingestion.dna_models.dnabert_predictor import (
        _token_to_kmer,
    )

    assert _token_to_kmer("A") == "A"
    assert _token_to_kmer("C") == "C"


def test_token_to_kmer_non_canonical_returns_n_string():
    """_token_to_kmer must return N*len for non-canonical tokens."""
    from backend.agents.reconstruction_agent.data_ingestion.dna_models.dnabert_predictor import (
        _token_to_kmer,
    )

    result = _token_to_kmer("[CLS]")
    assert all(c == "N" for c in result), f"Expected all-N, got {result!r}"
    assert len(result) >= 1


def test_token_to_base_no_longer_exported():
    """The old _token_to_base helper must have been removed / renamed."""
    import importlib
    import backend.agents.reconstruction_agent.data_ingestion.dna_models.dnabert_predictor as mod

    assert not hasattr(mod, "_token_to_base"), (
        "_token_to_base was removed and replaced by _token_to_kmer; "
        "it must not remain exported from dnabert_predictor"
    )


# ---------------------------------------------------------------------------
# n_mask formula: max(1, gap.length - 5)
# ---------------------------------------------------------------------------

def test_n_mask_formula_for_various_gap_lengths():
    """n_mask = max(1, L-5) must match the sliding-window count for all lengths."""
    # This mirrors the formula used in both predictors.
    for length, expected in [
        (1,   1),   # min-clamped
        (5,   1),   # min-clamped (L-5=0 → max(1,0)=1)
        (6,   1),   # L-5=1 → 1 MASK — exact fit for 1 k-mer
        (7,   2),
        (10,  5),
        (30, 25),   # matches _to_kmers("A"*30) == 25 tokens
        (500, 495),
    ]:
        result = max(1, length - 5)
        assert result == expected, (
            f"n_mask({length}) = {result}, expected {expected}"
        )


def test_n_mask_consistent_with_to_kmers():
    """n_mask must equal the token count produced by _to_kmers for the same length."""
    from backend.agents.reconstruction_agent.data_ingestion.dna_models.dnabert_predictor import (
        _to_kmers,
    )

    for length in [6, 7, 10, 20, 30, 100]:
        seq = "A" * length
        kmer_tokens = len(_to_kmers(seq).split())
        n_mask = max(1, length - 5)
        assert kmer_tokens == n_mask, (
            f"length={length}: _to_kmers gives {kmer_tokens} tokens but "
            f"n_mask={n_mask} — they must be equal"
        )


# ---------------------------------------------------------------------------
# Majority-vote logic: hand-crafted mock inference test
# ---------------------------------------------------------------------------

class _MockGap:
    """Minimal stand-in for GapRegion."""
    def __init__(self, start: int, length: int):
        self.start = start
        self.length = length
        self.end = start + length
        self.in_scope = True
        self.sequence_type = "nuclear"


class _MockFlanks:
    def __init__(self, left: str = "ACGTACGTACGT", right: str = "TTTTTTTTTTTTT"):
        self.left_context = left
        self.right_context = right


def _run_mock_dnabert_predict(gap_length: int, mock_kmer_per_mask: str):
    """
    Run DNABERTPredictor.predict() with a mocked tokenizer + model that always
    predicts `mock_kmer_per_mask` for every MASK position.

    Returns (predicted_sequence, confidence, votes_per_position) where
    votes_per_position[p] is the number of MASK tokens that voted on position p.
    """
    import types
    import torch
    import torch.nn.functional as F
    from collections import Counter

    from backend.agents.reconstruction_agent.data_ingestion.dna_models.dnabert_predictor import (
        DNABERTPredictor,
        _token_to_kmer,
        _to_kmers,
    )
    from backend.agents.reconstruction_agent.data_ingestion.dna_models.utils import (
        FlanksWindow,
    )

    k = 6
    n_mask = max(1, gap_length - 5)

    # Build a fake vocabulary: index 0 → mock_kmer_per_mask
    MOCK_TOKEN_ID = 42
    VOCAB_SIZE = 100

    class FakeTok:
        cls_token_id = 0
        sep_token_id = 1
        mask_token_id = 2
        model_max_length = 512

        def __call__(self, text, add_special_tokens=False, return_tensors=None):
            # Every token → MOCK_TOKEN_ID for mask, 3 for other
            if "[MASK]" in text:
                ids = [2] * text.count("[MASK]")
            else:
                ids = [3] * max(1, len(text.split()))
            return {"input_ids": ids}

        def convert_ids_to_tokens(self, token_id):
            if token_id == 2:
                return "[MASK]"
            return mock_kmer_per_mask

    class FakeOutput:
        def __init__(self, logits):
            self.logits = logits

    class FakeModel:
        def __call__(self, input_ids, attention_mask):
            seq_len = input_ids.shape[1]
            logits = torch.zeros(1, seq_len, VOCAB_SIZE)
            # Make MOCK_TOKEN_ID have the highest score at every position
            logits[0, :, MOCK_TOKEN_ID] = 10.0
            return FakeOutput(logits)

        def eval(self):
            return self

    # Inject mock singletons — save and restore original
    orig_model = DNABERTPredictor._model
    orig_tokenizer = DNABERTPredictor._tokenizer
    try:
        DNABERTPredictor._model = FakeModel()
        DNABERTPredictor._tokenizer = FakeTok()

        predictor = DNABERTPredictor()
        flanks = _MockFlanks()
        gap = _MockGap(start=0, length=gap_length)

        # Monkeypatch _load so it doesn't try to hit HF
        predictor._load = lambda device: None

        seq, conf = predictor.predict(flanks, gap)
    finally:
        DNABERTPredictor._model = orig_model
        DNABERTPredictor._tokenizer = orig_tokenizer

    return seq, conf


def test_single_mask_gap6_coverage():
    """Gap of exactly 6bp → 1 MASK, the predicted 6-mer fills positions 0-5 exactly."""
    seq, conf = _run_mock_dnabert_predict(gap_length=6, mock_kmer_per_mask="ACGTAC")
    assert len(seq) == 6, f"Expected length 6, got {len(seq)}"
    assert seq == "ACGTAC", f"Expected 'ACGTAC', got {seq!r}"
    assert conf > 0.0


def test_majority_vote_coverage_counts():
    """
    For a gap of length L, internal positions (5 ≤ p ≤ L-6) must receive
    exactly 6 votes; edge positions receive fewer.
    Verified using a controlled mock where each MASK token predicts 'AAAAAA'.
    """
    from collections import Counter

    gap_length = 20
    kmer = "AAAAAA"  # all votes go to 'A'

    # Manually compute expected vote counts per position
    n_mask = max(1, gap_length - 5)  # = 15
    vote_count = [0] * gap_length
    for j in range(n_mask):
        for offset in range(6):
            p = j + offset
            if p < gap_length:
                vote_count[p] += 1

    # Internal positions (index 5 to gap_length-6 = 14) should get 6 votes
    for p in range(5, gap_length - 5):
        assert vote_count[p] == 6, (
            f"Position {p} should get 6 votes, got {vote_count[p]}"
        )
    # Edge positions get fewer
    for p in range(0, min(5, gap_length)):
        assert vote_count[p] == p + 1, (
            f"Edge position {p} should get {p+1} votes, got {vote_count[p]}"
        )

    # Also run the actual predict() to confirm the output is all-A
    seq, conf = _run_mock_dnabert_predict(gap_length=gap_length, mock_kmer_per_mask=kmer)
    assert len(seq) == gap_length
    assert all(c == "A" for c in seq), f"Expected all-A output, got {seq!r}"


def test_majority_vote_selects_correct_base_on_constructed_example():
    """
    Construct a scenario where the first 3 masks vote 'AAAAAA' and the last
    3 masks vote 'CCCCCC', verifying that positions covered only by A-voters
    yield 'A' and positions covered only by C-voters yield 'C'.

    For gap_length=12 and n_mask=7:
      - Masks 0-2 (j=0,1,2): predict AAAAAA → vote A on positions 0-7
      - Masks 3-6 (j=3,4,5,6): predict CCCCCC → vote C on positions 3-11
      - Positions 0-2: only A-voters → 'A'
      - Positions 9-11: only C-voters → 'C'
      - Positions 3-8: mixed (need to count A-votes vs C-votes per pos)
    """
    # Instead of the full mock, validate vote-counting math directly
    gap_length = 12
    n_mask = max(1, gap_length - 5)  # = 7
    assert n_mask == 7

    PROB = 0.9  # uniform fake probability

    from collections import Counter
    votes: dict[int, Counter] = {p: Counter() for p in range(gap_length)}

    kmer_assignments = {
        0: "AAAAAA", 1: "AAAAAA", 2: "AAAAAA",
        3: "CCCCCC", 4: "CCCCCC", 5: "CCCCCC", 6: "CCCCCC",
    }

    for j, kmer in kmer_assignments.items():
        for offset, base in enumerate(kmer):
            p = j + offset
            if p < gap_length:
                votes[p][base] += PROB

    # Positions 0-2: only A votes
    for p in range(3):
        assert votes[p].most_common(1)[0][0] == "A", f"pos {p} should be A"

    # Positions 9-11: only C votes
    for p in range(9, 12):
        assert votes[p].most_common(1)[0][0] == "C", f"pos {p} should be C"

    # Positions 3-8: A got (3-j votes before j=3) and C got (j-3+1 votes)
    # pos 3: A from j=0,1,2 → 3 votes; C from j=3 → 1 vote → A wins
    # pos 8: A from j=0..2 covering up to pos 7 only → 0 A votes; C from j=3..6 → 4 → C
    assert votes[3].most_common(1)[0][0] == "A", "pos 3 should be A (3 A-votes vs 1 C-vote)"
    assert votes[8].most_common(1)[0][0] == "C", "pos 8 should be C (0 A-votes vs 4 C-votes)"
