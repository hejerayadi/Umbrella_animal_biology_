"""
Seq2Seq custom predictor for DNA gap reconstruction (Post-Sprint 1).

Implements a lightweight encoder-decoder Transformer trained from scratch on
the masking pipeline produced by ``windowing.py``.  The model is only active
when a local checkpoint file exists at ``CHECKPOINT_PATH``; otherwise
``_seq2seq_available()`` returns ``False`` and the router falls through to
AzureGPT.

Architecture
------------
- Vocabulary  : {A, C, G, T, N, <BOS>, <EOS>, <PAD>}  — 8 tokens
- Encoder     : 2 TransformerEncoderLayer(d_model=256, nhead=4)
- Decoder     : 2 TransformerDecoderLayer(d_model=256, nhead=4)
- Projection  : Linear(256 → 8) with log-softmax
- Positional  : sinusoidal (no extra params)
- Decoding    : beam search (beam=4) with greedy fallback on OOM

Requirements covered: 4.1, 4.2, 4.3, 4.4, 4.5
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
import torch.nn.functional as F

from .utils import (
    FlanksWindow,
    _get_free_vram_gb,
    _normalize_length,
    _pick_dtype,
    _resolve_device,
)

if TYPE_CHECKING:
    from ...schema import GapRegion

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Vocabulary
# ---------------------------------------------------------------------------

VOCAB: dict[str, int] = {
    "A": 0,
    "C": 1,
    "G": 2,
    "T": 3,
    "N": 4,
    "<BOS>": 5,
    "<EOS>": 6,
    "<PAD>": 7,
}
ID_TO_BASE: dict[int, str] = {v: k for k, v in VOCAB.items()}
VOCAB_SIZE = len(VOCAB)

BOS_ID = VOCAB["<BOS>"]
EOS_ID = VOCAB["<EOS>"]
PAD_ID = VOCAB["<PAD>"]

# ---------------------------------------------------------------------------
# Sinusoidal positional encoding
# ---------------------------------------------------------------------------


class _SinusoidalPE(nn.Module):
    """Sinusoidal positional encoding (no learned parameters)."""

    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        # shape: (1, max_len, d_model)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to *x* (batch, seq, d_model)."""
        return x + self.pe[:, : x.size(1)]  # type: ignore[index]


# ---------------------------------------------------------------------------
# Seq2Seq model
# ---------------------------------------------------------------------------


class Seq2SeqModel(nn.Module):
    """Lightweight encoder-decoder Transformer for DNA gap filling.

    Parameters
    ----------
    vocab_size : int
        Size of the shared vocabulary (default 8).
    d_model : int
        Model dimension (default 256).
    nhead : int
        Number of attention heads (default 4).
    num_encoder_layers : int
        Encoder depth (default 2).
    num_decoder_layers : int
        Decoder depth (default 2).
    dim_feedforward : int
        FFN hidden size (default 512).
    dropout : float
        Dropout probability (default 0.1).
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model: int = 256,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        self.src_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.tgt_embed = nn.Embedding(vocab_size, d_model, padding_idx=PAD_ID)
        self.pos_enc = _SinusoidalPE(d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)

        self.proj = nn.Linear(d_model, vocab_size)

    def encode(
        self,
        src: torch.Tensor,
        src_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Encode source tokens → memory (batch, src_len, d_model)."""
        x = self.pos_enc(self.src_embed(src) * math.sqrt(self.d_model))
        return self.encoder(x, src_key_padding_mask=src_key_padding_mask)

    def decode_step(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Single decoder forward → log-probs (batch, tgt_len, vocab_size)."""
        x = self.pos_enc(self.tgt_embed(tgt) * math.sqrt(self.d_model))
        out = self.decoder(
            x,
            memory,
            tgt_mask=tgt_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )
        return F.log_softmax(self.proj(out), dim=-1)


# ---------------------------------------------------------------------------
# Seq2SeqPredictor
# ---------------------------------------------------------------------------

# Default local checkpoint path (populated after training)
_MODULE_DIR = Path(__file__).parent
_DEFAULT_CHECKPOINT = _MODULE_DIR / "seq2seq_checkpoint.pt"


class Seq2SeqPredictor:
    """Zero-shot predictor backed by a locally trained ``Seq2SeqModel``.

    The checkpoint is looked up at ``CHECKPOINT_PATH`` (configurable at
    class level).  When absent, ``_seq2seq_available()`` returns ``False``
    and the cascade router skips this model silently.

    Decoding
    --------
    Beam search (beam=4) is attempted first.  If a
    ``torch.cuda.OutOfMemoryError`` is raised during beam expansion, the
    predictor falls back to greedy decoding automatically.

    ConfidenceScore
    ---------------
    Mean of per-token log-probabilities (normalised to [0.0, 1.0] via
    ``sigmoid`` of the raw mean log-prob value).

    Requirements: 4.1, 4.2, 4.3, 4.4, 4.5
    """

    CHECKPOINT_PATH: Path = _DEFAULT_CHECKPOINT
    BEAM_SIZE: int = 4
    # VRAM threshold under which we skip beam search and go greedy
    _BEAM_VRAM_MIN_GB: float = 4.0

    # Class-level singleton cache (Req 4.3)
    _model: "Seq2SeqModel | None" = None

    # ------------------------------------------------------------------
    # Availability check (Req 4.5)
    # ------------------------------------------------------------------

    @classmethod
    def _seq2seq_available(cls) -> bool:
        """Return ``True`` iff a local checkpoint file exists.

        Used by ``DNAModelRouter._seq2seq_available()`` to decide whether
        to include Seq2Seq in the cascade.
        """
        return cls.CHECKPOINT_PATH.exists()

    # ------------------------------------------------------------------
    # Model loading (Req 4.3)
    # ------------------------------------------------------------------

    @classmethod
    def _load(cls, device: str) -> None:
        """Load the trained checkpoint into *device* if not cached.

        Raises
        ------
        FileNotFoundError
            When ``CHECKPOINT_PATH`` does not exist.
        """
        if cls._model is not None:
            return  # singleton cache hit

        if not cls.CHECKPOINT_PATH.exists():
            raise FileNotFoundError(
                f"Seq2Seq checkpoint not found at {cls.CHECKPOINT_PATH}. "
                "Train the model first (Post-Sprint 1)."
            )

        dtype = _pick_dtype(device, vram_budget_gb=4.0)
        logger.debug(
            "Seq2SeqPredictor: loading checkpoint from %s (device=%s, dtype=%s)",
            cls.CHECKPOINT_PATH,
            device,
            dtype,
        )

        model = Seq2SeqModel()
        state = torch.load(cls.CHECKPOINT_PATH, map_location=device)
        model.load_state_dict(state)
        model.to(device=device, dtype=dtype)
        model.eval()
        cls._model = model

        logger.debug("Seq2SeqPredictor: checkpoint loaded successfully.")

    # ------------------------------------------------------------------
    # Sequence encoding helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _encode_seq(seq: str) -> list[int]:
        """Convert a DNA string to a list of vocabulary token IDs."""
        return [VOCAB.get(base.upper(), VOCAB["N"]) for base in seq]

    @staticmethod
    def _decode_ids(ids: list[int]) -> str:
        """Convert token IDs to a DNA string (skips special tokens)."""
        special = {BOS_ID, EOS_ID, PAD_ID}
        bases = []
        for tid in ids:
            if tid in special:
                continue
            base = ID_TO_BASE.get(tid, "N")
            if base.startswith("<"):
                bases.append("N")
            else:
                bases.append(base)
        return "".join(bases)

    # ------------------------------------------------------------------
    # Greedy decoding
    # ------------------------------------------------------------------

    def _greedy_decode(
        self,
        memory: torch.Tensor,
        target_length: int,
        device: str,
    ) -> tuple[list[int], float]:
        """Greedy autoregressive decoding up to *target_length* tokens.

        Returns
        -------
        tuple[list[int], float]
            ``(token_ids, mean_log_prob)``
        """
        generated: list[int] = []
        log_probs: list[float] = []
        tgt = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)

        model = self.__class__._model
        assert model is not None

        with torch.no_grad():
            for _ in range(target_length):
                tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                    tgt.size(1), device=device
                )
                log_prob_dist = model.decode_step(tgt, memory, tgt_mask=tgt_mask)
                next_log_probs = log_prob_dist[0, -1, :]  # (vocab_size,)
                next_id = int(next_log_probs.argmax().item())
                if next_id == EOS_ID:
                    break
                generated.append(next_id)
                log_probs.append(float(next_log_probs[next_id].item()))
                tgt = torch.cat(
                    [tgt, torch.tensor([[next_id]], dtype=torch.long, device=device)],
                    dim=1,
                )

        mean_lp = sum(log_probs) / len(log_probs) if log_probs else -10.0
        return generated, mean_lp

    # ------------------------------------------------------------------
    # Beam search decoding
    # ------------------------------------------------------------------

    def _beam_decode(
        self,
        memory: torch.Tensor,
        target_length: int,
        device: str,
        beam_size: int = 4,
    ) -> tuple[list[int], float]:
        """Beam search decoding.

        Each beam is a ``(score, token_ids)`` pair where ``score`` is the
        cumulative log-probability.

        Returns
        -------
        tuple[list[int], float]
            Best beam ``(token_ids, mean_log_prob)``.
        """
        model = self.__class__._model
        assert model is not None

        # Beams: list of (cumulative_log_prob, [token_ids])
        beams: list[tuple[float, list[int]]] = [(0.0, [BOS_ID])]
        completed: list[tuple[float, list[int]]] = []

        with torch.no_grad():
            for step in range(target_length):
                candidates: list[tuple[float, list[int]]] = []
                for score, ids in beams:
                    tgt = torch.tensor([ids], dtype=torch.long, device=device)
                    tgt_mask = nn.Transformer.generate_square_subsequent_mask(
                        tgt.size(1), device=device
                    )
                    log_prob_dist = model.decode_step(tgt, memory, tgt_mask=tgt_mask)
                    next_lp = log_prob_dist[0, -1, :]  # (vocab_size,)

                    # Top beam_size tokens
                    topk_vals, topk_ids = torch.topk(next_lp, k=min(beam_size, VOCAB_SIZE))
                    for lp_val, tok_id in zip(topk_vals.tolist(), topk_ids.tolist()):
                        if tok_id == EOS_ID:
                            completed.append((score + lp_val, ids[1:]))  # drop BOS
                        else:
                            candidates.append((score + lp_val, ids + [tok_id]))

                if not candidates:
                    break

                # Keep top beam_size candidates
                candidates.sort(key=lambda x: x[0], reverse=True)
                beams = candidates[:beam_size]

        # Merge completed + active beams; pick best
        all_beams = completed + [(sc, ids[1:]) for sc, ids in beams]  # drop BOS
        if not all_beams:
            return [], -10.0

        best_score, best_ids = max(all_beams, key=lambda x: x[0])
        length = max(len(best_ids), 1)
        mean_lp = best_score / length
        return best_ids, mean_lp

    # ------------------------------------------------------------------
    # Public predict
    # ------------------------------------------------------------------

    def predict(
        self,
        flanks: FlanksWindow,
        gap: "GapRegion",
    ) -> tuple[str, float]:
        """Predict the sequence filling *gap* from *flanks*.

        Parameters
        ----------
        flanks : FlanksWindow
            Flanking context (left + right).
        gap : GapRegion
            Gap region with ``length`` field driving target length.

        Returns
        -------
        tuple[str, float]
            ``(predicted_sequence, confidence)`` where
            ``len(predicted_sequence) == gap.length`` and
            ``confidence ∈ [0.0, 1.0]``.

        Raises
        ------
        torch.cuda.OutOfMemoryError
            Re-raised after ``empty_cache()`` when greedy fallback also OOMs.
        """
        device = _resolve_device()
        self._load(device)

        model = self.__class__._model
        assert model is not None

        # ----------------------------------------------------------------
        # Encode context: left_context + right_context as source tokens
        # ----------------------------------------------------------------
        src_ids = self._encode_seq(flanks.left_context + flanks.right_context)
        src_tensor = torch.tensor([src_ids], dtype=torch.long, device=device)

        t_start = time.perf_counter()
        try:
            with torch.no_grad():
                memory = model.encode(src_tensor)

            # Beam search or greedy depending on available VRAM
            use_beam = _get_free_vram_gb() >= self._BEAM_VRAM_MIN_GB
            try:
                if use_beam:
                    token_ids, mean_lp = self._beam_decode(
                        memory, gap.length, device, beam_size=self.BEAM_SIZE
                    )
                else:
                    token_ids, mean_lp = self._greedy_decode(memory, gap.length, device)
            except torch.cuda.OutOfMemoryError:
                # Beam OOM → fall back to greedy (Req 4.2)
                logger.warning(
                    "Seq2SeqPredictor: beam search OOM for gap length=%d, "
                    "falling back to greedy decoding.",
                    gap.length,
                )
                torch.cuda.empty_cache()
                token_ids, mean_lp = self._greedy_decode(memory, gap.length, device)

        except torch.cuda.OutOfMemoryError as oom:
            logger.error(
                "Seq2SeqPredictor: OOM during inference for gap length=%d: %s",
                gap.length,
                oom,
            )
            torch.cuda.empty_cache()
            raise

        elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # ----------------------------------------------------------------
        # ConfidenceScore: sigmoid of mean log-prob → [0.0, 1.0]  (Req 4.4)
        # ----------------------------------------------------------------
        import math as _math
        confidence = float(1.0 / (1.0 + _math.exp(-mean_lp)))
        confidence = max(0.0, min(1.0, confidence))

        # ----------------------------------------------------------------
        # Decode and normalise length (Req 6.5)
        # ----------------------------------------------------------------
        raw_seq = self._decode_ids(token_ids)
        predicted_sequence = _normalize_length(raw_seq, gap.length)

        logger.debug(
            "Seq2SeqPredictor: gap_length=%d confidence=%.4f time_ms=%.1f",
            gap.length,
            confidence,
            elapsed_ms,
        )

        return predicted_sequence, confidence
