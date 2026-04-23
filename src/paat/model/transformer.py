"""
Tiny GPT-style decoder transformer for ADAT demo.

Uses HuggingFace ``transformers`` GPT-2 implementation with a small config
(~3 M parameters) so that a full training iteration fits in minutes on a
single A6000.  Larger configurations are easy to request via
``build_model(size="small")`` etc.
"""

from __future__ import annotations

from dataclasses import dataclass

from transformers import GPT2Config, GPT2LMHeadModel


@dataclass(frozen=True)
class ModelSize:
    n_layer: int
    n_embd: int
    n_head: int
    n_positions: int = 512

    @property
    def approx_params(self) -> int:
        """Rough parameter count (excluding embeddings)."""
        # 12 * n_embd^2 per transformer block is the usual rule of thumb.
        return 12 * self.n_embd * self.n_embd * self.n_layer


SIZES: dict[str, ModelSize] = {
    "tiny":   ModelSize(n_layer=4, n_embd=256, n_head=4),   # ~3 M (no emb)
    "small":  ModelSize(n_layer=6, n_embd=384, n_head=6),   # ~10 M
    "medium": ModelSize(n_layer=12, n_embd=768, n_head=12), # ~85 M (Pythia-70M-ish)
}


def build_model(vocab_size: int, size: str = "tiny") -> GPT2LMHeadModel:
    """Instantiate a GPT-2 model with the given vocabulary size.

    Args:
        vocab_size: Size of the tokenizer vocabulary.
        size:       Key into :data:`SIZES`.
    """
    if size not in SIZES:
        raise ValueError(f"Unknown size '{size}'.  Valid: {sorted(SIZES)}")
    s = SIZES[size]
    config = GPT2Config(
        vocab_size=vocab_size,
        n_positions=s.n_positions,
        n_embd=s.n_embd,
        n_layer=s.n_layer,
        n_head=s.n_head,
        # Lightweight defaults for a training-from-scratch demo.
        attn_pdrop=0.0,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
    )
    return GPT2LMHeadModel(config)
