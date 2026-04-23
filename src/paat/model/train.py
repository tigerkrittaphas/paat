"""
Training and inference utilities for the ADAT LLM.

Two routines matter here:

* :func:`train_llm` — standard AdamW training on a packed token stream.
  Returns the trained model plus its held-out perplexity.

* :func:`compute_per_token_ce` — runs the model over a token stream and
  returns the *per-vocabulary-item* average cross-entropy loss.  This is
  the :math:`\\mathcal{L}_M(x_i)` quantity that ADAT uses to decide which
  tokens to prune.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2LMHeadModel


@dataclass
class TrainConfig:
    batch_size: int = 32
    lr: float = 3e-4
    weight_decay: float = 0.01
    warmup_steps: int = 100
    grad_clip: float = 1.0


def _make_dataloader(tokens: np.ndarray, seq_len: int, batch_size: int,
                     shuffle: bool) -> DataLoader:
    """Pack a flat token array into (N, seq_len) and wrap in a DataLoader."""
    n_seqs = len(tokens) // seq_len
    packed = tokens[: n_seqs * seq_len].reshape(n_seqs, seq_len)
    tensor = torch.from_numpy(packed.astype(np.int64))
    ds = TensorDataset(tensor)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=True, drop_last=True)


def train_llm(
    model: GPT2LMHeadModel,
    train_tokens: np.ndarray,
    eval_tokens: np.ndarray,
    seq_len: int,
    device: str = "cuda",
    config: TrainConfig | None = None,
    log_every: int = 50,
) -> tuple[GPT2LMHeadModel, float]:
    """Train an LLM on packed tokens and return (model, eval_ppl)."""
    cfg = config or TrainConfig()
    model = model.to(device)
    model.train()

    optim = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )
    loader = _make_dataloader(train_tokens, seq_len, cfg.batch_size, shuffle=True)
    total_steps = len(loader)

    def lr_schedule(step: int) -> float:
        if step < cfg.warmup_steps:
            return step / max(1, cfg.warmup_steps)
        progress = (step - cfg.warmup_steps) / max(1, total_steps - cfg.warmup_steps)
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_schedule)

    step = 0
    loss_ema = None
    for (batch,) in loader:
        batch = batch.to(device, non_blocking=True)
        optim.zero_grad(set_to_none=True)
        out = model(input_ids=batch, labels=batch)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()
        scheduler.step()

        loss_val = loss.item()
        loss_ema = loss_val if loss_ema is None else 0.9 * loss_ema + 0.1 * loss_val
        step += 1
        if step % log_every == 0:
            print(f"  step {step}/{total_steps}  loss {loss_ema:.3f}  "
                  f"lr {scheduler.get_last_lr()[0]:.2e}")

    eval_ppl = evaluate_perplexity(model, eval_tokens, seq_len, cfg.batch_size, device)
    return model, eval_ppl


@torch.no_grad()
def evaluate_perplexity(
    model: GPT2LMHeadModel,
    tokens: np.ndarray,
    seq_len: int,
    batch_size: int,
    device: str = "cuda",
) -> float:
    """Compute held-out perplexity on a packed token stream."""
    model.eval()
    loader = _make_dataloader(tokens, seq_len, batch_size, shuffle=False)
    total_nll = 0.0
    total_count = 0
    for (batch,) in loader:
        batch = batch.to(device, non_blocking=True)
        out = model(input_ids=batch, labels=batch)
        n_toks = batch.numel() - batch.size(0)  # shift-by-one loses 1 token per row
        total_nll += out.loss.item() * n_toks
        total_count += n_toks
    return float(math.exp(total_nll / max(1, total_count)))


@torch.no_grad()
def compute_per_token_ce(
    model: GPT2LMHeadModel,
    tokens: np.ndarray,
    seq_len: int,
    vocab_size: int,
    batch_size: int = 32,
    device: str = "cuda",
) -> np.ndarray:
    """Average cross-entropy loss per vocabulary item (:math:`\\mathcal{L}_M`).

    For every position in the token stream we compute the CE of predicting
    the *next* token, then group by that next-token id and average.  Token
    ids that never appear as a target receive ``+inf`` (they contribute no
    information to the pruning score and will be ranked as low-value).

    Returns:
        Array of shape ``(vocab_size,)`` with the mean CE per token id.
    """
    model.eval()
    loader = _make_dataloader(tokens, seq_len, batch_size, shuffle=False)

    loss_sum = torch.zeros(vocab_size, dtype=torch.float64, device=device)
    loss_cnt = torch.zeros(vocab_size, dtype=torch.float64, device=device)

    for (batch,) in loader:
        batch = batch.to(device, non_blocking=True)
        # Logits for positions 0..T-2 predict positions 1..T-1.
        logits = model(input_ids=batch).logits[:, :-1, :]
        targets = batch[:, 1:]

        # Per-position CE without reduction: (B, T-1)
        ce = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            targets.reshape(-1),
            reduction="none",
        )
        tgt_flat = targets.reshape(-1)
        loss_sum.index_add_(0, tgt_flat, ce.to(torch.float64))
        loss_cnt.index_add_(
            0, tgt_flat, torch.ones_like(ce, dtype=torch.float64)
        )

    mean = (loss_sum / loss_cnt.clamp(min=1)).cpu().numpy()
    # Tokens never seen get +inf so they will be ranked low after scoring.
    mean[loss_cnt.cpu().numpy() == 0] = np.inf
    return mean
