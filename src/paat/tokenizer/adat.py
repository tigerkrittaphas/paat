"""
ADAT — iterative LLM-guided vocabulary pruning.

Reference:
    Zheng et al. (2024) "Enhancing Large Language Models through Adaptive
    Tokenizers", NeurIPS 2024.

The loop is:

1. Start from an initial Unigram tokenizer (large, e.g. 32 K).
2. Repeat for N iterations:
   a. Tokenize training / inference corpora with the current vocabulary.
   b. Train a small LLM from scratch on the training corpus.
   c. Compute per-token cross-entropy :math:`\\mathcal{L}_M(x_i)` on the
      inference corpus.
   d. Combine with the Unigram score :math:`\\mathcal{L}_P(x_i)` (taken
      directly from the tokenizer's piece scores) through the balance
      function :math:`F(a, b) = a / (\\lambda \\log(b+1))`.
   e. Apply loss momentum across iterations to stabilise rankings.
   f. Keep the top-K pieces; rebuild a smaller Unigram tokenizer.
3. Stop when the target vocabulary size has been reached.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from paat.model.train import TrainConfig, compute_per_token_ce, train_llm
from paat.model.transformer import build_model
from paat.tokenizer.unigram import (
    SPECIAL_TOKENS,
    build_hf_unigram,
    get_pieces_with_scores,
)


# --------------------------------------------------------------------- config

@dataclass
class ADATConfig:
    """Hyper-parameters for one ADAT run."""
    initial_vocab_size: int = 32_000
    target_vocab_size: int = 16_000
    n_iterations: int = 3
    seq_len: int = 512
    train_tokens_per_iter: int = 5_000_000
    eval_tokens_per_iter: int = 2_000_000

    model_size: str = "tiny"

    # Balance function  F(a, b) = a / (lambda * log(b + 1))
    balance_lambda: float = 1.0

    # Loss momentum (beta * L_{j-1} + L_j)
    momentum: float = 0.9

    train: TrainConfig = field(default_factory=TrainConfig)


@dataclass
class IterationLog:
    iteration: int
    vocab_size_before: int
    vocab_size_after: int
    train_ppl: float
    n_train_tokens: int
    n_eval_tokens: int
    n_tokens_with_loss: int


# --------------------------------------------------------------- tokenisation

def encode_corpus(
    tokenizer: Tokenizer,
    texts: list[str],
    max_tokens: int | None = None,
    batch_size: int = 256,
) -> np.ndarray:
    """Tokenize a corpus and return a flat int32 array of token ids.

    Processes ``texts`` in batches so the intermediate Python list of ints
    never exceeds ``batch_size`` documents at once.  Stops early once
    ``max_tokens`` ids have been collected, avoiding tokenising more of the
    corpus than the training budget requires.
    """
    chunks: list[np.ndarray] = []
    total = 0
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        encodings = tokenizer.encode_batch(batch, add_special_tokens=False)
        chunk = np.array(
            [id_ for enc in encodings for id_ in enc.ids], dtype=np.int32
        )
        if len(chunk) == 0:
            continue
        chunks.append(chunk)
        total += len(chunk)
        if max_tokens is not None and total >= max_tokens:
            break
    if not chunks:
        return np.array([], dtype=np.int32)
    arr = np.concatenate(chunks)
    if max_tokens is not None:
        arr = arr[:max_tokens]
    return arr


# ------------------------------------------------------------------- pruning

def balance_score(
    piece_scores: np.ndarray,
    llm_loss: np.ndarray,
    lam: float,
) -> np.ndarray:
    """Compute :math:`F(\\mathcal{L}_P, \\mathcal{L}_M) = L_P / (\\lambda \\log(L_M+1))`.

    * ``piece_scores``:  Unigram log-probabilities (higher = more useful).
      These are the scores stored by the tokenizer.
    * ``llm_loss``:      Per-token cross-entropy (lower = better predicted).

    Tokens never observed in the inference corpus have ``llm_loss = inf``
    and thus receive a score of 0 — they are ranked at the bottom and
    pruned first.
    """
    denom = lam * np.log(llm_loss + 1.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        score = np.where(denom > 0, piece_scores / denom, -np.inf)
    # Unseen tokens (llm_loss = inf → denom = inf → score = -0) and
    # any other non-finite results should rank at the bottom.
    unseen = ~np.isfinite(llm_loss)
    score[unseen] = -np.inf
    score[~np.isfinite(score)] = -np.inf
    return score

def get_coverage_protected_ids(pieces: list[tuple[str, float]]) -> list[int]:
    """Return ids of pieces that are the sole coverage for at least one Unicode codepoint.

    If these pieces were pruned, that character would have no subword representation
    and would always map to <unk>.  We protect them the same way we protect specials.
    """
    from collections import defaultdict
    # For each Unicode codepoint, collect which piece ids cover it.
    codepoint_to_ids: dict[str, list[int]] = defaultdict(list)
    for i, (piece, _) in enumerate(pieces):
        # Strip SentencePiece metaspace prefix before checking characters.
        surface = piece.lstrip("▁")
        for ch in surface:
            codepoint_to_ids[ch].append(i)

    protected: list[int] = []
    for ch, ids in codepoint_to_ids.items():
        if len(ids) == 1:
            protected.append(ids[0])
    return protected


def select_surviving_pieces(
    pieces: list[tuple[str, float]],
    score: np.ndarray,
    target_size: int,
    protected_ids: list[int],
) -> list[tuple[str, float]]:
    """Keep the ``target_size`` highest-scoring pieces plus special tokens.

    Always preserves the four special tokens at the beginning of the vocab
    to maintain stable ids (``<unk>``=0, ``<s>``=1, ``</s>``=2, ``<pad>``=3).
    """
    # Force specials to the top of the ranking.
    adjusted = score.copy()
    for pid in protected_ids:
        adjusted[pid] = np.inf

    keep_idx = np.argsort(-adjusted)[:target_size]
    keep_set = set(int(i) for i in keep_idx)
    for pid in protected_ids:
        keep_set.add(pid)

    # Preserve original ordering with specials first.
    ordered: list[tuple[str, float]] = [pieces[i] for i in protected_ids]
    for i, piece in enumerate(pieces):
        if i in protected_ids:
            continue
        if i in keep_set:
            ordered.append(piece)
    return ordered


# ------------------------------------------------------------------- main loop

def run_adat(
    initial_tokenizer: Tokenizer,
    train_texts: list[str],
    eval_texts: list[str],
    config: ADATConfig,
    output_dir: Path,
    device: str = "cuda",
    seed: int = 42,
) -> tuple[Tokenizer, list[IterationLog]]:
    """Run the ADAT iterative pruning loop.

    Args:
        initial_tokenizer: Large HF Unigram tokenizer (vocab >= initial_vocab_size).
        train_texts:       Training corpus (list of documents).
        eval_texts:        Held-out corpus used for LLM loss + PPL reporting.
        config:            :class:`ADATConfig`.
        output_dir:        Where to save per-iteration tokenizers + logs.

    Returns:
        The final pruned tokenizer and a list of per-iteration logs.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    tokenizer = initial_tokenizer
    current_vocab = tokenizer.get_vocab_size()
    if current_vocab < config.initial_vocab_size:
        raise ValueError(
            f"Initial tokenizer has {current_vocab} pieces but "
            f"config.initial_vocab_size is {config.initial_vocab_size}."
        )

    # Pruning schedule: linearly decrease from init → target over N iterations.
    sizes = np.linspace(
        config.initial_vocab_size,
        config.target_vocab_size,
        config.n_iterations + 1,
        dtype=int,
    ).tolist()

    logs: list[IterationLog] = []
    # Persistent per-iteration ranking state for momentum.  Scores are keyed
    # by piece string because token ids change after each pruning step.
    momentum_score: dict[str, float] = {}

    special_ids = [tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS]

    for it in range(1, config.n_iterations + 1):
        target_size = sizes[it]
        print(f"\n====== ADAT iteration {it}/{config.n_iterations}  "
              f"({current_vocab} -> {target_size}) ======")

        # Tokenize just enough of each corpus to fill the per-iter token budget.
        print("  [tokenize] encoding train / eval corpora ...")
        train_slice = encode_corpus(
            tokenizer, train_texts, max_tokens=config.train_tokens_per_iter
        )
        eval_slice = encode_corpus(
            tokenizer, eval_texts, max_tokens=config.eval_tokens_per_iter
        )
        print(f"  [tokenize] train: {len(train_slice):,} ids   "
              f"eval: {len(eval_slice):,} ids")

        # Fresh model each iteration — "randomly initialized" per the paper.
        model = build_model(vocab_size=current_vocab, size=config.model_size)
        print(f"  [llm] training {config.model_size} model from scratch on "
              f"{len(train_slice):,} tokens ...")
        model, train_ppl = train_llm(
            model, train_slice, eval_slice, config.seq_len,
            device=device, config=config.train,
        )
        print(f"  [llm] held-out PPL = {train_ppl:.2f}")

        # Per-token LLM cross-entropy on the held-out slice.
        print("  [score] computing per-token CE ...")
        llm_loss = compute_per_token_ce(
            model, eval_slice, config.seq_len, current_vocab,
            batch_size=config.train.batch_size, device=device,
        )
        n_seen = int(np.isfinite(llm_loss).sum())
        print(f"  [score] {n_seen:,}/{current_vocab:,} tokens observed in eval")

        # Piece scores (Unigram log-probs) from the current tokenizer.
        pieces = get_pieces_with_scores(tokenizer)
        piece_scores = np.array([s for _, s in pieces], dtype=np.float64)

        raw_score = balance_score(piece_scores, llm_loss, config.balance_lambda)

        # Apply momentum using piece strings as stable keys.
        smoothed = raw_score.copy()
        for i, (piece, _) in enumerate(pieces):
            prev = momentum_score.get(piece, 0.0)
            smoothed[i] = config.momentum * prev + raw_score[i]
        # Update state for next iteration (survivors only, handled below).

        # Prune to target size.
        coverage_ids = get_coverage_protected_ids(pieces)
        all_protected = list(dict.fromkeys(special_ids + coverage_ids))  # deduplicated, order preserved
        surviving = select_surviving_pieces(
            pieces, smoothed, target_size, all_protected,
        )
        tokenizer = build_hf_unigram(surviving)
        current_vocab = tokenizer.get_vocab_size()

        # Refresh momentum state to cover only the survivors.
        surviving_names = {p for p, _ in surviving}
        new_momentum: dict[str, float] = {}
        for i, (piece, _) in enumerate(pieces):
            if piece in surviving_names and np.isfinite(smoothed[i]):
                new_momentum[piece] = float(smoothed[i])
        momentum_score = new_momentum

        # Save per-iteration tokenizer + log.
        tok_path = output_dir / f"iter_{it:02d}_vocab{current_vocab}.json"
        tokenizer.save(str(tok_path))

        log = IterationLog(
            iteration=it,
            vocab_size_before=len(pieces),
            vocab_size_after=current_vocab,
            train_ppl=train_ppl,
            n_train_tokens=int(len(train_slice)),
            n_eval_tokens=int(len(eval_slice)),
            n_tokens_with_loss=n_seen,
        )
        logs.append(log)
        print(f"  [save] tokenizer → {tok_path}   ppl={train_ppl:.2f}")

    # Persist summary.
    summary = {
        "config": {
            "initial_vocab_size": config.initial_vocab_size,
            "target_vocab_size": config.target_vocab_size,
            "n_iterations": config.n_iterations,
            "model_size": config.model_size,
            "balance_lambda": config.balance_lambda,
            "momentum": config.momentum,
        },
        "iterations": [log.__dict__ for log in logs],
    }
    (output_dir / "adat_log.json").write_text(json.dumps(summary, indent=2))

    # Save final tokenizer in a stable location.
    final_path = output_dir / "tokenizer.json"
    tokenizer.save(str(final_path))
    print(f"\n[done] final ADAT tokenizer → {final_path}")

    return tokenizer, logs
