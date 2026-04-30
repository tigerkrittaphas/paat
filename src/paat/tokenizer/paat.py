"""
PAAT — Parity-Aware Adaptive Tokenizer.

Extends ADAT (Zheng et al. 2024) with a cross-lingual fairness term in the
pruning score.  The base ADAT balance function ranks pieces by

    F(L_P, L_M) = L_P / (lambda * log(L_M + 1))

where L_P is the Unigram log-probability of the piece and L_M is the
LLM's per-token cross-entropy.  PAAT adds a third signal: each piece is
weighted by how much it *serves* badly-tokenized languages, measured via
per-language tokens-per-byte (the headline metric of Foroutan et al. 2025
"Parity-Aware Byte-Pair Encoding").  The combined ranking is

    F_paat(i) = F(i) + alpha * (parity_weight(i) - 1)

* parity_weight(i) ≈ 1  for pieces used uniformly across languages.
* parity_weight(i) > 1  for pieces concentrated in high-tokens-per-byte
                        (under-served) languages.
* parity_weight(i) < 1  for pieces concentrated in low-tokens-per-byte
                        (over-served) languages.

When ``alpha = 0`` the algorithm reduces exactly to ADAT.

Reuses :mod:`paat.tokenizer.adat` for the shared pieces (corpus encoding,
balance function, coverage-protection, vocabulary selection).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from tokenizers import Tokenizer

from paat.model.train import TrainConfig, compute_per_token_ce, train_llm
from paat.model.transformer import build_model
from paat.tokenizer.adat import (
    IterationLog,
    balance_score,
    encode_corpus,
    get_coverage_protected_ids,
    select_surviving_pieces,
)
from paat.tokenizer.unigram import (
    SPECIAL_TOKENS,
    build_hf_unigram,
    get_pieces_with_scores,
)


# --------------------------------------------------------------------- config

@dataclass
class PAATConfig:
    """Hyper-parameters for one PAAT run.

    A drop-in superset of :class:`paat.tokenizer.adat.ADATConfig` plus the
    ``parity_alpha`` knob that activates the cross-lingual fairness bonus.
    """
    initial_vocab_size: int = 32_000
    target_vocab_size: int = 16_000
    n_iterations: int = 3
    seq_len: int = 512
    train_tokens_per_iter: int = 5_000_000
    eval_tokens_per_iter: int = 2_000_000

    model_size: str = "tiny"

    # ADAT balance function  F(a, b) = a / (lambda * log(b + 1))
    balance_lambda: float = 1.0

    # Loss momentum (beta * L_{j-1} + L_j)
    momentum: float = 0.9

    # Parity-aware bonus weight.  0 reduces PAAT to plain ADAT.  Sensible
    # range: 0.5 – 5.0; tune against the Gini-of-tokens-per-byte parity
    # metric on a held-out FLORES+ slice.
    parity_alpha: float = 1.0

    train: TrainConfig = field(default_factory=TrainConfig)


# ----------------------------------------------------------- parity weighting

def compute_parity_weights(
    tokenizer: Tokenizer,
    parity_texts: dict[str, list[str]],
) -> tuple[np.ndarray, dict[str, float]]:
    """Per-piece parity weight from per-language FLORES+ samples.

    The weight is the usage-share-weighted relative tokens-per-byte across
    languages: pieces predominantly used in badly-served (high tpb)
    languages receive weights > 1, while pieces used in over-served
    languages get weights < 1.

    Args:
        tokenizer:     Current HF tokenizer; ids are taken from its vocab.
        parity_texts:  Mapping ``{lang_code: list[parallel_sentences]}``.
                       Sentences should be parallel across languages
                       (FLORES+ devtest is the natural fit) so per-language
                       ``tokens_per_byte`` is comparable.

    Returns:
        ``(weights, per_lang_tpb)`` where ``weights`` has shape
        ``(vocab_size,)`` and is centred around 1.0, and ``per_lang_tpb``
        maps language code to its tokens-per-byte under the current
        tokenizer (useful for logging).
    """
    vocab_size = tokenizer.get_vocab_size()
    langs = [l for l, s in parity_texts.items() if s]
    n_langs = len(langs)
    if n_langs == 0:
        return np.ones(vocab_size, dtype=np.float64), {}

    freq = np.zeros((vocab_size, n_langs), dtype=np.float64)
    n_bytes_per_lang = np.zeros(n_langs, dtype=np.float64)

    for li, lang in enumerate(langs):
        sentences = parity_texts[lang]
        encs = tokenizer.encode_batch(sentences, add_special_tokens=False)
        for enc in encs:
            ids = enc.ids
            if not ids:
                continue
            # vectorised increment
            np.add.at(freq[:, li], ids, 1.0)
        n_bytes_per_lang[li] = sum(len(s.encode("utf-8")) for s in sentences)

    n_tokens_per_lang = freq.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        tpb = np.where(
            n_bytes_per_lang > 0,
            n_tokens_per_lang / n_bytes_per_lang,
            0.0,
        )
    valid = tpb > 0
    if not valid.any():
        return np.ones(vocab_size, dtype=np.float64), {}

    mean_tpb = float(tpb[valid].mean())
    lang_penalty = np.where(valid, tpb / mean_tpb, 1.0)

    piece_total = freq.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(piece_total > 0, freq / piece_total, 0.0)
    weights = share @ lang_penalty                             # (vocab_size,)

    # Pieces never seen in the parity corpus get neutral weight (no bias).
    unseen = piece_total.squeeze(-1) == 0
    weights[unseen] = 1.0

    per_lang_tpb = {lang: float(tpb[i]) for i, lang in enumerate(langs)}
    return weights, per_lang_tpb


def parity_aware_score(
    piece_scores: np.ndarray,
    llm_loss: np.ndarray,
    parity_weights: np.ndarray,
    balance_lambda: float,
    parity_alpha: float,
) -> np.ndarray:
    """ADAT balance score plus an additive parity bonus.

    The bonus is centred around 0 (``parity_weight - 1``), so a piece used
    uniformly across languages gets the same score it would in plain ADAT.
    Unseen pieces (LLM never observed them) keep ``-inf`` from the base
    balance score and are still pruned first.
    """
    base = balance_score(piece_scores, llm_loss, balance_lambda)
    if parity_alpha == 0.0:
        return base
    bonus = parity_alpha * (parity_weights - 1.0)
    finite = np.isfinite(base)
    out = base.copy()
    out[finite] = base[finite] + bonus[finite]
    return out


# ------------------------------------------------------------------- main loop

def run_paat(
    initial_tokenizer: Tokenizer,
    train_texts: list[str],
    eval_texts: list[str],
    parity_texts: dict[str, list[str]],
    config: PAATConfig,
    output_dir: Path,
    device: str = "cuda",
    seed: int = 42,
) -> tuple[Tokenizer, list[IterationLog]]:
    """Run the parity-aware adaptive tokenizer pruning loop.

    Identical in shape to :func:`paat.tokenizer.adat.run_adat`, with one
    extra step per iteration: encode ``parity_texts`` (per-language
    parallel sentences), compute per-piece parity weights, and add
    ``config.parity_alpha * (weight - 1)`` to the balance score before
    pruning.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    np.random.default_rng(seed)  # for any future stochastic step

    tokenizer = initial_tokenizer
    current_vocab = tokenizer.get_vocab_size()
    if current_vocab < config.initial_vocab_size:
        raise ValueError(
            f"Initial tokenizer has {current_vocab} pieces but "
            f"config.initial_vocab_size is {config.initial_vocab_size}."
        )
    if not parity_texts and config.parity_alpha != 0.0:
        raise ValueError(
            "config.parity_alpha != 0 but no parity_texts provided."
        )

    sizes = np.linspace(
        config.initial_vocab_size,
        config.target_vocab_size,
        config.n_iterations + 1,
        dtype=int,
    ).tolist()

    logs: list[IterationLog] = []
    momentum_score: dict[str, float] = {}
    special_ids = [tokenizer.token_to_id(tok) for tok in SPECIAL_TOKENS]

    for it in range(1, config.n_iterations + 1):
        target_size = sizes[it]
        print(f"\n====== PAAT iteration {it}/{config.n_iterations}  "
              f"({current_vocab} -> {target_size}) ======")

        print("  [tokenize] encoding train / eval corpora ...")
        train_slice = encode_corpus(
            tokenizer, train_texts, max_tokens=config.train_tokens_per_iter
        )
        eval_slice = encode_corpus(
            tokenizer, eval_texts, max_tokens=config.eval_tokens_per_iter
        )
        print(f"  [tokenize] train: {len(train_slice):,} ids   "
              f"eval: {len(eval_slice):,} ids")

        model = build_model(vocab_size=current_vocab, size=config.model_size)
        print(f"  [llm] training {config.model_size} model from scratch on "
              f"{len(train_slice):,} tokens ...")
        model, train_ppl = train_llm(
            model, train_slice, eval_slice, config.seq_len,
            device=device, config=config.train,
        )
        print(f"  [llm] held-out PPL = {train_ppl:.2f}")

        print("  [score] computing per-token CE ...")
        llm_loss = compute_per_token_ce(
            model, eval_slice, config.seq_len, current_vocab,
            batch_size=config.train.batch_size, device=device,
        )
        n_seen = int(np.isfinite(llm_loss).sum())
        print(f"  [score] {n_seen:,}/{current_vocab:,} tokens observed in eval")

        pieces = get_pieces_with_scores(tokenizer)
        piece_scores = np.array([s for _, s in pieces], dtype=np.float64)

        # ── Parity weighting ──────────────────────────────────────────────
        if config.parity_alpha != 0.0 and parity_texts:
            print("  [parity] computing per-language tokens-per-byte ...")
            parity_weights, per_lang_tpb = compute_parity_weights(
                tokenizer, parity_texts
            )
            tpb_vals = list(per_lang_tpb.values())
            if tpb_vals:
                print(f"  [parity] tpb min={min(tpb_vals):.4f}  "
                      f"max={max(tpb_vals):.4f}  "
                      f"mean={float(np.mean(tpb_vals)):.4f}  "
                      f"alpha={config.parity_alpha}")
        else:
            parity_weights = np.ones(current_vocab, dtype=np.float64)
            per_lang_tpb = {}

        raw_score = parity_aware_score(
            piece_scores, llm_loss,
            parity_weights, config.balance_lambda, config.parity_alpha,
        )

        # Momentum smoothing (piece string keyed → survives id changes).
        smoothed = raw_score.copy()
        for i, (piece, _) in enumerate(pieces):
            prev = momentum_score.get(piece, 0.0)
            smoothed[i] = config.momentum * prev + raw_score[i]

        coverage_ids = get_coverage_protected_ids(pieces)
        all_protected = list(dict.fromkeys(special_ids + coverage_ids))
        surviving = select_surviving_pieces(
            pieces, smoothed, target_size, all_protected,
        )
        tokenizer = build_hf_unigram(surviving)
        current_vocab = tokenizer.get_vocab_size()

        surviving_names = {p for p, _ in surviving}
        new_momentum: dict[str, float] = {}
        for i, (piece, _) in enumerate(pieces):
            if piece in surviving_names and np.isfinite(smoothed[i]):
                new_momentum[piece] = float(smoothed[i])
        momentum_score = new_momentum

        tok_path = output_dir / f"iter_{it:02d}_vocab{current_vocab}.json"
        tokenizer.save(str(tok_path))

        # Persist per-iteration parity diagnostics.
        if per_lang_tpb:
            (output_dir / f"iter_{it:02d}_parity.json").write_text(
                json.dumps(per_lang_tpb, indent=2, sort_keys=True)
            )

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

    summary = {
        "config": {
            "initial_vocab_size": config.initial_vocab_size,
            "target_vocab_size": config.target_vocab_size,
            "n_iterations": config.n_iterations,
            "model_size": config.model_size,
            "balance_lambda": config.balance_lambda,
            "momentum": config.momentum,
            "parity_alpha": config.parity_alpha,
        },
        "iterations": [log.__dict__ for log in logs],
    }
    (output_dir / "paat_log.json").write_text(json.dumps(summary, indent=2))

    final_path = output_dir / "tokenizer.json"
    tokenizer.save(str(final_path))
    print(f"\n[done] final PAAT tokenizer → {final_path}")

    return tokenizer, logs
