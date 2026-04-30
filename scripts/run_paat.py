"""
Run PAAT (Parity-Aware Adaptive Tokenizer) end-to-end.

Mirrors :mod:`scripts.run_adat` but uses the parity-aware pruning loop in
:mod:`paat.tokenizer.paat`.  Per-piece scores combine ADAT's ``L_P / log(L_M+1)``
balance with an additive bonus weighted by per-language tokens-per-byte
(measured on FLORES+).

Usage:
    python scripts/run_paat.py \\
        --data-dir data/raw/mc4 \\
        --flores-dir data/raw/flores \\
        --output-dir models/tokenizers/paat_phase1 \\
        --initial-vocab 32000 --target-vocab 16000 --iterations 3 \\
        --parity-alpha 1.0
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from paat.data.languages import ALL_LANGUAGES
from paat.model.train import TrainConfig
from paat.tokenizer.adat import encode_corpus
from paat.tokenizer.paat import PAATConfig, run_paat
from paat.model.transformer import build_model
from paat.model.train import train_llm
from paat.tokenizer.unigram import (
    sentencepiece_to_hf_unigram,
    train_unigram_sentencepiece,
)


def load_mc4_texts(data_dir: Path, langs: list[str], total_docs: int,
                   seed: int) -> list[str]:
    """Load mC4 docs proportional to MC4_NATURAL_COUNTS."""
    from paat.data.languages import MC4_NATURAL_COUNTS

    rng = random.Random(seed)
    total_natural = sum(MC4_NATURAL_COUNTS.get(l, 0) for l in langs)
    texts: list[str] = []
    for lang in langs:
        path = data_dir / f"{lang}.jsonl"
        if not path.exists():
            continue
        natural = MC4_NATURAL_COUNTS.get(lang, 1)
        n_docs = max(1, round(total_docs * natural / total_natural))
        docs = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    docs.append(json.loads(line)["text"])
                if len(docs) >= n_docs:
                    break
        rng.shuffle(docs)
        texts.extend(docs)
    rng.shuffle(texts)
    return texts


def split_texts(texts: list[str], train_frac: float,
                seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = list(texts)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * train_frac)
    return shuffled[:split], shuffled[split:]


def load_flores_per_lang(
    flores_dir: Path,
    langs: list[str],
    max_sentences: int | None = None,
    split: str = "dev",
) -> dict[str, list[str]]:
    """Load parallel FLORES+ sentences per language for parity weighting.

    Defaults to the ``dev`` split — reserved as the parity-aware *training*
    split.  ``devtest`` is held out for ``eval_parity.py`` so that the
    parity numbers we report are not the same data the bonus was fit on.
    Pass ``split=None`` only for legacy/debug runs that want both splits.
    """
    out: dict[str, list[str]] = {}
    for lang in langs:
        path = flores_dir / f"{lang}.jsonl"
        if not path.exists():
            continue
        sentences: list[str] = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                if split is not None and rec.get("split") != split:
                    continue
                sentences.append(rec["sentence"])
                if max_sentences is not None and len(sentences) >= max_sentences:
                    break
        if sentences:
            out[lang] = sentences
    return out


def eval_tokenizer_ppl(
    tokenizer_path: Path,
    eval_texts: list[str],
    vocab_size: int,
    seq_len: int,
    train_tokens: int,
    eval_tokens: int,
    model_size: str,
    train_cfg: TrainConfig,
    device: str,
) -> float:
    """Train a fresh small LLM on the given tokenizer and return held-out PPL."""
    from tokenizers import Tokenizer as HFTokenizer
    tok = HFTokenizer.from_file(str(tokenizer_path))

    train_split, eval_split = (
        eval_texts[: len(eval_texts) // 2],
        eval_texts[len(eval_texts) // 2:],
    )
    train_ids = encode_corpus(tok, train_split, max_tokens=train_tokens)
    eval_ids = encode_corpus(tok, eval_split, max_tokens=eval_tokens)

    model = build_model(vocab_size=vocab_size, size=model_size)
    model, ppl = train_llm(
        model, train_ids, eval_ids, seq_len,
        device=device, config=train_cfg,
    )
    return ppl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/mc4"))
    parser.add_argument("--flores-dir", type=Path, default=Path("data/raw/flores"),
                        help="Directory of per-language FLORES+ JSONL files used "
                             "to compute the parity weights.")
    parser.add_argument("--output-dir", type=Path,
                        default=Path("models/tokenizers/paat_phase1"))
    parser.add_argument("--initial-vocab", type=int, default=32_000)
    parser.add_argument("--target-vocab", type=int, default=16_000)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--total-docs", type=int, default=500_000)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--train-tokens-per-iter", type=int, default=5_000_000)
    parser.add_argument("--eval-tokens-per-iter", type=int, default=2_000_000)
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--balance-lambda", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--parity-alpha", type=float, default=1.0,
                        help="Weight of the parity bonus added to the ADAT "
                             "balance score.  0 disables parity awareness "
                             "(equivalent to ADAT).  Sensible range: 0.5–5.")
    parser.add_argument("--parity-max-sentences", type=int, default=None,
                        help="Cap FLORES+ sentences per language (default: all).")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[WARN] CUDA not available — running on CPU will be very slow.")

    # ---------------------------------------------------------------- corpus
    print("Loading mC4 texts ...")
    texts = load_mc4_texts(args.data_dir, ALL_LANGUAGES, args.total_docs, args.seed)
    print(f"  loaded {len(texts):,} documents across {len(ALL_LANGUAGES)} languages")

    train_texts, eval_texts = split_texts(texts, train_frac=0.9, seed=args.seed)
    print(f"  split: {len(train_texts):,} train   {len(eval_texts):,} eval")

    # ------------------------------------------------------------ parity data
    parity_texts: dict[str, list[str]] = {}
    if args.parity_alpha != 0.0:
        print(f"\nLoading FLORES+ parity sentences from {args.flores_dir} ...")
        parity_texts = load_flores_per_lang(
            args.flores_dir, ALL_LANGUAGES,
            max_sentences=args.parity_max_sentences,
        )
        if not parity_texts:
            raise SystemExit(
                f"No FLORES+ sentences found under {args.flores_dir}; "
                "use --parity-alpha 0 to disable, or fix the path."
            )
        n_sents = sum(len(v) for v in parity_texts.values())
        print(f"  loaded {n_sents:,} parallel sentences across "
              f"{len(parity_texts)} languages")
    else:
        print("[parity] alpha=0, parity-aware scoring disabled (== ADAT)")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- initial unigram
    init_dir = args.output_dir / "sp_init"
    init_model = init_dir / "sp.model"
    if init_model.exists():
        print(f"\n[init] reusing existing initial SentencePiece model at {init_model}")
    else:
        print(f"\n[init] training initial {args.initial_vocab:,}-piece Unigram ...")
        train_unigram_sentencepiece(
            texts=train_texts, output_dir=init_dir, vocab_size=args.initial_vocab,
        )

    initial_hf = sentencepiece_to_hf_unigram(init_model)
    print(f"[init] HF Unigram ready ({initial_hf.get_vocab_size():,} pieces)")

    # ------------------------------------------------------------------ paat
    paat_dir = args.output_dir / "paat"
    train_cfg = TrainConfig(batch_size=args.batch_size, lr=args.lr)
    cfg = PAATConfig(
        initial_vocab_size=args.initial_vocab,
        target_vocab_size=args.target_vocab,
        n_iterations=args.iterations,
        seq_len=args.seq_len,
        train_tokens_per_iter=args.train_tokens_per_iter,
        eval_tokens_per_iter=args.eval_tokens_per_iter,
        model_size=args.model_size,
        balance_lambda=args.balance_lambda,
        momentum=args.momentum,
        parity_alpha=args.parity_alpha,
        train=train_cfg,
    )

    final_paat_path = paat_dir / "tokenizer.json"
    if final_paat_path.exists():
        print(f"\n[paat] reusing existing PAAT tokenizer at {final_paat_path}")
        from tokenizers import Tokenizer as HFTokenizer
        final_tok = HFTokenizer.from_file(str(final_paat_path))
    else:
        print(f"\n[paat] running PAAT loop "
              f"({args.initial_vocab:,} -> {args.target_vocab:,} in "
              f"{args.iterations} iters, alpha={args.parity_alpha}) ...")
        final_tok, _ = run_paat(
            initial_hf, train_texts, eval_texts, parity_texts, cfg, paat_dir,
            device=device, seed=args.seed,
        )

    # ----------------------------------------------------- same-size baseline
    base_dir = args.output_dir / "baseline"
    base_sp = base_dir / "sp.model"
    base_hf_path = base_dir / "tokenizer.json"
    if not args.skip_baseline:
        if base_hf_path.exists():
            print(f"\n[baseline] reusing existing baseline at {base_hf_path}")
        else:
            print(f"\n[baseline] training direct {args.target_vocab:,}-piece Unigram ...")
            train_unigram_sentencepiece(
                texts=train_texts, output_dir=base_dir, vocab_size=args.target_vocab,
            )
            base_hf = sentencepiece_to_hf_unigram(base_sp)
            base_hf.save(str(base_hf_path))

    # ------------------------------------------------------- ppl comparison
    if args.skip_comparison:
        print("\n[compare] skipped per --skip-comparison")
        return

    print("\n[compare] training tiny LLMs on each tokenizer for PPL comparison ...")
    comp_train_tokens = args.train_tokens_per_iter
    comp_eval_tokens = args.eval_tokens_per_iter

    print("  [compare] PAAT tokenizer ...")
    paat_ppl = eval_tokenizer_ppl(
        final_paat_path, eval_texts, final_tok.get_vocab_size(), args.seq_len,
        comp_train_tokens, comp_eval_tokens, args.model_size, train_cfg, device,
    )
    print(f"  [compare] PAAT held-out PPL = {paat_ppl:.2f}")

    base_ppl = None
    if not args.skip_baseline:
        print("  [compare] Baseline Unigram tokenizer ...")
        base_ppl = eval_tokenizer_ppl(
            base_hf_path, eval_texts, args.target_vocab, args.seq_len,
            comp_train_tokens, comp_eval_tokens, args.model_size, train_cfg, device,
        )
        print(f"  [compare] Baseline held-out PPL = {base_ppl:.2f}")

    summary = {
        "paat": {
            "tokenizer": str(final_paat_path),
            "vocab_size": final_tok.get_vocab_size(),
            "ppl": paat_ppl,
            "parity_alpha": args.parity_alpha,
        },
        "baseline": None if base_ppl is None else {
            "tokenizer": str(base_hf_path),
            "vocab_size": args.target_vocab,
            "ppl": base_ppl,
        },
        "delta_ppl": None if base_ppl is None else (base_ppl - paat_ppl),
    }
    out = args.output_dir / "comparison.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[compare] wrote summary to {out}")

    if base_ppl is not None:
        verdict = "PAAT wins" if paat_ppl < base_ppl else "Baseline wins"
        print(f"\n  {verdict}:  PAAT={paat_ppl:.2f}   Baseline={base_ppl:.2f}   "
              f"Δ={base_ppl - paat_ppl:+.2f}")


if __name__ == "__main__":
    main()
