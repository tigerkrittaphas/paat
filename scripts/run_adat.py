"""
Run ADAT Phase 1 end-to-end: initial Unigram → iterative pruning → final
tokenizer, with a fair same-size Unigram baseline for comparison.

Usage:
    python scripts/run_adat.py \\
        --data-dir data/raw/mc4 \\
        --output-dir models/tokenizers/adat_phase1 \\
        --initial-vocab 32000 --target-vocab 16000 --iterations 3

The script writes:
    <out>/sp_init/sp.model             initial SentencePiece Unigram
    <out>/adat/iter_XX_vocab*.json     intermediate ADAT tokenizers
    <out>/adat/tokenizer.json          final ADAT tokenizer
    <out>/adat/adat_log.json           per-iteration log
    <out>/baseline/sp.model            direct Unigram baseline (same size)
    <out>/baseline/tokenizer.json      baseline in HF format
    <out>/comparison.json              PPL comparison on held-out mC4 slice
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from paat.data.languages import ALL_LANGUAGES
from paat.model.train import TrainConfig, evaluate_perplexity, train_llm
from paat.model.transformer import build_model
from paat.tokenizer.adat import ADATConfig, encode_corpus, run_adat
from paat.tokenizer.unigram import (
    sentencepiece_to_hf_unigram,
    train_unigram_sentencepiece,
)


def load_mc4_texts(data_dir: Path, langs: list[str], docs_per_lang: int,
                   seed: int) -> list[str]:
    """Load up to ``docs_per_lang`` documents per language."""
    rng = random.Random(seed)
    texts: list[str] = []
    for lang in langs:
        path = data_dir / f"{lang}.jsonl"
        if not path.exists():
            continue
        docs = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    docs.append(json.loads(line)["text"])
        rng.shuffle(docs)
        texts.extend(docs[:docs_per_lang])
    rng.shuffle(texts)
    return texts


def split_texts(texts: list[str], train_frac: float,
                seed: int) -> tuple[list[str], list[str]]:
    rng = random.Random(seed)
    shuffled = list(texts)
    rng.shuffle(shuffled)
    split = int(len(shuffled) * train_frac)
    return shuffled[:split], shuffled[split:]


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
    """Train a fresh small LLM on the given tokenizer and return held-out PPL.

    This is the fair, apples-to-apples comparison metric between ADAT and
    the baseline: same model size, same token budget, different vocabs.
    """
    from tokenizers import Tokenizer as HFTokenizer
    tok = HFTokenizer.from_file(str(tokenizer_path))

    train_texts, eval_split = eval_texts[:len(eval_texts) // 2], eval_texts[len(eval_texts) // 2:]
    train_ids = encode_corpus(tok, train_texts)[:train_tokens]
    eval_ids = encode_corpus(tok, eval_split)[:eval_tokens]

    model = build_model(vocab_size=vocab_size, size=model_size)
    model, ppl = train_llm(
        model, train_ids, eval_ids, seq_len,
        device=device, config=train_cfg,
    )
    return ppl


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/mc4"))
    parser.add_argument("--output-dir", type=Path,
                        default=Path("models/tokenizers/adat_phase1"))
    parser.add_argument("--initial-vocab", type=int, default=32_000)
    parser.add_argument("--target-vocab", type=int, default=16_000)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--docs-per-lang", type=int, default=2_000,
                        help="mC4 docs sampled per language for training.")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--train-tokens-per-iter", type=int, default=5_000_000)
    parser.add_argument("--eval-tokens-per-iter", type=int, default=2_000_000)
    parser.add_argument("--model-size", choices=["tiny", "small", "medium"],
                        default="tiny")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--balance-lambda", type=float, default=1.0)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--skip-baseline", action="store_true",
                        help="Skip training the baseline Unigram tokenizer.")
    parser.add_argument("--skip-comparison", action="store_true",
                        help="Skip the final PPL comparison.")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda":
        print("[WARN] CUDA not available — running on CPU will be very slow.")

    # ---------------------------------------------------------------- corpus
    print("Loading mC4 texts ...")
    texts = load_mc4_texts(args.data_dir, ALL_LANGUAGES, args.docs_per_lang, args.seed)
    print(f"  loaded {len(texts):,} documents across {len(ALL_LANGUAGES)} languages")

    train_texts, eval_texts = split_texts(texts, train_frac=0.9, seed=args.seed)
    print(f"  split: {len(train_texts):,} train   {len(eval_texts):,} eval")

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

    # ------------------------------------------------------------------ adat
    adat_dir = args.output_dir / "adat"
    train_cfg = TrainConfig(batch_size=args.batch_size, lr=args.lr)
    cfg = ADATConfig(
        initial_vocab_size=args.initial_vocab,
        target_vocab_size=args.target_vocab,
        n_iterations=args.iterations,
        seq_len=args.seq_len,
        train_tokens_per_iter=args.train_tokens_per_iter,
        eval_tokens_per_iter=args.eval_tokens_per_iter,
        model_size=args.model_size,
        balance_lambda=args.balance_lambda,
        momentum=args.momentum,
        train=train_cfg,
    )

    final_adat_path = adat_dir / "tokenizer.json"
    if final_adat_path.exists():
        print(f"\n[adat] reusing existing ADAT tokenizer at {final_adat_path}")
        from tokenizers import Tokenizer as HFTokenizer
        final_tok = HFTokenizer.from_file(str(final_adat_path))
    else:
        print(f"\n[adat] running ADAT loop "
              f"({args.initial_vocab:,} -> {args.target_vocab:,} in "
              f"{args.iterations} iters) ...")
        final_tok, _ = run_adat(
            initial_hf, train_texts, eval_texts, cfg, adat_dir,
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
            print(f"\n[baseline] training direct {args.target_vocab:,}-piece "
                  f"Unigram ...")
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

    print("  [compare] ADAT tokenizer ...")
    adat_ppl = eval_tokenizer_ppl(
        final_adat_path, eval_texts, final_tok.get_vocab_size(), args.seq_len,
        comp_train_tokens, comp_eval_tokens, args.model_size, train_cfg, device,
    )
    print(f"  [compare] ADAT held-out PPL = {adat_ppl:.2f}")

    base_ppl = None
    if not args.skip_baseline:
        print("  [compare] Baseline Unigram tokenizer ...")
        base_ppl = eval_tokenizer_ppl(
            base_hf_path, eval_texts, args.target_vocab, args.seq_len,
            comp_train_tokens, comp_eval_tokens, args.model_size, train_cfg, device,
        )
        print(f"  [compare] Baseline held-out PPL = {base_ppl:.2f}")

    summary = {
        "adat": {
            "tokenizer": str(final_adat_path),
            "vocab_size": final_tok.get_vocab_size(),
            "ppl": adat_ppl,
        },
        "baseline": None if base_ppl is None else {
            "tokenizer": str(base_hf_path),
            "vocab_size": args.target_vocab,
            "ppl": base_ppl,
        },
        "delta_ppl": None if base_ppl is None else (base_ppl - adat_ppl),
    }
    out = args.output_dir / "comparison.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"\n[compare] wrote summary to {out}")

    if base_ppl is not None:
        verdict = "ADAT wins" if adat_ppl < base_ppl else "Baseline wins"
        print(f"\n  {verdict}:  ADAT={adat_ppl:.2f}   Baseline={base_ppl:.2f}   "
              f"Δ={base_ppl - adat_ppl:+.2f}")


if __name__ == "__main__":
    main()
