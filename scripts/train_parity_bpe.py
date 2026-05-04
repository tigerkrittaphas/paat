"""
Train a parity-aware BPE tokenizer (Foroutan et al., 2025) on mC4 demo or full data.

Mirrors the interface of scripts/train_tokenizer.py — same data layout
(per-language <lang>.jsonl files in --data-dir) and same output layout
(tokenizer.json in --output-dir, loadable by scripts/eval_parity.py).

Unlike classic BPE, parity-aware BPE picks each merge to balance
compression across languages, and therefore needs:
  * one input corpus per language (provided automatically from --data-dir),
  * a multi-parallel development set per language for parity computation
    (FLORES+, by default --flores-dir data/raw/flores).

Usage:
    # Demo data (after: python scripts/download_flores.py
    #             and: python scripts/download_mc4.py --demo)
    python scripts/train_parity_bpe.py \\
        --data-dir data/raw/mc4 \\
        --flores-dir data/raw/flores \\
        --output-dir models/tokenizers/parity_bpe_demo \\
        --vocab-size 32000

    # Subset of languages
    python scripts/train_parity_bpe.py \\
        --data-dir data/raw/mc4 \\
        --output-dir models/tokenizers/parity_bpe_demo \\
        --languages en zh ar hi sw am

    # Moving-window variant
    python scripts/train_parity_bpe.py \\
        --data-dir data/raw/mc4 \\
        --output-dir models/tokenizers/parity_bpe_window \\
        --variant window --window-size 100 --alpha 2

    # Hybrid: first 5000 merges global (standard BPE), rest parity-driven
    python scripts/train_parity_bpe.py \\
        --data-dir data/raw/mc4 \\
        --output-dir models/tokenizers/parity_bpe_hybrid \\
        --global-merges 5000
"""

import argparse
import time
from pathlib import Path

from paat.data.languages import ALL_LANGUAGES
from paat.tokenizer.parity_bpe import train_parity_bpe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a parity-aware BPE tokenizer on mC4 JSONL data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/mc4"),
        help="Directory with per-language <lang>.jsonl mC4 files.",
    )
    parser.add_argument(
        "--flores-dir",
        type=Path,
        default=Path("data/raw/flores"),
        help="Directory with per-language FLORES+ JSONL files used as the "
             "multi-parallel parity dev set. Required unless --ratio is given.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/tokenizers/parity_bpe_demo"),
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32_000,
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum merge pair frequency.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        metavar="LANG",
        help="Subset of language codes (default: all available in --data-dir).",
    )
    parser.add_argument(
        "--variant",
        choices=("base", "window"),
        default="base",
        help="Parity-aware BPE variant (default: %(default)s).",
    )
    parser.add_argument(
        "--global-merges",
        type=int,
        default=0,
        help="Number of initial global-statistics (classic-BPE) merges before "
             "switching to parity-driven merging. Default: %(default)s.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=100,
        help="Window size for the moving-window variant (default: %(default)s).",
    )
    parser.add_argument(
        "--alpha",
        type=int,
        default=2,
        help="Window threshold ratio for the moving-window variant (default: %(default)s).",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        nargs="+",
        default=None,
        metavar="R",
        help="Per-language compression ratios — alternative to using FLORES+ "
             "dev sets. Must match the number of languages.",
    )
    parser.add_argument(
        "--total-docs",
        type=int,
        default=None,
        metavar="N",
        help="Total mC4 docs across all languages, distributed proportionally "
             "to MC4_NATURAL_COUNTS.  Default: None = stream entire language "
             "files (gigabytes each; only sane on a pre-sampled corpus).",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose progress output from the learner.",
    )
    args = parser.parse_args()

    if args.languages:
        unknown = [l for l in args.languages if l not in ALL_LANGUAGES]
        if unknown:
            raise ValueError(f"Unknown language codes: {unknown}")
        langs = args.languages
    else:
        langs = None

    available = sorted(args.data_dir.glob("*.jsonl"))
    if not available:
        raise FileNotFoundError(
            f"No JSONL files found in {args.data_dir}. "
            "Run scripts/download_mc4.py first."
        )

    flores_dir = None if args.ratio is not None else args.flores_dir

    print(f"Data dir:    {args.data_dir}  ({len(available)} language files)")
    if flores_dir is not None:
        print(f"FLORES dir:  {flores_dir}")
    if args.ratio is not None:
        print(f"Ratios:      {args.ratio}")
    print(f"Output dir:  {args.output_dir}")
    print(f"Vocab size:  {args.vocab_size:,}")
    print(f"Variant:     {args.variant}")
    if args.global_merges:
        print(f"Hybrid:      first {args.global_merges} merges global, rest parity-driven")
    if langs:
        print(f"Languages:   {langs}")
    print()

    t0 = time.time()
    train_parity_bpe(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        languages=langs,
        flores_dir=flores_dir,
        variant=args.variant,
        global_merges=args.global_merges,
        window_size=args.window_size,
        alpha=args.alpha,
        ratio=args.ratio,
        verbose=args.verbose,
        total_docs=args.total_docs,
    )
    elapsed = time.time() - t0
    print(f"Training time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
