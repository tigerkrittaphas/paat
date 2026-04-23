"""
Train a classic BPE tokenizer on mC4 demo or full data.

Usage:
    # Demo data (run after: python scripts/download_mc4.py --demo)
    python scripts/train_tokenizer.py \
        --data-dir data/raw/mc4 \
        --output-dir models/tokenizers/bpe_demo \
        --vocab-size 32000

    # Full data
    python scripts/train_tokenizer.py \
        --data-dir data/raw/mc4 \
        --output-dir models/tokenizers/bpe_full \
        --vocab-size 64000

    # Subset of languages
    python scripts/train_tokenizer.py \
        --data-dir data/raw/mc4 \
        --output-dir models/tokenizers/bpe_demo \
        --languages en zh ar hi sw am
"""

import argparse
import time
from pathlib import Path

from paat.data.languages import ALL_LANGUAGES
from paat.tokenizer.train import train_bpe


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a BPE tokenizer on mC4 JSONL data."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw/mc4"),
        help="Directory with per-language <lang>.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models/tokenizers/bpe_demo"),
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
        help="Subset of language codes (default: all available in data-dir).",
    )
    args = parser.parse_args()

    # Resolve languages to train on
    if args.languages:
        unknown = [l for l in args.languages if l not in ALL_LANGUAGES]
        if unknown:
            raise ValueError(f"Unknown language codes: {unknown}")
        langs = args.languages
    else:
        langs = None  # train_bpe will use all *.jsonl files

    available = sorted(args.data_dir.glob("*.jsonl"))
    if not available:
        raise FileNotFoundError(
            f"No JSONL files found in {args.data_dir}. "
            "Run scripts/download_mc4.py first."
        )
    print(f"Data dir:   {args.data_dir}  ({len(available)} language files)")
    print(f"Output dir: {args.output_dir}")
    print(f"Vocab size: {args.vocab_size:,}")
    if langs:
        print(f"Languages:  {langs}")
    print()

    t0 = time.time()
    train_bpe(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        languages=langs,
    )
    elapsed = time.time() - t0
    print(f"Training time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")


if __name__ == "__main__":
    main()
