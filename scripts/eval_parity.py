"""
Evaluate tokenizer parity across languages using FLORES+ sentences.

Computes per-language fertility (tokens/words) and aggregate parity metrics.
Outputs a JSON report and a console table sorted by fertility.

Usage:
    python scripts/eval_parity.py \
        --tokenizer models/tokenizers/bpe_demo \
        --flores-dir data/raw/flores \
        --output results/parity/bpe_demo.json

    # Subset of languages
    python scripts/eval_parity.py \
        --tokenizer models/tokenizers/bpe_demo \
        --languages en zh ar hi sw am yo
"""

import argparse
import json
from pathlib import Path

from paat.data.languages import ALL_LANGUAGES
from paat.parity.metrics import compute_parity_report, report_to_dict
from paat.tokenizer.train import load_tokenizer


def print_table(report_dict: dict) -> None:
    summary = report_dict["summary"]
    langs = report_dict["languages"]  # sorted by fertility

    print()
    print(f"{'lang':>5}  {'fertility':>9}  {'tok/sent':>9}  {'avg_tok_len':>11}  {'unk_rate':>9}")
    print("-" * 52)
    for l in langs:
        print(
            f"  {l['lang']:>4}  {l['fertility']:>9.3f}"
            f"  {l['n_tokens']/l['n_sentences'] if l['n_sentences'] else 0:>9.1f}"
            f"  {l['avg_token_len']:>11.2f}"
            f"  {l['unk_rate']:>9.5f}"
        )
    print("-" * 52)
    print(
        f"  {'ALL':>4}  {summary['mean_fertility']:>9.3f}"
        f"  {'':>9}"
        f"  {'':>11}"
        f"  {'':>9}"
    )
    print()
    print(f"  Fertility std:   {summary['std_fertility']:.4f}  "
          f"(spread across languages; lower = more fair)")
    print(f"  Fertility ratio: {summary['fertility_ratio']:.2f}x  "
          f"(max/min; 1.0 = perfect parity)")
    print(f"  Max: {summary['max_fertility']:.3f}   Min: {summary['min_fertility']:.3f}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate tokenizer parity on FLORES+ evaluation data."
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("models/tokenizers/bpe_demo"),
        help="Directory containing tokenizer.json.",
    )
    parser.add_argument(
        "--flores-dir",
        type=Path,
        default=Path("data/raw/flores"),
        help="Directory with per-language FLORES+ JSONL files.",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        metavar="LANG",
        help="Language codes to evaluate (default: all with a FLORES+ file).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path to write JSON report (default: results/parity/<tokenizer_name>.json).",
    )
    args = parser.parse_args()

    # Resolve language list
    if args.languages:
        unknown = [l for l in args.languages if l not in ALL_LANGUAGES]
        if unknown:
            raise ValueError(f"Unknown language codes: {unknown}")
        langs = args.languages
    else:
        langs = sorted(
            p.stem for p in args.flores_dir.glob("*.jsonl")
            if p.stem in ALL_LANGUAGES
        )
    if not langs:
        raise FileNotFoundError(
            f"No FLORES+ files found in {args.flores_dir}. "
            "Run scripts/download_flores.py first."
        )

    output = args.output or (
        Path("results/parity") / f"{args.tokenizer.name}.json"
    )

    print(f"Tokenizer:  {args.tokenizer}")
    print(f"FLORES dir: {args.flores_dir}")
    print(f"Languages:  {len(langs)}")
    print()

    tokenizer = load_tokenizer(args.tokenizer)
    print(f"Vocab size: {tokenizer.get_vocab_size():,}")
    print()

    report = compute_parity_report(tokenizer, args.flores_dir, langs)
    report_dict = report_to_dict(report)

    print_table(report_dict)

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    print(f"Report saved to {output}")


if __name__ == "__main__":
    main()
