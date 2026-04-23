"""
Download mC4 multilingual corpus for PAAT experiments.

Languages: all 96 that overlap between mC4 and FLORES+ (see languages.py).

Full run: preserves the natural resource distribution (uniform scale across
MC4_NATURAL_COUNTS). Totals ~178 GB for the complete corpus.

Demo run (--demo): 5 000 docs per language, ~1 GB total. Use this to
validate the full pipeline before committing to the full-scale download.

Data is streamed from HuggingFace — no full materialisation required.
Output: data/raw/mc4/<lang>.jsonl  (one JSON object per line: {"text": ...})

Usage:
    python scripts/download_mc4.py --demo
    python scripts/download_mc4.py --demo --languages am sw th yo
    python scripts/download_mc4.py                          # full scale
    python scripts/download_mc4.py --languages de fr es     # full, subset
    python scripts/download_mc4.py --output-dir /data/mc4
"""

import argparse
import json
import sys
from pathlib import Path

from datasets import load_dataset
from tqdm import tqdm

from paat.data.languages import ALL_LANGUAGES, DEMO_DOC_COUNT, get_doc_counts


def download_language(lang: str, num_docs: int, output_dir: Path) -> None:
    out_path = output_dir / f"{lang}.jsonl"

    if out_path.exists():
        existing = sum(1 for _ in out_path.open())
        if existing >= num_docs:
            print(f"[{lang}] already complete ({existing:,} docs), skipping.")
            return
        print(f"[{lang}] partial file ({existing:,}/{num_docs:,} docs), re-downloading.")
        out_path.unlink()

    print(f"[{lang}] streaming {num_docs:,} docs from mC4 ...")
    ds = load_dataset("allenai/c4", lang, split="train", streaming=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with out_path.open("w", encoding="utf-8") as f:
        for example in tqdm(ds, total=num_docs, desc=lang, leave=False):
            f.write(json.dumps({"text": example["text"]}, ensure_ascii=False) + "\n")
            written += 1
            if written >= num_docs:
                break

    print(f"[{lang}] wrote {written:,} docs -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download mC4 corpus with natural language distribution."
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=f"Demo mode: {DEMO_DOC_COUNT:,} docs per language (~1 GB total).",
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=ALL_LANGUAGES,
        metavar="LANG",
        help="Language codes to download (default: all 96).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/mc4"),
    )
    args = parser.parse_args()

    unknown = [l for l in args.languages if l not in ALL_LANGUAGES]
    if unknown:
        print(f"Unknown language codes: {unknown}", file=sys.stderr)
        sys.exit(1)

    doc_counts = get_doc_counts(demo=args.demo)
    total_docs = sum(doc_counts[l] for l in args.languages)
    total_gb = total_docs * 2_000 / 1e9

    mode = "DEMO" if args.demo else "FULL"
    print(f"[{mode}] {len(args.languages)} languages | "
          f"~{total_docs:,} docs | ~{total_gb:.1f} GB")
    print()

    for lang in args.languages:
        download_language(lang, doc_counts[lang], args.output_dir)


if __name__ == "__main__":
    main()
