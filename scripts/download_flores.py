"""
Download FLORES+ parallel evaluation corpus from HuggingFace.

Covers all 96 languages that overlap between mC4 and FLORES+.
Loads the full dataset once, then filters per language and writes
per-language JSONL files to data/raw/flores/<lang>.jsonl.

Each output record:
  {"id": <int>, "sentence": <str>, "split": "dev"|"devtest",
   "domain": <str>, "topic": <str>}

Usage:
    python scripts/download_flores.py
    python scripts/download_flores.py --languages af am ar
    python scripts/download_flores.py --output-dir data/raw/flores
"""

import argparse
import json
from pathlib import Path

from datasets import load_dataset

from paat.data.languages import ALL_LANGUAGES, LANG_REGISTRY


def save_language(lang: str, ds_full, output_dir: Path) -> None:
    out_path = output_dir / f"{lang}.jsonl"
    if out_path.exists():
        print(f"[{lang}] already exists, skipping.")
        return

    iso3, script = LANG_REGISTRY[lang]

    # Some languages have multiple varieties in FLORES+ sharing the same
    # iso_639_3 + iso_15924 (e.g. Catalan/Valencian, Norwegian Bokmål/radical).
    # Keep only the primary variety: glottocode tie-broken by taking the entry
    # with the lexicographically smallest glottocode, variant="" preferred.
    def _keep(ex: dict) -> bool:
        return (
            ex["iso_639_3"] == iso3
            and ex["iso_15924"] == script
            and ex.get("variant", "") == ""
        )

    # For languages with no variant="" rows, fall back to all rows (edge case).
    records = []
    for split_name in ("dev", "devtest"):
        filtered = ds_full[split_name].filter(_keep, desc=f"{lang}/{split_name}")
        if len(filtered) == 0:
            # Fallback: take first glottocode alphabetically
            filtered = ds_full[split_name].filter(
                lambda ex: ex["iso_639_3"] == iso3 and ex["iso_15924"] == script,
                desc=f"{lang}/{split_name} (fallback)",
            )

        # If multiple glottocodes remain (e.g. cat=stan1289+vale1252), keep
        # only the lexicographically first one (= the primary standard variety).
        glottocodes = sorted({r["glottocode"] for r in filtered})
        if len(glottocodes) > 1:
            primary = glottocodes[0]
            filtered = filtered.filter(lambda ex: ex["glottocode"] == primary)

        for ex in filtered:
            records.append({
                "id": ex["id"],
                "sentence": ex["text"],
                "split": split_name,
                "domain": ex["domain"],
                "topic": ex["topic"],
            })

    with out_path.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[{lang}] {len(records)} sentences -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download FLORES+ evaluation data for all 96 PAAT languages."
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=ALL_LANGUAGES,
        metavar="LANG",
        help="ISO 639-1 / mC4 codes to save (default: all 96).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/flores"),
    )
    args = parser.parse_args()

    unknown = [l for l in args.languages if l not in LANG_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown language codes: {unknown}. Valid: {sorted(LANG_REGISTRY)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Loading openlanguagedata/flores_plus ...")
    ds_full = load_dataset("openlanguagedata/flores_plus")

    for lang in args.languages:
        save_language(lang, ds_full, args.output_dir)


if __name__ == "__main__":
    main()
