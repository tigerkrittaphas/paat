"""
Sample mC4 to a smaller, upload-friendly subset for cloud runs.

Mirrors the proportional-to-``MC4_NATURAL_COUNTS`` slicing that
``scripts/run_paat.py`` and ``scripts/pretrain_lm.py`` already perform via
``load_mc4_texts``.  Each language's output file contains the first
``round(total_docs * natural[lang] / sum(natural))`` lines of the source
JSONL — i.e. byte-for-byte the same documents the trainer would have
loaded locally, just persisted to disk so they can be uploaded once.

Output layout (matches ``data/raw/mc4/`` so no code changes are needed
on the cloud side):

    <output-dir>/
        af.jsonl
        am.jsonl
        ...

Usage:
    # ~3 M docs (covers the standard pretraining preset; ~10–15 GB)
    python scripts/sample_mc4.py --total-docs 3000000 \\
        --output-dir data/raw/mc4_sampled

    # Match the run_paat.py default (500 K docs; ~2–3 GB)
    python scripts/sample_mc4.py --total-docs 500000 \\
        --output-dir data/raw/mc4_paat500k

After sampling, tar+upload:

    tar -cf - data/raw/mc4_sampled data/raw/flores | zstd -T0 > mc4_bundle.tar.zst
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from paat.data.languages import ALL_LANGUAGES, MC4_NATURAL_COUNTS


def plan_allocation(total_docs: int, langs: list[str]) -> dict[str, int]:
    """Return ``{lang: n_docs}`` matching ``load_mc4_texts``' rounding.

    Identical formula: ``max(1, round(total_docs * natural / sum_natural))``.
    Languages absent from ``MC4_NATURAL_COUNTS`` get the same fallback (``1``).
    """
    total_natural = sum(MC4_NATURAL_COUNTS.get(l, 0) for l in langs)
    return {
        lang: max(1, round(total_docs * MC4_NATURAL_COUNTS.get(lang, 1) / total_natural))
        for lang in langs
    }


def copy_first_n_lines(src: Path, dst: Path, n: int) -> int:
    """Copy at most ``n`` non-empty lines from ``src`` to ``dst``.

    Streams line-by-line so memory is bounded regardless of source size.
    Returns the number of lines actually written.
    """
    dst.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(line if line.endswith("\n") else line + "\n")
            written += 1
            if written >= n:
                break
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/mc4"),
                        help="Source mC4 directory (one <lang>.jsonl per language).")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Destination for the sampled subset.")
    parser.add_argument("--total-docs", type=int, required=True,
                        help="Target total documents across all languages "
                             "(distributed proportional to MC4_NATURAL_COUNTS).")
    parser.add_argument("--langs", nargs="*", default=None,
                        help="Restrict to these language codes.  Default: all 96.")
    parser.add_argument("--include-flores", type=Path, default=None,
                        help="If given, also copy this FLORES+ directory verbatim "
                             "into <output-dir-parent>/flores/.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the per-language plan without writing files.")
    args = parser.parse_args()

    langs = args.langs or ALL_LANGUAGES
    plan = plan_allocation(args.total_docs, langs)

    print(f"\nSampling plan: {sum(plan.values()):,} docs across {len(plan)} langs "
          f"(target: {args.total_docs:,})")
    print(f"  data-dir:   {args.data_dir}")
    print(f"  output-dir: {args.output_dir}")

    # Quick top/bottom preview.
    sorted_plan = sorted(plan.items(), key=lambda kv: -kv[1])
    print("\n  top 5:    ", "  ".join(f"{l}={n:,}" for l, n in sorted_plan[:5]))
    print("  bottom 5: ", "  ".join(f"{l}={n:,}" for l, n in sorted_plan[-5:]))

    if args.dry_run:
        print("\n[dry-run] no files written.")
        return

    args.output_dir.mkdir(parents=True, exist_ok=True)

    total_written = 0
    missing: list[str] = []
    short: list[tuple[str, int, int]] = []
    for i, lang in enumerate(langs, start=1):
        src = args.data_dir / f"{lang}.jsonl"
        dst = args.output_dir / f"{lang}.jsonl"
        target = plan[lang]
        if not src.exists():
            missing.append(lang)
            continue
        n = copy_first_n_lines(src, dst, target)
        total_written += n
        if n < target:
            short.append((lang, n, target))
        if i % 10 == 0 or i == len(langs):
            print(f"  [{i:>3}/{len(langs)}] {lang}: wrote {n:,}/{target:,}  "
                  f"(running total {total_written:,})")

    if missing:
        print(f"\n[WARN] {len(missing)} languages missing in source: {missing}")
    if short:
        print(f"\n[WARN] {len(short)} languages produced fewer docs than requested:")
        for lang, got, want in short[:10]:
            print(f"        {lang}: {got:,}/{want:,}")
        if len(short) > 10:
            print(f"        ... and {len(short) - 10} more")

    # Optional FLORES+ passthrough.
    if args.include_flores is not None:
        flores_dst = args.output_dir.parent / "flores"
        if flores_dst.exists():
            print(f"\n[flores] {flores_dst} already exists, skipping copy.")
        else:
            print(f"\n[flores] copying {args.include_flores} -> {flores_dst}")
            shutil.copytree(args.include_flores, flores_dst)

    # Disk usage.
    total_bytes = sum(p.stat().st_size for p in args.output_dir.glob("*.jsonl"))
    print(f"\nDone.  {total_written:,} documents.  "
          f"Size on disk: {total_bytes / 1e9:.2f} GB.")
    print(f"\nNext: tar -cf - {args.output_dir} | zstd -T0 -19 > mc4_sample.tar.zst")


if __name__ == "__main__":
    main()
