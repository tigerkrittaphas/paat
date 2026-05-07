"""
Download NTREX-128 (Microsoft news translations) for the 96 PAAT languages.

NTREX-128 is a fully multi-parallel evaluation set: 1997 newstest2019 source
sentences professionally translated into 128 target languages.  We use it as
a held-out parity-evaluation corpus that no PAAT tokenizer has seen during
training (PAAT and parity-BPE both consume FLORES+ dev as a parity signal).

The repo lives at https://github.com/MicrosoftTranslator/NTREX-128 and stores
each language as ``NTREX-128/newstest2019-ref.<bcp47>.txt`` — one sentence
per line, line-aligned across languages.

Each output record (data/raw/ntrex/<mc4_lang>.jsonl):

    {"id": <int line index>, "sentence": "...", "split": "devtest"}

The ``"split": "devtest"`` value is intentional — eval_parity.py's default
loader keeps records where ``split == "devtest"``, so no flag changes are
needed downstream.

Usage:
    python scripts/download_ntrex.py
    python scripts/download_ntrex.py --ntrex-repo vendor/NTREX-128
    python scripts/download_ntrex.py --languages am ar bn
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from paat.data.languages import ALL_LANGUAGES, LANG_REGISTRY


NTREX_REPO_URL = "https://github.com/MicrosoftTranslator/NTREX-128"

# mC4 lang code -> NTREX-128 BCP-47 stem (the "<bcp47>" in
# newstest2019-ref.<bcp47>.txt).  NTREX uses BCP-47 with region/script tags
# that don't always match iso_639_3 + iso_15924, so this mapping is
# hand-curated rather than auto-derived from LANG_REGISTRY.
#
# Verified against the NTREX-128 repo file listing (commit at the time of
# the exp_3 plan).  Any code missing from the actual repo will be skipped
# at download time with a clear log line; cross-check NTREX-128/README.md
# if coverage looks too low.
NTREX_LANG_MAP: dict[str, str] = {
    "af":  "afr",
    "sq":  "sqi",            # NTREX uses sqi, not als
    "am":  "amh",
    "ar":  "arb",
    "az":  "aze-Latn",
    "be":  "bel",
    "bn":  "ben",
    "bg":  "bul",
    "ca":  "cat",
    "ceb": "ceb",
    "cs":  "ces",
    "zh":  "zho-CN",         # Simplified Mandarin (NTREX uses zho-CN/zho-TW, not zho-Hans)
    "cy":  "cym",
    "da":  "dan",
    "de":  "deu",
    "et":  "est",
    "el":  "ell",
    "en":  "eng-US",         # NTREX has eng-US and eng-GB; pick the US ref
    "eo":  "epo",
    "eu":  "eus",
    "fil": "fil",
    "fi":  "fin",
    "fr":  "fra",
    "gd":  "gla",
    "ga":  "gle",
    "gl":  "glg",
    "gu":  "guj",
    "ht":  "hat",
    "ha":  "hau",
    "iw":  "heb",            # mC4 uses legacy 'iw'; NTREX uses 'heb'
    "hi":  "hin",
    "hu":  "hun",
    "hy":  "hye",
    "ig":  "ibo",
    "id":  "ind",
    "is":  "isl",
    "it":  "ita",
    "jv":  "jav",
    "ja":  "jpn",
    "kn":  "kan",
    "ka":  "kat",
    "kk":  "kaz",
    "mn":  "mon",            # NTREX uses 'mon' (Khalkha Mongolian, Cyrl)
    "km":  "khm",
    "ky":  "kir",
    "ku":  "kmr",
    "ko":  "kor",
    "lo":  "lao",
    "lt":  "lit",
    "lb":  "ltz",
    "lv":  "lvs",
    "ml":  "mal",
    "mr":  "mar",
    "mk":  "mkd",
    "mt":  "mlt",
    "mi":  "mri",
    "my":  "mya",
    "nl":  "nld",
    "no":  "nob",
    "ne":  "npi",
    "ny":  "nya",
    "pa":  "pan-Guru",
    "ps":  "pus",
    "fa":  "fas",            # Persian (Western); NTREX uses 'fas'
    "mg":  "plt",
    "pl":  "pol",
    "pt":  "por-BR",         # NTREX has por-BR and por-PT; pick BR
    "ro":  "ron",
    "ru":  "rus",
    "si":  "sin",
    "sk":  "slk",
    "sl":  "slv",
    "sm":  "smo",
    "sn":  "sna",
    "sd":  "snd",
    "so":  "som",
    "st":  "sot",
    "es":  "spa",
    "sr":  "srp-Cyrl",
    "su":  "sun",
    "sv":  "swe",
    "sw":  "swh",
    "ta":  "tam",
    "te":  "tel",
    "tg":  "tgk",
    "th":  "tha",
    "tr":  "tur",
    "uk":  "ukr",
    "ur":  "urd",
    "uz":  "uzn",
    "vi":  "vie",
    "xh":  "xho",
    "yi":  "ydd",
    "yo":  "yor",
    "ms":  "zsm",
    "zu":  "zul",
}


def ensure_repo(repo_path: Path) -> Path:
    """Clone NTREX-128 if it's not already at ``repo_path``.

    Returns the path to the ``NTREX-128/`` data subdirectory containing the
    per-language .txt files.
    """
    if repo_path.exists() and (repo_path / "NTREX-128").is_dir():
        print(f"[ntrex] reusing existing checkout at {repo_path}")
        return repo_path / "NTREX-128"

    if shutil.which("git") is None:
        raise RuntimeError(
            "git not found on PATH — install git or pre-clone NTREX-128 to "
            f"{repo_path} (expected layout: {repo_path}/NTREX-128/*.txt)"
        )

    repo_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[ntrex] cloning {NTREX_REPO_URL} -> {repo_path}")
    subprocess.run(
        ["git", "clone", "--depth", "1", NTREX_REPO_URL, str(repo_path)],
        check=True,
    )
    data_dir = repo_path / "NTREX-128"
    if not data_dir.is_dir():
        raise RuntimeError(
            f"NTREX clone succeeded but {data_dir} is missing — "
            "the upstream repo layout may have changed."
        )
    return data_dir


def save_language(
    mc4_lang: str,
    ntrex_data_dir: Path,
    output_dir: Path,
) -> tuple[bool, int]:
    """Convert one NTREX language file to per-language JSONL.

    Returns (wrote_or_skipped, n_sentences).  ``wrote_or_skipped`` is True
    if the output file exists at the end of the call (either freshly written
    or already present); False if the source file was missing.
    """
    out_path = output_dir / f"{mc4_lang}.jsonl"
    if out_path.exists():
        n = sum(1 for _ in out_path.open(encoding="utf-8"))
        print(f"[{mc4_lang}] already exists ({n} sentences), skipping.")
        return True, n

    bcp47 = NTREX_LANG_MAP.get(mc4_lang)
    if bcp47 is None:
        print(f"[{mc4_lang}] no NTREX_LANG_MAP entry — skipping.")
        return False, 0

    src = ntrex_data_dir / f"newstest2019-ref.{bcp47}.txt"
    if not src.exists():
        print(f"[{mc4_lang}] missing source file {src.name} — skipping.")
        return False, 0

    n = 0
    with src.open(encoding="utf-8") as fh, out_path.open("w", encoding="utf-8") as out:
        for idx, line in enumerate(fh):
            sent = line.rstrip("\n")
            if not sent.strip():
                continue
            out.write(json.dumps(
                {"id": idx, "sentence": sent, "split": "devtest"},
                ensure_ascii=False,
            ) + "\n")
            n += 1
    print(f"[{mc4_lang}] {n} sentences -> {out_path}")
    return True, n


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download NTREX-128 evaluation data for PAAT languages."
    )
    parser.add_argument(
        "--languages",
        nargs="+",
        default=ALL_LANGUAGES,
        metavar="LANG",
        help="mC4 codes to save (default: all 96).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/raw/ntrex"),
    )
    parser.add_argument(
        "--ntrex-repo",
        type=Path,
        default=Path("vendor/NTREX-128"),
        help="Local path to clone NTREX-128 to (or reuse if it already exists).",
    )
    args = parser.parse_args()

    unknown = [l for l in args.languages if l not in LANG_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown language codes: {unknown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ntrex_data_dir = ensure_repo(args.ntrex_repo)

    counts: dict[str, int] = {}
    skipped: list[str] = []
    for lang in args.languages:
        wrote, n = save_language(lang, ntrex_data_dir, args.output_dir)
        if wrote:
            counts[lang] = n
        else:
            skipped.append(lang)

    print()
    print(f"[ntrex] wrote {len(counts)}/{len(args.languages)} languages to {args.output_dir}")
    if skipped:
        print(f"[ntrex] skipped (no NTREX file): {' '.join(skipped)}")

    # Multi-parallel alignment assertion: every NTREX file must have the
    # same sentence count, since NTREX-128 is fully multi-parallel.
    written_counts = {p.stem: sum(1 for _ in p.open(encoding="utf-8"))
                      for p in args.output_dir.glob("*.jsonl")}
    distinct = set(written_counts.values())
    if len(distinct) > 1:
        # Show the offenders explicitly so the breakage is debuggable.
        modal = max(distinct, key=lambda c: sum(1 for v in written_counts.values() if v == c))
        offenders = {k: v for k, v in written_counts.items() if v != modal}
        raise AssertionError(
            f"Non-aligned NTREX sentence counts (modal={modal}, offenders={offenders}). "
            "NTREX-128 is multi-parallel — every file must have the same line count."
        )
    print(f"[ntrex] alignment OK: {len(written_counts)} files × {distinct.pop() if distinct else 0} sentences")


if __name__ == "__main__":
    main()
