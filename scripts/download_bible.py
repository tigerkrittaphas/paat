"""
Download Bible parallel translations for the 96 PAAT languages.

The Bible is the most multi-parallel text in existence: archaic register,
heavy religious vocabulary, very different distribution from mC4's news/web
domain.  Used here as a HARD cross-domain held-out parity-evaluation set
(complementing the in-domain NTREX-128 news corpus).

We use ``bible-nlp/biblenlp-corpus`` from HuggingFace, with iso_639_3 keys.
Bible corpora use inconsistent code conventions (macrolanguages, glottocodes,
region tags), so the mC4 → bible mapping is hand-curated.

Two-pass algorithm (verse alignment is intersection-first):

  1. Load every candidate language's Bible, collect per-lang sets of
     (book, chapter, verse) keys, compute the STRICT intersection across
     all candidate languages — only verses present in every Bible are kept.
     The canonical verse list is cached at data/raw/bible/_verse_index.json.

  2. For each language, write data/raw/bible/<mc4_lang>.jsonl with records
     ordered by the canonical verse list, so all per-lang files are
     positionally aligned and ``compute_parity_report`` sees parallel data.

Each output record:

    {"id": "<book>.<ch>.<v>", "sentence": "...", "split": "devtest"}

Languages that are missing or don't fully cover the intersection are
skipped (and logged to data/raw/bible/_skipped.json).

Usage:
    python scripts/download_bible.py
    python scripts/download_bible.py --languages en es de fr
    python scripts/download_bible.py --output-dir data/raw/bible
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from paat.data.languages import ALL_LANGUAGES, LANG_REGISTRY


# mC4 lang code -> bible-nlp subset id (typically iso_639_3).
# Hand-curated because biblenlp uses macrolanguage codes for some entries
# (e.g. 'arb' vs 'ara') and Bible coverage is sparse for many low-resource
# languages.  Languages without a known bible-nlp subset are mapped to None
# and skipped silently in pass 1.
BIBLE_LANG_MAP: dict[str, str | None] = {
    "af":  "afr",
    "sq":  "sqi",
    "am":  "amh",
    "ar":  "arb",
    "az":  "azb",            # Azerbaijani — biblenlp typically has azb (South)
    "be":  "bel",
    "bn":  "ben",
    "bg":  "bul",
    "ca":  "cat",
    "ceb": "ceb",
    "cs":  "ces",
    "zh":  "cmn",            # Mandarin
    "cy":  "cym",
    "da":  "dan",
    "de":  "deu",
    "et":  "est",
    "el":  "ell",
    "en":  "eng",
    "eo":  "epo",
    "eu":  "eus",
    "fil": "tgl",            # Filipino → Tagalog in biblenlp
    "fi":  "fin",
    "fr":  "fra",
    "gd":  "gla",
    "ga":  "gle",
    "gl":  "glg",
    "gu":  "guj",
    "ht":  "hat",
    "ha":  "hau",
    "iw":  "heb",
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
    "mn":  "khk",            # Mongolian (Khalkha)
    "km":  "khm",
    "ky":  "kir",
    "ku":  "kmr",
    "ko":  "kor",
    "lo":  "lao",
    "lt":  "lit",
    "lb":  "ltz",
    "lv":  "lav",
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
    "pa":  "pan",
    "ps":  "pus",
    "fa":  "pes",
    "mg":  "mlg",            # Malagasy (macrolanguage)
    "pl":  "pol",
    "pt":  "por",
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
    "sr":  "srp",
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
    "ms":  "zlm",            # Malay
    "zu":  "zul",
}


VREF_RE = re.compile(r"^([1-3A-Z]{2,4})[ \.](\d+)[:\.](\d+)$")


def parse_vref(vref: str) -> tuple[str, int, int] | None:
    """Parse a verse reference like ``"GEN 1:1"`` or ``"1JN.3.16"``.

    Returns ``(book, chapter, verse)`` or ``None`` if the string does not
    match the expected format (some bible-nlp rows are non-canonical
    headers / paratext that we should drop).
    """
    m = VREF_RE.match(vref.strip())
    if not m:
        return None
    return m.group(1), int(m.group(2)), int(m.group(3))


def load_bible_for_lang(subset_id: str) -> dict[tuple[str, int, int], str] | None:
    """Load one bible-nlp language subset → ``{(book, ch, v): text}``.

    Tries a few common load patterns to be resilient to upstream schema
    changes.  Returns ``None`` if the language can't be loaded — caller
    will skip and log to _skipped.json.
    """
    # Lazy import: keep CLI startup snappy and avoid a hard datasets
    # dependency at import-time of this module.
    from datasets import load_dataset

    candidates = [
        # (config_or_None, message_for_logs)
        (subset_id, f"per-language config '{subset_id}'"),
        (None, "default config"),
    ]

    last_err: Exception | None = None
    for config, label in candidates:
        try:
            ds = load_dataset(
                "bible-nlp/biblenlp-corpus",
                config,
                trust_remote_code=True,
            ) if config else load_dataset("bible-nlp/biblenlp-corpus", trust_remote_code=True)
        except Exception as e:  # noqa: BLE001 — best-effort schema probe
            last_err = e
            continue

        # Take the first split (usually 'train') — biblenlp ships verses
        # as a single split per language.
        split_name = next(iter(ds.keys()))
        rows = ds[split_name]
        cols = set(rows.column_names)

        # Filter to the requested language if the table is multi-language.
        if "language" in cols:
            rows = rows.filter(lambda ex: ex["language"] == subset_id,
                               desc=f"bible/{subset_id} filter")
            if len(rows) == 0:
                last_err = ValueError(f"no rows for language={subset_id}")
                continue
        elif "iso" in cols:
            rows = rows.filter(lambda ex: ex["iso"] == subset_id,
                               desc=f"bible/{subset_id} filter")
            if len(rows) == 0:
                last_err = ValueError(f"no rows for iso={subset_id}")
                continue

        # Pick the verse-text column.  bible-nlp variants use 'text',
        # 'translation', or 'verse' depending on version.
        text_col = next((c for c in ("text", "translation", "verse") if c in cols), None)
        vref_col = next((c for c in ("vref", "ref", "verse_id") if c in cols), None)
        if text_col is None or vref_col is None:
            last_err = ValueError(
                f"bible-nlp schema not recognised (cols={sorted(cols)}); "
                "expected text-like + vref-like columns"
            )
            continue

        verses: dict[tuple[str, int, int], str] = {}
        for ex in rows:
            key = parse_vref(str(ex[vref_col]))
            if key is None:
                continue
            txt = ex[text_col]
            if not txt or not txt.strip():
                continue
            # If duplicates exist (re-translations), keep the first occurrence.
            verses.setdefault(key, txt.strip())

        if not verses:
            last_err = ValueError("no parseable verses after filtering")
            continue

        print(f"  [{subset_id}] loaded {len(verses):,} verses via {label}")
        return verses

    print(f"  [{subset_id}] FAILED to load: {last_err}")
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download Bible evaluation data for the 96 PAAT languages."
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
        default=Path("data/raw/bible"),
    )
    parser.add_argument(
        "--min-intersection",
        type=int,
        default=200,
        help=(
            "Minimum number of verses in the strict intersection.  Aborts "
            "if the intersection shrinks below this — usually means a "
            "single broken bible is dragging coverage to zero."
        ),
    )
    args = parser.parse_args()

    unknown = [l for l in args.languages if l not in LANG_REGISTRY]
    if unknown:
        raise ValueError(f"Unknown language codes: {unknown}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    verse_index_path = args.output_dir / "_verse_index.json"
    skipped_path = args.output_dir / "_skipped.json"

    # ── Pass 1: load every candidate Bible, compute strict intersection ──
    candidates: list[tuple[str, str]] = []   # (mc4_lang, subset_id)
    for lang in args.languages:
        sub = BIBLE_LANG_MAP.get(lang)
        if sub is None:
            continue
        if (args.output_dir / f"{lang}.jsonl").exists() and verse_index_path.exists():
            # Idempotent skip: per-lang already written and we have a cached
            # canonical verse index — nothing to do for this language.
            print(f"[{lang}] already exists, skipping pass 1.")
            continue
        candidates.append((lang, sub))

    skipped: list[dict] = []

    if verse_index_path.exists() and not candidates:
        canonical = [tuple(k) for k in json.loads(verse_index_path.read_text())]
        print(f"[bible] reusing cached verse index ({len(canonical)} verses)")
        per_lang_verses: dict[str, dict[tuple[str, int, int], str]] = {}
    else:
        print(f"[bible] pass 1 — loading {len(candidates)} candidate Bibles ...")
        per_lang_verses = {}
        for lang, sub in candidates:
            verses = load_bible_for_lang(sub)
            if verses is None:
                skipped.append({"lang": lang, "subset": sub, "reason": "load_failed"})
                continue
            per_lang_verses[lang] = verses

        if not per_lang_verses and not verse_index_path.exists():
            raise RuntimeError(
                "No candidate Bibles loaded successfully — cannot compute "
                "verse intersection.  Check network access and the "
                "bible-nlp/biblenlp-corpus schema."
            )

        # Strict intersection across ALL successfully-loaded candidates.
        if verse_index_path.exists():
            # Combine cached index with newly loaded langs.  The cached
            # index is authoritative; new langs must cover it.
            cached = {tuple(k) for k in json.loads(verse_index_path.read_text())}
            for lang, verses in per_lang_verses.items():
                missing = cached - set(verses)
                if missing:
                    skipped.append({
                        "lang": lang, "subset": BIBLE_LANG_MAP[lang],
                        "reason": "incomplete_intersection_coverage",
                        "missing_verses": len(missing),
                    })
            canonical = sorted(cached)
        else:
            intersection = set.intersection(
                *(set(v.keys()) for v in per_lang_verses.values())
            )
            if len(intersection) < args.min_intersection:
                raise RuntimeError(
                    f"Bible intersection collapsed to {len(intersection)} verses "
                    f"(< --min-intersection={args.min_intersection}).  Inspect "
                    "per-language verse counts above to find the offender, then "
                    "either drop it from BIBLE_LANG_MAP or rerun with a smaller "
                    "candidate set."
                )
            canonical = sorted(intersection)
            verse_index_path.write_text(
                json.dumps([list(k) for k in canonical], ensure_ascii=False, indent=2)
            )
            print(f"[bible] strict intersection = {len(canonical)} verses "
                  f"across {len(per_lang_verses)} languages")
            print(f"[bible] cached canonical verse index -> {verse_index_path}")

    # ── Pass 2: write per-language JSONL in canonical verse order ──
    n_written = 0
    for lang, verses in per_lang_verses.items():
        out_path = args.output_dir / f"{lang}.jsonl"
        if out_path.exists():
            print(f"[{lang}] already exists, skipping pass 2.")
            continue
        # Re-check coverage in case skipped above.
        if any(s["lang"] == lang for s in skipped):
            continue
        with out_path.open("w", encoding="utf-8") as fh:
            for key in canonical:
                txt = verses.get(key)
                if txt is None:
                    # Should not happen — canonical is the intersection — but
                    # be safe so we never silently misalign.
                    raise RuntimeError(
                        f"[{lang}] missing verse {key} despite being in canonical index"
                    )
                vid = f"{key[0]}.{key[1]}.{key[2]}"
                fh.write(json.dumps(
                    {"id": vid, "sentence": txt, "split": "devtest"},
                    ensure_ascii=False,
                ) + "\n")
        n_written += 1
        print(f"[{lang}] {len(canonical)} verses -> {out_path}")

    if skipped:
        skipped_path.write_text(json.dumps(skipped, ensure_ascii=False, indent=2))
        print(f"[bible] skipped {len(skipped)} languages -> {skipped_path}")

    # Coverage report (counts what's actually on disk, not just what we
    # wrote in this run, so it survives partial reruns).
    on_disk = sorted(p.stem for p in args.output_dir.glob("*.jsonl")
                     if not p.stem.startswith("_"))
    print(f"\n[bible] {len(on_disk)}/{len(args.languages)} PAAT langs covered, "
          f"{len(canonical) if canonical else 0} aligned verses")

    # Multi-parallel alignment assertion: every Bible JSONL must have the
    # same line count (= len(canonical)).
    counts = {p: sum(1 for _ in (args.output_dir / f"{p}.jsonl").open(encoding="utf-8"))
              for p in on_disk}
    distinct = set(counts.values())
    if len(distinct) > 1:
        offenders = {k: v for k, v in counts.items() if v != max(distinct, key=lambda c: sum(1 for v in counts.values() if v == c))}
        raise AssertionError(
            f"Non-aligned Bible verse counts: {offenders}. "
            "Pass 2 must write canonical verse order — investigate."
        )


if __name__ == "__main__":
    main()
