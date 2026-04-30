"""
BPE tokenizer training for PAAT.

Trains a classic BPE tokenizer (whitespace pre-tokenization + NFKC
normalisation) on mC4 JSONL files using the HuggingFace `tokenizers`
library.  This is the *baseline* tokenizer; parity-aware variants are
trained in later stages of the ADAT loop.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import NFKC
from tokenizers.pre_tokenizers import Whitespace
from tokenizers.trainers import BpeTrainer


# Tokens reserved for downstream fine-tuning compatibility.
SPECIAL_TOKENS = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]


def _text_iter(
    data_dir: Path,
    languages: list[str] | None = None,
    docs_per_lang: dict[str, int] | None = None,
) -> Iterator[str]:
    """Yield raw text strings from <data_dir>/<lang>.jsonl files.

    Args:
        data_dir:      Directory containing per-language JSONL files.
        languages:     If given, only read files whose stem is in this list.
        docs_per_lang: Per-language doc caps. None means read all docs.
    """
    files = sorted(data_dir.glob("*.jsonl"))
    if languages:
        lang_set = set(languages)
        files = [f for f in files if f.stem in lang_set]
    if not files:
        raise FileNotFoundError(f"No JSONL files found in {data_dir}")
    for path in files:
        lang = path.stem
        limit = docs_per_lang.get(lang) if docs_per_lang else None
        count = 0
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)["text"]
                    count += 1
                    if limit is not None and count >= limit:
                        break


def _proportional_caps(
    data_dir: Path,
    languages: list[str] | None,
    total_docs: int,
) -> dict[str, int]:
    """Compute per-language doc caps proportional to MC4_NATURAL_COUNTS.

    Languages not present in MC4_NATURAL_COUNTS get a weight of 1 so they
    still receive a small allocation rather than being dropped.
    """
    from paat.data.languages import MC4_NATURAL_COUNTS

    files = sorted(data_dir.glob("*.jsonl"))
    if languages:
        lang_set = set(languages)
        files = [f for f in files if f.stem in lang_set]
    langs = [f.stem for f in files]

    total_natural = sum(MC4_NATURAL_COUNTS.get(l, 1) for l in langs)
    caps: dict[str, int] = {}
    for lang in langs:
        natural = MC4_NATURAL_COUNTS.get(lang, 1)
        caps[lang] = max(1, round(total_docs * natural / total_natural))
    return caps


def train_bpe(
    data_dir: Path,
    output_dir: Path,
    vocab_size: int = 32_000,
    min_frequency: int = 2,
    languages: list[str] | None = None,
    total_docs: int | None = None,
) -> Tokenizer:
    """Train a BPE tokenizer and save it to *output_dir*.

    Args:
        data_dir:      Directory with per-language ``<lang>.jsonl`` files.
        output_dir:    Where to write ``tokenizer.json`` (created if absent).
        vocab_size:    Target vocabulary size.
        min_frequency: Minimum merge pair frequency.
        languages:     Subset of language codes to train on (default: all).
        total_docs:    Total documents across all languages, distributed
                       proportionally to MC4_NATURAL_COUNTS. None = all docs.

    Returns:
        The trained :class:`~tokenizers.Tokenizer` object.
    """
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    tokenizer.normalizer = NFKC()
    tokenizer.pre_tokenizer = Whitespace()

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
    )

    docs_per_lang = (
        _proportional_caps(data_dir, languages, total_docs)
        if total_docs is not None
        else None
    )
    if docs_per_lang is not None:
        total = sum(docs_per_lang.values())
        print(f"  Total docs cap: {total:,} (proportional to MC4_NATURAL_COUNTS)")

    tokenizer.train_from_iterator(
        _text_iter(data_dir, languages, docs_per_lang),
        trainer=trainer,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / "tokenizer.json"
    tokenizer.save(str(out_path))
    print(f"Tokenizer saved to {out_path}  (vocab size: {tokenizer.get_vocab_size():,})")
    return tokenizer


def load_tokenizer(model_dir: Path) -> Tokenizer:
    """Load a previously saved tokenizer from *model_dir*."""
    path = model_dir / "tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"No tokenizer.json found in {model_dir}")
    return Tokenizer.from_file(str(path))
