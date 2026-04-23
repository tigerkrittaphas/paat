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


def _text_iter(data_dir: Path, languages: list[str] | None = None) -> Iterator[str]:
    """Yield raw text strings from <data_dir>/<lang>.jsonl files.

    Args:
        data_dir:  Directory containing per-language JSONL files.
        languages: If given, only read files whose stem is in this list.
    """
    files = sorted(data_dir.glob("*.jsonl"))
    if languages:
        lang_set = set(languages)
        files = [f for f in files if f.stem in lang_set]
    if not files:
        raise FileNotFoundError(f"No JSONL files found in {data_dir}")
    for path in files:
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    yield json.loads(line)["text"]


def train_bpe(
    data_dir: Path,
    output_dir: Path,
    vocab_size: int = 32_000,
    min_frequency: int = 2,
    languages: list[str] | None = None,
) -> Tokenizer:
    """Train a BPE tokenizer and save it to *output_dir*.

    Args:
        data_dir:      Directory with per-language ``<lang>.jsonl`` files.
        output_dir:    Where to write ``tokenizer.json`` (created if absent).
        vocab_size:    Target vocabulary size.
        min_frequency: Minimum merge pair frequency.
        languages:     Subset of language codes to train on (default: all).

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

    tokenizer.train_from_iterator(
        _text_iter(data_dir, languages),
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
