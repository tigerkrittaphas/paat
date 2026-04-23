"""
Tokenizer parity metrics for PAAT.

Core metric: **fertility** — the average number of subword tokens produced
per whitespace-delimited word.  A perfectly fair multilingual tokenizer
would yield equal fertility across all languages; divergence indicates that
some languages are "over-fragmented" relative to others.

Reference: Rust et al. (2021) "How Good is Your Tokenizer?" ACL 2021.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import NamedTuple

import numpy as np
from tokenizers import Tokenizer


class LanguageStats(NamedTuple):
    lang: str
    n_sentences: int
    n_words: int          # whitespace-split words
    n_tokens: int         # subword tokens from the tokenizer
    fertility: float      # n_tokens / n_words
    avg_token_len: float  # mean character length of tokens (excl. special)
    unk_rate: float       # fraction of tokens that are [UNK]


def compute_language_stats(
    tokenizer: Tokenizer,
    sentences: list[str],
    lang: str,
) -> LanguageStats:
    """Compute parity metrics for a single language.

    Args:
        tokenizer:  Trained tokenizer to evaluate.
        sentences:  List of evaluation sentences (e.g. from FLORES+).
        lang:       Language code (used as label only).

    Returns:
        A :class:`LanguageStats` named-tuple.
    """
    unk_id = tokenizer.token_to_id("[UNK]")
    total_words = 0
    total_tokens = 0
    total_unk = 0
    token_lengths: list[int] = []

    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        enc = tokenizer.encode(sent)
        toks = enc.tokens

        total_words += len(words)
        total_tokens += len(toks)
        total_unk += sum(1 for t in toks if tokenizer.token_to_id(t) == unk_id)
        token_lengths.extend(len(t) for t in toks)

    if total_words == 0:
        return LanguageStats(lang, 0, 0, 0, 0.0, 0.0, 0.0)

    fertility = total_tokens / total_words
    avg_tok_len = float(np.mean(token_lengths)) if token_lengths else 0.0
    unk_rate = total_unk / total_tokens if total_tokens else 0.0

    return LanguageStats(
        lang=lang,
        n_sentences=len(sentences),
        n_words=total_words,
        n_tokens=total_tokens,
        fertility=fertility,
        avg_token_len=avg_tok_len,
        unk_rate=unk_rate,
    )


def load_flores_sentences(flores_dir: Path, lang: str) -> list[str]:
    """Load FLORES+ sentences for one language (dev + devtest)."""
    path = flores_dir / f"{lang}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"FLORES+ file not found: {path}")
    sentences = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                sentences.append(json.loads(line)["sentence"])
    return sentences


class ParityReport(NamedTuple):
    stats: list[LanguageStats]        # one entry per language
    mean_fertility: float
    std_fertility: float              # spread across languages (parity gap)
    max_fertility: float
    min_fertility: float
    fertility_ratio: float            # max / min  (1.0 = perfect parity)


def compute_parity_report(
    tokenizer: Tokenizer,
    flores_dir: Path,
    languages: list[str],
) -> ParityReport:
    """Evaluate parity across all given languages.

    Args:
        tokenizer:  Tokenizer to evaluate.
        flores_dir: Directory with per-language FLORES+ JSONL files.
        languages:  Language codes to include.

    Returns:
        A :class:`ParityReport` with per-language stats and aggregate metrics.
    """
    stats: list[LanguageStats] = []
    for lang in languages:
        sentences = load_flores_sentences(flores_dir, lang)
        s = compute_language_stats(tokenizer, sentences, lang)
        stats.append(s)
        print(f"  [{lang:>4}] fertility={s.fertility:.3f}  "
              f"unk={s.unk_rate:.4f}  sentences={s.n_sentences}")

    fertilities = [s.fertility for s in stats if s.n_sentences > 0]
    mean_f = float(np.mean(fertilities))
    std_f = float(np.std(fertilities))
    max_f = max(fertilities)
    min_f = min(fertilities)
    ratio = max_f / min_f if min_f > 0 else math.inf

    return ParityReport(
        stats=stats,
        mean_fertility=mean_f,
        std_fertility=std_f,
        max_fertility=max_f,
        min_fertility=min_f,
        fertility_ratio=ratio,
    )


def report_to_dict(report: ParityReport) -> dict:
    """Serialise a :class:`ParityReport` to a JSON-friendly dict."""
    return {
        "summary": {
            "mean_fertility": round(report.mean_fertility, 4),
            "std_fertility": round(report.std_fertility, 4),
            "max_fertility": round(report.max_fertility, 4),
            "min_fertility": round(report.min_fertility, 4),
            "fertility_ratio": round(report.fertility_ratio, 4),
            "n_languages": len(report.stats),
        },
        "languages": [
            {
                "lang": s.lang,
                "n_sentences": s.n_sentences,
                "n_words": s.n_words,
                "n_tokens": s.n_tokens,
                "fertility": round(s.fertility, 4),
                "avg_token_len": round(s.avg_token_len, 4),
                "unk_rate": round(s.unk_rate, 6),
            }
            for s in sorted(report.stats, key=lambda x: x.fertility)
        ],
    }
