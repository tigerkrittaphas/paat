"""
Tokenizer parity metrics for PAAT.

We compute two families of metrics:

* **Legacy fertility** — tokens per whitespace-delimited word.  Retained for
  backward comparability with earlier reports, but **unreliable for languages
  without whitespace word boundaries** (Chinese, Japanese, Thai, Khmer, Lao,
  Myanmar), where ``str.split()`` typically returns a single "word" per
  sentence and drastically inflates the fertility number.

* **Parallel-sentence metrics** (recommended) — FLORES+ is aligned across
  languages: sentence *i* in Thai is a translation of sentence *i* in English,
  etc.  Therefore ``tokens_per_sentence`` and ``tokens_per_byte`` compare
  like-for-like content across languages and are robust to script and
  whitespace differences.  We also report the **Gini coefficient of
  tokens-per-sentence across languages**, which is the headline fairness
  metric used in Foroutan et al. 2025 ("Parity-Aware Byte-Pair Encoding").

References:
    Rust et al. (2021) "How Good is Your Tokenizer?" ACL 2021.
    Foroutan et al. (2025) "Parity-Aware Byte-Pair Encoding: Improving
        Cross-lingual Fairness in Tokenization." arXiv:2508.04796.
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
    n_words: int                  # whitespace-split (unreliable for CJK/Thai)
    n_tokens: int
    n_bytes: int                  # UTF-8 byte length of the raw text
    fertility: float              # tokens / words  (legacy, unreliable for CJK)
    tokens_per_sentence: float    # tokens / parallel sentence  (primary)
    tokens_per_byte: float        # tokens / UTF-8 byte         (script-aware)
    avg_token_len: float
    unk_rate: float


def compute_language_stats(
    tokenizer: Tokenizer,
    sentences: list[str],
    lang: str,
) -> LanguageStats:
    """Compute parity metrics for a single language.

    Args:
        tokenizer:  Trained tokenizer to evaluate.
        sentences:  Parallel evaluation sentences (e.g. from FLORES+).
                    Must be aligned across languages for cross-lingual
                    comparison via ``tokens_per_sentence``.
        lang:       Language code (used as label only).

    Returns:
        A :class:`LanguageStats` named-tuple.
    """
    # Unigram tokenizers use "<unk>"; BPE tokenizers use "[UNK]".  Try both.
    unk_id = tokenizer.token_to_id("<unk>")
    if unk_id is None:
        unk_id = tokenizer.token_to_id("[UNK]")

    total_words = 0
    total_tokens = 0
    total_bytes = 0
    total_unk = 0
    n_valid_sentences = 0
    token_lengths: list[int] = []

    for sent in sentences:
        words = sent.split()
        if not words:
            continue
        enc = tokenizer.encode(sent)
        toks = enc.tokens

        n_valid_sentences += 1
        total_words += len(words)
        total_tokens += len(toks)
        total_bytes += len(sent.encode("utf-8"))
        if unk_id is not None:
            total_unk += sum(1 for tid in enc.ids if tid == unk_id)
        token_lengths.extend(len(t) for t in toks)

    if n_valid_sentences == 0:
        return LanguageStats(lang, 0, 0, 0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

    fertility = total_tokens / total_words if total_words else 0.0
    tokens_per_sentence = total_tokens / n_valid_sentences
    tokens_per_byte = total_tokens / total_bytes if total_bytes else 0.0
    avg_tok_len = float(np.mean(token_lengths)) if token_lengths else 0.0
    unk_rate = total_unk / total_tokens if total_tokens else 0.0

    return LanguageStats(
        lang=lang,
        n_sentences=n_valid_sentences,
        n_words=total_words,
        n_tokens=total_tokens,
        n_bytes=total_bytes,
        fertility=fertility,
        tokens_per_sentence=tokens_per_sentence,
        tokens_per_byte=tokens_per_byte,
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


def gini(values: list[float]) -> float:
    """Gini coefficient of a non-negative distribution.

    0.0 = perfect equality across all values; approaches 1.0 as inequality
    grows.  Used in Foroutan et al. 2025 as the headline fairness metric
    (Gini of tokens-per-parallel-sentence across languages).
    """
    v = sorted(float(x) for x in values)
    n = len(v)
    if n == 0:
        return 0.0
    s = sum(v)
    if s <= 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(v))
    return (2.0 * cum) / (n * s) - (n + 1) / n


class ParityReport(NamedTuple):
    stats: list[LanguageStats]

    # Parallel-sentence metrics (primary).
    mean_tokens_per_sentence: float
    std_tokens_per_sentence: float
    max_tokens_per_sentence: float
    min_tokens_per_sentence: float
    tokens_per_sentence_ratio: float     # max/min  (1.0 = perfect parity)
    gini_tokens_per_sentence: float      # 0.0 = perfect equality

    # Byte-normalized metrics (secondary).
    gini_tokens_per_byte: float

    # Legacy fertility (unreliable for CJK/Thai, retained for backward compat).
    mean_fertility: float
    std_fertility: float
    max_fertility: float
    min_fertility: float
    fertility_ratio: float


def compute_parity_report(
    tokenizer: Tokenizer,
    flores_dir: Path,
    languages: list[str],
) -> ParityReport:
    """Evaluate parity across all given languages using parallel FLORES+ data.

    Args:
        tokenizer:  Tokenizer to evaluate.
        flores_dir: Directory with per-language FLORES+ JSONL files.  The
                    sentences *must* be parallel across languages (FLORES+
                    devtest satisfies this).
        languages:  Language codes to include.

    Returns:
        A :class:`ParityReport` with per-language stats and aggregate metrics.
    """
    stats: list[LanguageStats] = []
    for lang in languages:
        sentences = load_flores_sentences(flores_dir, lang)
        s = compute_language_stats(tokenizer, sentences, lang)
        stats.append(s)
        print(f"  [{lang:>4}] tok/sent={s.tokens_per_sentence:6.2f}  "
              f"tok/byte={s.tokens_per_byte:.4f}  "
              f"unk={s.unk_rate:.4f}  sentences={s.n_sentences}")

    valid = [s for s in stats if s.n_sentences > 0]
    if not valid:
        raise ValueError("No languages with valid sentences to aggregate.")

    tps = [s.tokens_per_sentence for s in valid]
    tpb = [s.tokens_per_byte for s in valid]
    fert = [s.fertility for s in valid]

    min_tps = min(tps)
    max_tps = max(tps)

    return ParityReport(
        stats=stats,
        mean_tokens_per_sentence=float(np.mean(tps)),
        std_tokens_per_sentence=float(np.std(tps)),
        max_tokens_per_sentence=max_tps,
        min_tokens_per_sentence=min_tps,
        tokens_per_sentence_ratio=max_tps / min_tps if min_tps > 0 else math.inf,
        gini_tokens_per_sentence=gini(tps),
        gini_tokens_per_byte=gini(tpb),
        mean_fertility=float(np.mean(fert)),
        std_fertility=float(np.std(fert)),
        max_fertility=max(fert),
        min_fertility=min(fert),
        fertility_ratio=max(fert) / min(fert) if min(fert) > 0 else math.inf,
    )


def report_to_dict(report: ParityReport) -> dict:
    """Serialise a :class:`ParityReport` to a JSON-friendly dict."""
    return {
        "summary": {
            # Primary metrics (parallel-sentence).
            "mean_tokens_per_sentence": round(report.mean_tokens_per_sentence, 4),
            "std_tokens_per_sentence": round(report.std_tokens_per_sentence, 4),
            "max_tokens_per_sentence": round(report.max_tokens_per_sentence, 4),
            "min_tokens_per_sentence": round(report.min_tokens_per_sentence, 4),
            "tokens_per_sentence_ratio": round(report.tokens_per_sentence_ratio, 4),
            "gini_tokens_per_sentence": round(report.gini_tokens_per_sentence, 4),
            "gini_tokens_per_byte": round(report.gini_tokens_per_byte, 4),
            # Legacy fertility (unreliable for CJK/Thai).
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
                "n_bytes": s.n_bytes,
                "tokens_per_sentence": round(s.tokens_per_sentence, 4),
                "tokens_per_byte": round(s.tokens_per_byte, 6),
                "fertility": round(s.fertility, 4),
                "avg_token_len": round(s.avg_token_len, 4),
                "unk_rate": round(s.unk_rate, 6),
            }
            for s in sorted(report.stats, key=lambda x: x.tokens_per_sentence)
        ],
    }
