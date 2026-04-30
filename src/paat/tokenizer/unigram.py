"""
Unigram tokenizer utilities for ADAT.

Initial vocabularies are trained with SentencePiece (fast, battle-tested
Unigram implementation), then converted to HuggingFace ``tokenizers``
Unigram models so we can easily rebuild the vocabulary after each ADAT
pruning step by passing an explicit ``(piece, score)`` list.
"""

from __future__ import annotations

import json
from pathlib import Path

import sentencepiece as spm
from tokenizers import Tokenizer, decoders, normalizers, pre_tokenizers
from tokenizers.models import Unigram
from tokenizers.processors import TemplateProcessing


SPECIAL_TOKENS = ["<unk>", "<s>", "</s>", "<pad>"]
UNK_ID = 0


def train_unigram_sentencepiece(
    texts: list[str],
    output_dir: Path,
    vocab_size: int,
    input_sentence_size: int = 500_000,
    character_coverage: float = 0.9995,
) -> Path:
    """Train an initial Unigram tokenizer with SentencePiece.

    Args:
        texts:                  List of documents to train on.
        output_dir:             Where to write ``sp.model`` / ``sp.vocab``.
        vocab_size:             Desired vocabulary size.
        input_sentence_size:    Cap on SentencePiece sampling (RAM control).
        character_coverage:     SentencePiece coverage parameter (lower for
                                scripts with many rare characters).

    Returns:
        Path to the ``sp.model`` file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_corpus = output_dir / "corpus.txt"

    print(f"[unigram-init] writing corpus to {tmp_corpus} ...")
    n_lines = 0
    with tmp_corpus.open("w", encoding="utf-8") as fh:
        for text in texts:
            # Strip embedded null bytes — mC4 contains them and SentencePiece
            # otherwise spams "Found null character" INFO logs (harmless but noisy).
            fh.write(text.replace("\0", "").replace("\n", " ") + "\n")
            n_lines += 1
    print(f"[unigram-init] corpus has {n_lines:,} lines")

    model_prefix = str(output_dir / "sp")
    print(f"[unigram-init] training Unigram vocab_size={vocab_size:,} ...")
    spm.SentencePieceTrainer.train(
        input=str(tmp_corpus),
        model_prefix=model_prefix,
        model_type="unigram",
        vocab_size=vocab_size,
        character_coverage=character_coverage,
        input_sentence_size=input_sentence_size,
        shuffle_input_sentence=True,
        # Reserve standard special tokens.
        unk_id=0,
        bos_id=1,
        eos_id=2,
        pad_id=3,
        unk_piece="<unk>",
        bos_piece="<s>",
        eos_piece="</s>",
        pad_piece="<pad>",
        # Stability / speed settings.
        num_threads=16,
        normalization_rule_name="nmt_nfkc",
    )

    # Corpus is large; remove after training to reclaim disk.
    tmp_corpus.unlink()
    return Path(f"{model_prefix}.model")


def sentencepiece_to_hf_unigram(sp_model_path: Path) -> Tokenizer:
    """Convert a SentencePiece Unigram model to a HuggingFace Tokenizer.

    HF's ``tokenizers.models.Unigram`` takes a ``(piece, score)`` list
    directly, which means we can rebuild it after pruning without
    retraining via SentencePiece.

    Args:
        sp_model_path: Path to ``<name>.model`` file from SentencePiece.

    Returns:
        A fully configured HF :class:`Tokenizer`.
    """
    sp = spm.SentencePieceProcessor(model_file=str(sp_model_path))
    pieces: list[tuple[str, float]] = []
    for i in range(sp.get_piece_size()):
        pieces.append((sp.id_to_piece(i), sp.get_score(i)))

    return build_hf_unigram(pieces)


def build_hf_unigram(pieces: list[tuple[str, float]]) -> Tokenizer:
    """Build a fully configured HF Unigram tokenizer from ``(piece, score)``.

    Expects the first four entries to be the special tokens
    (``<unk>``, ``<s>``, ``</s>``, ``<pad>``).  The Unigram model must
    know which id is the UNK id.
    """
    # SentencePiece uses the metaspace ('▁') convention — we keep that
    # so piece strings round-trip cleanly between SP and HF.
    tokenizer = Tokenizer(Unigram(vocab=pieces, unk_id=UNK_ID))
    tokenizer.normalizer = normalizers.NFKC()
    tokenizer.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="always")
    tokenizer.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="always")
    tokenizer.post_processor = TemplateProcessing(
        single="<s> $A </s>",
        special_tokens=[("<s>", 1), ("</s>", 2)],
    )
    return tokenizer


def get_pieces_with_scores(tokenizer: Tokenizer) -> list[tuple[str, float]]:
    """Extract the ``(piece, score)`` list from an HF Unigram tokenizer."""
    model_json = json.loads(tokenizer.to_str())
    vocab = model_json["model"]["vocab"]
    return [(piece, float(score)) for piece, score in vocab]
