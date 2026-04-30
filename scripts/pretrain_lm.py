"""
Pre-train a causal language model on mC4 with a given tokenizer.

Replicates the Pythia-70M experimental setup from Zheng et al. (NeurIPS 2024):
  6 layers, 512 hidden dim, 8 attention heads, lr=1e-3, batch=32.
Data is proportionally sampled mC4 (multilingual) instead of The Pile.
Output is saved in HuggingFace format, directly loadable by lm-evaluation-harness.

Usage:
    # Full Pythia-70M spec, 300M tokens (~3-4h on A6000)
    python scripts/pretrain_lm.py \\
        --tokenizer models/tokenizers/adat_full/adat \\
        --output-dir models/lm/adat \\
        --data-dir data/raw/mc4 \\
        --train-tokens 300000000

    # Quick validation (30 min)
    python scripts/pretrain_lm.py \\
        --tokenizer models/tokenizers/adat_full/adat \\
        --output-dir models/lm/adat_test \\
        --data-dir data/raw/mc4 \\
        --train-tokens 10000000
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from transformers import (
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    PreTrainedTokenizerFast,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)
from datasets import Dataset

from paat.data.languages import ALL_LANGUAGES, MC4_NATURAL_COUNTS
from paat.model.transformer import build_model


def _iter_docs_proportional(
    data_dir: Path, langs: list[str], total_docs: int,
) -> "Iterator[str]":
    """Yield mC4 docs interleaved across languages, proportional to natural counts.

    Streams from disk so RAM stays bounded by ``buffer_size`` open file
    handles plus one in-flight document, regardless of the total budget.
    Doc counts per language are pinned via the same formula as
    ``load_mc4_texts`` in :mod:`scripts.run_paat`.
    """
    total_natural = sum(MC4_NATURAL_COUNTS.get(l, 0) for l in langs)
    quotas: dict[str, int] = {}
    handles: dict[str, "object"] = {}
    for lang in langs:
        path = data_dir / f"{lang}.jsonl"
        if not path.exists():
            continue
        natural = MC4_NATURAL_COUNTS.get(lang, 1)
        quotas[lang] = max(1, round(total_docs * natural / total_natural))
        handles[lang] = path.open(encoding="utf-8")

    served: dict[str, int] = {l: 0 for l in handles}
    try:
        # Round-robin pull until every language is exhausted or quota hit.
        active = list(handles.keys())
        while active:
            still_active: list[str] = []
            for lang in active:
                if served[lang] >= quotas[lang]:
                    continue
                fh = handles[lang]
                line = fh.readline()
                if not line:                 # end of file
                    continue
                line = line.strip()
                if not line:
                    still_active.append(lang)
                    continue
                served[lang] += 1
                still_active.append(lang)
                yield json.loads(line)["text"]
            active = still_active
    finally:
        for fh in handles.values():
            fh.close()


def stream_pack_sequences(
    data_dir: Path,
    langs: list[str],
    total_docs: int,
    tokenizer: PreTrainedTokenizerFast,
    seq_len: int,
    max_tokens: int,
) -> list[list[int]]:
    """Tokenize on the fly and pack into ``seq_len`` blocks until budget hit.

    Replaces the old "load everything, then tokenize" pipeline.  The full
    Python list of documents never exists in memory — only the in-flight
    buffer (≤ seq_len ints) plus the packed sequences themselves
    (which are what we actually need).
    """
    buffer: list[int] = []
    sequences: list[list[int]] = []
    target_seqs = max_tokens // seq_len
    n_docs = 0
    for text in _iter_docs_proportional(data_dir, langs, total_docs):
        n_docs += 1
        ids = tokenizer.encode(text)
        buffer.extend(ids)
        while len(buffer) >= seq_len:
            sequences.append(buffer[:seq_len])
            buffer = buffer[seq_len:]
            if len(sequences) >= target_seqs:
                print(f"  [stream] {n_docs:,} docs -> {len(sequences):,} sequences "
                      f"({len(sequences) * seq_len:,} tokens)")
                return sequences
        if n_docs % 50_000 == 0:
            print(f"  [stream] {n_docs:,} docs -> {len(sequences):,} sequences "
                  f"({len(sequences) * seq_len:,} tokens)")
    print(f"  [stream] corpus exhausted at {n_docs:,} docs -> "
          f"{len(sequences):,} sequences  ({len(sequences) * seq_len:,} tokens)")
    return sequences


class TokenCountCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs and state.global_step % 500 == 0:
            tokens_seen = state.global_step * args.per_device_train_batch_size * args.max_seq_length if hasattr(args, "max_seq_length") else 0
            if "loss" in (logs or {}):
                print(f"  step {state.global_step:>6}  loss={logs['loss']:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Pre-train a causal LM (Pythia-70M spec) on multilingual mC4."
    )
    parser.add_argument("--tokenizer", type=Path, required=True,
                        help="Directory containing tokenizer.json.")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to save the trained model (HuggingFace format).")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw/mc4"))
    parser.add_argument("--train-tokens", type=int, default=300_000_000,
                        help="Token budget for training (default: 300M ≈ 3-4h on A6000).")
    parser.add_argument("--total-docs", type=int, default=500_000,
                        help="Docs loaded per run, distributed proportionally to mC4 counts.")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3,
                        help="Peak LR (paper uses 1e-3 for 70M model, Table 13).")
    parser.add_argument("--model-size", type=str, default="pythia70m",
                        help="Architecture size key: tiny | small | pythia70m | medium.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ── Tokenizer ──────────────────────────────────────────────────────────
    tok_path = args.tokenizer / "tokenizer.json"
    if not tok_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found in {args.tokenizer}")

    hf_tok = PreTrainedTokenizerFast(tokenizer_file=str(tok_path))
    # Reuse special tokens already present in the underlying tokenizer rather
    # than adding new ones — adding inflates the vocab on top of the base
    # tokenizer (the bug that produced 43,209-vocab BPE in the first round).
    # Unigram tokenizers carry  <unk> <s> </s> <pad>;  BPE carries
    # [UNK] [CLS] [SEP] [PAD] [MASK].  Map common aliases first; only fall
    # back to adding new pieces if the vocab really has nothing usable.
    base_vocab = hf_tok.get_vocab()
    role_aliases = {
        "pad_token": ["<pad>", "[PAD]"],
        "bos_token": ["<s>",   "[CLS]"],
        "eos_token": ["</s>",  "[SEP]"],
        "unk_token": ["<unk>", "[UNK]"],
    }
    for role, candidates in role_aliases.items():
        if getattr(hf_tok, role) is not None:
            continue
        for piece in candidates:
            if piece in base_vocab:
                setattr(hf_tok, role, piece)
                break
    # Last-resort additions (only if the tokenizer truly lacks the role).
    additions: dict[str, str] = {}
    for role, fallback in [("pad_token", "<|pad|>"),
                           ("bos_token", "<|bos|>"),
                           ("eos_token", "<|eos|>")]:
        if getattr(hf_tok, role) is None:
            additions[role] = fallback
    if additions:
        print(f"[WARN] tokenizer missing roles {list(additions)}; "
              f"adding new pieces (will grow vocab).")
        hf_tok.add_special_tokens(additions)

    vocab_size = len(hf_tok)  # equals base vocab unless `additions` fired

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_model(vocab_size=vocab_size, size=args.model_size)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model size:  {args.model_size}  ({n_params:,} parameters)")
    print(f"Vocab size:  {vocab_size:,}")

    # Resize embedding table if we added special tokens beyond the base vocab.
    model.resize_token_embeddings(vocab_size)

    # ── Data ───────────────────────────────────────────────────────────────
    print(f"\nStreaming mC4 (up to {args.total_docs:,} docs, "
          f"target {args.train_tokens:,} tokens) ...")
    sequences = stream_pack_sequences(
        args.data_dir, ALL_LANGUAGES, args.total_docs,
        hf_tok, args.seq_len, args.train_tokens,
    )
    actual_tokens = len(sequences) * args.seq_len
    if actual_tokens < args.train_tokens:
        print(f"[WARN] only packed {actual_tokens:,} tokens "
              f"(target {args.train_tokens:,}); raise --total-docs to feed more.")

    rng = random.Random(args.seed)
    rng.shuffle(sequences)
    split = max(1, int(len(sequences) * 0.005))
    train_seqs = sequences[split:]
    eval_seqs = sequences[:split]

    train_ds = Dataset.from_dict({"input_ids": train_seqs})
    eval_ds = Dataset.from_dict({"input_ids": eval_seqs})

    # ── Training ───────────────────────────────────────────────────────────
    max_steps = len(train_seqs) // args.batch_size
    warmup_steps = min(2000, max_steps // 20)

    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    use_fp16 = torch.cuda.is_available() and not use_bf16

    training_args = TrainingArguments(
        output_dir=str(args.output_dir),
        max_steps=max_steps,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        warmup_steps=warmup_steps,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_steps=200,
        save_steps=2000,
        eval_steps=1000,
        eval_strategy="steps",
        save_total_limit=2,
        bf16=use_bf16,
        fp16=use_fp16,
        dataloader_num_workers=4,
        seed=args.seed,
        report_to="none",
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=hf_tok, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    print(f"\nTraining for {max_steps:,} steps "
          f"(batch={args.batch_size}, seq={args.seq_len})...")
    print(f"LR={args.learning_rate}, warmup={warmup_steps} steps, cosine decay.\n")

    trainer.train()

    # ── Save ───────────────────────────────────────────────────────────────
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.output_dir))
    hf_tok.save_pretrained(str(args.output_dir))

    # Write a summary for easy reference.
    summary = {
        "tokenizer": str(args.tokenizer),
        "model_size": args.model_size,
        "n_params": n_params,
        "vocab_size": vocab_size,
        "train_tokens": actual_tokens,
        "seq_len": args.seq_len,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "max_steps": max_steps,
    }
    with (args.output_dir / "pretrain_summary.json").open("w") as f:
        json.dump(summary, f, indent=2)

    print(f"\nModel saved to {args.output_dir}")
    print(f"Summary: {args.output_dir}/pretrain_summary.json")


if __name__ == "__main__":
    main()
