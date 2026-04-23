# Parity-aware Adaptive Tokenization (PAAT)

**Team:** Tiger Chaisutyakorn, Brianna Grissom, Rianna Santra, Nour Massri

## Overview

PAAT combines two ideas to produce a tokenizer that is both performance-aware and cross-lingually fair:

1. **Adaptive tokenization (ADAT)** — start with a large vocabulary, iteratively train a small LLM on a data subset, compute per-token loss, and prune tokens by a combined frequency + loss score.
2. **Parity-aware loss** — augment the pruning objective with a parity loss that penalizes compression inequality across languages, ensuring the vocabulary does not systematically over-segment low-resource scripts.

## Repository Layout

```
src/paat/
  data/        # language registry (96 mC4 ∩ FLORES+ languages)
  tokenizer/   # BPE, Unigram, and ADAT iterative pruning
  model/       # tiny GPT-2 and training loop for the ADAT inner model
  parity/      # per-language compression and fairness metrics (Gini, tok/sent)
scripts/       # entry points for each step in the experiment
configs/       # YAML/JSON configs for experiments
data/          # download outputs (raw data not tracked)
docs/          # experiment write-ups
results/       # JSON metric reports
```

## Baselines

| Method | Purpose |
|---|---|
| Classic BPE (32K) | Fairness + PPL reference |
| Direct Unigram (16K) | Same-size baseline vs ADAT |
| ADAT 16K (Zheng et al., 2024) | Performance-oriented reference |
| Parity-aware BPE (Foroutan et al., 2025) | Fairness-oriented reference |

## Datasets

- **Training:** mC4 multilingual corpus — 96 languages (the overlap with FLORES+), preserving the natural resource distribution.
- **Parity evaluation:** FLORES+ (2,009 parallel sentences across all 96 languages).

## Requirements

- Python 3.12
- CUDA GPU (tested on NVIDIA RTX A6000 50 GB)
- ~1 GB disk for demo data, ~178 GB for the full mC4 download

## Setup

```bash
pip install -e ".[dev]"
# or, with uv:
uv sync
```

Verify CUDA is available:

```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## End-to-End Experiment Recipe

Every command below is run from the repository root. Commands are idempotent where possible — re-running will skip completed steps.

### Step 1 — Download evaluation data (FLORES+)

FLORES+ is small (~10 MB for all 96 languages) and is required for every parity evaluation. Run this first.

```bash
python scripts/download_flores.py
```

Output: `data/raw/flores/<lang>.jsonl` (2,009 parallel sentences per language, `dev` + `devtest` splits).

### Step 2 — Download training data (mC4)

Choose one of the two download modes. Start with demo mode to validate the pipeline; switch to full mode for the real experiment.

**Demo mode** — 5,000 docs per language × 96 languages ≈ 1 GB:

```bash
python scripts/download_mc4.py --demo
```

**Full mode** — natural resource distribution per `MC4_NATURAL_COUNTS` ≈ 178 GB:

```bash
python scripts/download_mc4.py
```

**Subset** — restrict to specific languages:

```bash
python scripts/download_mc4.py --demo --languages en zh ar hi sw am
```

Output: `data/raw/mc4/<lang>.jsonl` (one JSON object per line: `{"text": ...}`). Files are streamed from HuggingFace, so no full-dataset download is materialised — each language is fetched independently.

### Step 3 — Train the classic BPE baseline (32K)

```bash
python scripts/train_tokenizer.py \
    --data-dir data/raw/mc4 \
    --output-dir models/tokenizers/bpe_demo \
    --vocab-size 32000
```

Output: `models/tokenizers/bpe_demo/tokenizer.json`.

### Step 4 — Evaluate BPE baseline parity

```bash
python scripts/eval_parity.py \
    --tokenizer models/tokenizers/bpe_demo \
    --flores-dir data/raw/flores \
    --output results/parity/bpe_demo.json
```

This prints a per-language table of tokens-per-sentence, tokens-per-byte, and legacy fertility, plus aggregate Gini-based fairness metrics.

### Step 5 — Run ADAT Phase 1 (32K → 16K, 3 iterations)

This is the main experiment. It:

1. Trains an initial 32K SentencePiece Unigram on the training split.
2. Runs 3 iterations of ADAT pruning (32K → 26.7K → 21.3K → 16K). Each iteration trains a tiny GPT-2 from scratch, computes per-token cross-entropy on a held-out slice, combines with the Unigram piece score, and keeps the top-K pieces.
3. Trains a direct 16K Unigram baseline on the same data for a same-size comparison.
4. Trains both tokenizers' LLMs to convergence and reports held-out PPL.

```bash
python scripts/run_adat.py \
    --data-dir data/raw/mc4 \
    --output-dir models/tokenizers/adat_phase1 \
    --initial-vocab 32000 \
    --target-vocab 16000 \
    --iterations 3 \
    --docs-per-lang 2000 \
    --train-tokens-per-iter 5000000 \
    --eval-tokens-per-iter 2000000 \
    --seq-len 512 \
    --batch-size 32 \
    --model-size tiny
```

Approximate wall time on an A6000: ~40 minutes for the ADAT loop + ~15 minutes for the baseline Unigram + ~15 minutes for the PPL comparison.

Outputs:

| Path | Contents |
|---|---|
| `models/tokenizers/adat_phase1/sp_init/sp.model` | Initial 32K SentencePiece Unigram |
| `models/tokenizers/adat_phase1/adat/iter_{01,02,03}_vocab*.json` | Intermediate ADAT tokenizers |
| `models/tokenizers/adat_phase1/adat/tokenizer.json` | Final ADAT 16K tokenizer |
| `models/tokenizers/adat_phase1/adat/adat_log.json` | Per-iteration training log |
| `models/tokenizers/adat_phase1/baseline/tokenizer.json` | Direct 16K Unigram baseline |
| `models/tokenizers/adat_phase1/comparison.json` | PPL comparison summary |

**Micro-test** — before running the full Phase 1, validate the pipeline end-to-end with a 5-minute run:

```bash
python scripts/run_adat.py \
    --data-dir data/raw/mc4 \
    --output-dir models/tokenizers/adat_microtest \
    --initial-vocab 8000 \
    --target-vocab 6000 \
    --iterations 2 \
    --docs-per-lang 50 \
    --train-tokens-per-iter 200000 \
    --eval-tokens-per-iter 50000 \
    --seq-len 128 \
    --batch-size 16 \
    --model-size tiny
```

### Step 6 — Evaluate ADAT and baseline parity

Run the parity eval on both tokenizers produced in Step 5:

```bash
python scripts/eval_parity.py \
    --tokenizer models/tokenizers/adat_phase1/adat \
    --flores-dir data/raw/flores \
    --output results/parity/adat_phase1.json

python scripts/eval_parity.py \
    --tokenizer models/tokenizers/adat_phase1/baseline \
    --flores-dir data/raw/flores \
    --output results/parity/unigram_baseline_16k.json
```

### Step 7 — Inspect results

```bash
# PPL comparison (ADAT vs same-size Unigram baseline)
cat models/tokenizers/adat_phase1/comparison.json

# Parity reports — look at "gini_tokens_per_sentence" and
# "tokens_per_sentence_ratio" for the headline fairness numbers.
cat results/parity/adat_phase1.json | python -m json.tool | head -40
cat results/parity/unigram_baseline_16k.json | python -m json.tool | head -40
cat results/parity/bpe_demo.json | python -m json.tool | head -40
```

See [docs/adat-phase1-results.md](docs/adat-phase1-results.md) for the full Phase 1 analysis.

---

## Metrics

Tokenizer fairness is reported using metrics from Foroutan et al. 2025:

- **tokens_per_sentence** — average tokens produced per parallel FLORES+ sentence. Sentence *i* in every language is a translation of the same source, so this compares like-for-like content. Robust to script and whitespace differences (works for Chinese, Japanese, Thai, etc., where `str.split()` is meaningless).
- **tokens_per_byte** — tokens per UTF-8 byte. Accounts for per-script byte length (ASCII vs CJK UTF-8).
- **Gini (tokens/sentence)** — Gini coefficient of per-language tokens-per-sentence. 0 = perfect equality, 1 = maximum inequality. The headline fairness metric.
- **Legacy fertility** (tokens/word) — retained for backward comparability but unreliable for languages without whitespace word boundaries.

## References

- Foroutan et al. (2025). Parity-Aware Byte-Pair Encoding. arXiv:2508.04796
- Zheng et al. (2024). ADAT: Adaptive Tokenization. NeurIPS 2024.
- Petrov et al. (2023). Language Model Tokenizers Introduce Unfairness Between Languages. arXiv:2305.15425
- Rust et al. (2021). How Good is Your Tokenizer? ACL 2021.
- Sun et al. (2026). LiteToken. arXiv:2602.04706
