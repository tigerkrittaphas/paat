# Parity-aware Adaptive Tokenization (PAAT)

**Team:** Tiger Chaisutyakorn, Brianna Grissom, Rianna Santra, Nour Massri

## Overview

PAAT combines two ideas to produce a tokenizer that is both performance-aware and cross-lingually fair:

1. **Adaptive tokenization (ADAT)** — start with a large vocabulary, iteratively train a small LLM on a data subset, compute per-token loss, and prune tokens by a combined frequency + loss score.
2. **Parity-aware loss** — augment the pruning objective with a parity loss that penalizes compression inequality across languages, ensuring the vocabulary does not systematically over-segment low-resource scripts.

## Repository Layout

```
src/paat/
  tokenizer/   # vocabulary construction, BPE training, iterative pruning loop
  parity/      # per-language compression metrics and parity loss
  eval/        # perplexity and fertility evaluation utilities
scripts/       # entry points for training and evaluation runs
configs/       # YAML/JSON configs for experiments
data/          # download/preprocessing scripts (raw data not tracked)
notebooks/     # exploratory analysis
tests/
```

## Baselines

| Method | Goal |
|---|---|
| Standard BPE | Fairness + PPL reference |
| ADAT (reimplemented) | Performance reference |
| Parity-aware BPE (Foroutan et al., 2025) | Fairness reference |

## Datasets

- **Training:** mC4 multilingual corpus (15–20 languages across resource tiers)
- **Parity evaluation:** FLORES+ parallel benchmark

## Models

- Tokenizer optimization loop: Pythia-14M
- Final downstream evaluation: Pythia-70M
- Extended (compute permitting): Pythia-160M, Pythia-410M

## Setup

```bash
pip install -e ".[dev]"
```

## References

- Foroutan et al. (2025). Parity-Aware Byte-Pair Encoding. arXiv:2508.04796
- Zheng et al. (2024). ADAT: Adaptive Tokenization.
- Petrov et al. (2023). Language Model Tokenizers Introduce Unfairness Between Languages. arXiv:2305.15425
- Sun et al. (2026). LiteToken. arXiv:2602.04706
