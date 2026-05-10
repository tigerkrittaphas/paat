# Parity-Aware Adaptive Tokenization (PAAT)

**Team:** Tiger Chaisutyakorn, Brianna Grissom, Rianna Santra, Nour Massri

PAAT extends the ADAT tokenizer pipeline (Zheng et al. 2024) with a parity-aware loss term that closes the cross-lingual compression gap ADAT introduces. This repo contains the full two-stage experiment: Stage 1 trains and evaluates tokenizers; Stage 2 pretrains one Pythia-70M LM per tokenizer and measures downstream fairness via held-out perplexity across 96 FLORES+ languages.

See [`docs/final-report.md`](docs/final-report.md) for full results.

---

## Repository layout

```
src/paat/           # library: tokenizer, parity metrics, model
scripts/            # experiment entry points (see below)
configs/            # hyperparameter YAML/JSON files
data/               # downloaded data (not tracked by git)
models/             # tokenizers/ and lm/ outputs (not tracked)
results/            # JSON metric reports
results_pi/         # metric reports pulled from cloud runs
docs/               # write-ups and final report
notebooks/          # analysis notebooks
```

---

## Requirements

- Python 3.11–3.13
- CUDA GPU — tested on RTX A6000 (local) and A100/H100 80 GB (cloud)
- [`uv`](https://docs.astral.sh/uv/) for dependency management
- `zstd` and `tmux` (cloud path only)

---

## Setup

```bash
# Install uv (once)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create venv and install all dependencies
uv sync

# Verify CUDA
uv run python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

---

## Running locally

### 0. Download evaluation data (FLORES+)

Required before anything else. Small (~10 MB for 96 languages).

```bash
uv run python scripts/download_flores.py
# → data/raw/flores/<lang>.jsonl
```

### 1. Download and sample mC4

```bash
# Streaming download — keeps only a proportional sample, never stores the full corpus.
uv run python scripts/sample_mc4.py
# → data/raw/mc4_sampled/<lang>.jsonl   (~8 M docs, ~30 GB)

# Convenience symlink used by training scripts
ln -sfn mc4_sampled data/raw/mc4
```

For a quick smoke test, pass `--total-docs 100000` to sample_mc4.py.

### 2. Stage 1 — train tokenizers and evaluate parity

```bash
# Full run (standard preset — all 8 tokenizers, ~4–6 h on A6000)
bash scripts/exp_1_tokenizers.sh

# Smoke test (~10 min, small vocab)
PRESET=smoke bash scripts/exp_1_tokenizers.sh

# Custom alpha sweep or subset
ALPHAS="0.33 0.67 1.0" bash scripts/exp_1_tokenizers.sh
SKIP_PARITY_BPE=1 bash scripts/exp_1_tokenizers.sh   # skip the slow parity-BPE step
SKIP_TRAIN=1 bash scripts/exp_1_tokenizers.sh         # parity eval only (tokenizers exist)
```

Outputs:
```
models/tokenizers/{bpe,parity_bpe,adat,unigram,paat_a033,paat_a067,paat_a10,paat_a100_l0}/
results/parity/{bpe,parity_bpe,adat,unigram,paat_a033,...}.json
results/parity/comparison.json
```

### 3. Stage 2 — pretrain LMs and run downstream eval

Each model takes ~10 h at `standard` (1 B tokens) or ~30 h at `long` (3 B tokens) on a single A6000.

```bash
# 1 B-token run (standard)
PRESET=standard bash scripts/exp_2_pretrain_eval.sh

# 3 B-token run (long) — used for the paper
PRESET=long bash scripts/exp_2_pretrain_eval.sh

# Subset of models only
MODELS="adat paat_a10 paat_a100_l0" bash scripts/exp_2_pretrain_eval.sh

# Skip downstream eval (pretrain only) or vice versa
SKIP_EVAL=1    bash scripts/exp_2_pretrain_eval.sh
SKIP_PRETRAIN=1 bash scripts/exp_2_pretrain_eval.sh
```

Outputs:
```
models/lm/{bpe,parity_bpe,adat,unigram,paat_a033,...}/
results/downstream/comparison.json
```

### 4. Held-out perplexity across 96 languages

```bash
uv run python scripts/eval_perplexity.py
# → results/perplexity/perplexity.json
# → results/perplexity/perplexity_summary.json

# Subset of models or languages
uv run python scripts/eval_perplexity.py --models adat paat_a10 --languages en zh ar hi
```

---

## Running on Prime Intellect cloud

The cloud path offloads Stage 2 to an A100-80GB pod. Stage 1 is run locally first (tokenizers are small; the inner LLM only needs ~10 min/iteration on CPU-class hardware).

### SSH alias (one-time)

Add to `~/.ssh/config`:

```
Host pi-paat
    HostName <pod-ip>
    User root
    IdentityFile ~/.ssh/id_ed25519
    Port 22
```

Replace `<pod-ip>` with the address shown in the Prime Intellect dashboard. Update the alias when you spin up a new pod.

### Option A — Bundle upload (simplest, ~9 GB transfer)

Pack everything the pod needs into a single compressed archive, upload, and launch.

```bash
# 1. Build the bundle (includes tokenizers + mC4 sample + FLORES+)
bash scripts/pack_bundle.sh
# → paat_bundle.tar.zst  (~9 GB)

# 2. Upload and launch Stage 2
bash scripts/cloud_launch.sh

# Override preset or model list
PRESET=standard bash scripts/cloud_launch.sh
MODELS="adat paat_a10 paat_a100_l0" bash scripts/cloud_launch.sh

# Dry run — preflight only, no upload
DRY_RUN=1 bash scripts/cloud_launch.sh
```

### Option B — Persistent volume (fast subsequent runs, ~few MB per launch)

Upload data once to a PI persistent volume; future runs only ship source code.

```bash
# 1. Create a persistent volume in the PI dashboard and attach it to any pod.
#    Update ~/.ssh/config with the new pod's IP.

# 2. Upload data + tokenizers to the volume (one time, ~30 min)
bash scripts/init_data_volume.sh
# → destroys the upload pod after; the volume persists

# 3. All future launches skip the bundle — rsync source code only
USE_VOLUME=1 bash scripts/cloud_launch.sh
USE_VOLUME=1 PRESET=long bash scripts/cloud_launch.sh
```

### Monitor and retrieve results

```bash
# Attach to the tmux session on the pod
ssh pi-paat tmux attach -t paat-exp2
# Detach without killing: Ctrl-b d

# Live log tail without attaching
ssh pi-paat tail -f /workspace/paat/logs/exp_2/cloud_run.log

# Pull results back when done
rsync -avP --exclude='checkpoint-*' \
    pi-paat:/workspace/paat/models/lm/ ./models/lm_pi/
rsync -avP pi-paat:/workspace/paat/results/ ./results_pi/
rsync -avP pi-paat:/workspace/paat/logs/   ./logs_pi/

# Run perplexity eval locally on the pulled checkpoints
uv run python scripts/eval_perplexity.py \
    --models bpe parity_bpe unigram adat paat_a033 paat_a067 paat_a10 paat_a100_l0
# → results_pi/perplexity/perplexity.json
```

**Cost estimates (A100-80GB at ~$1.80/hr, 6 models):**

| Preset | Tokens/model | Wall time | Est. cost |
|---|---|---|---|
| `smoke` | ~10 M | ~10 min | <$1 |
| `standard` | 1 B | ~9 h | ~$15 |
| `long` | 3 B | ~25 h | ~$45 |

Destroy the pod from the PI dashboard when done — it bills by the second.

---

## Key environment variables

| Variable | Default | Effect |
|---|---|---|
| `PRESET` | `standard` | Token budget: `smoke` / `standard` / `long` |
| `MODELS` | all 6 | Space-separated model names to train |
| `ALPHAS` | `0.33 0.67 1.0` | PAAT parity weights for Stage 1 |
| `SKIP_EVAL` | `0` | Skip downstream eval in Stage 2 |
| `SKIP_PRETRAIN` | `0` | Skip LM training in Stage 2 |
| `LM_ROOT` | `models/lm` | Output dir for LMs (set to `models/lm_pi` for cloud pulls) |
| `PI_HOST` | `pi-paat` | SSH alias for the cloud pod |
| `USE_VOLUME` | `0` | Use persistent volume instead of bundle upload |
| `DRY_RUN` | `0` | Preflight only, no upload or launch |
| `AUTO_PACK` | `0` | Build bundle automatically if missing |

---

## References

- Zheng et al. (2024). ADAT: Adaptive Tokenization. *NeurIPS 2024*.
- Foroutan et al. (2025). Parity-Aware Byte-Pair Encoding. *arXiv:2508.04796*.
- Petrov et al. (2023). Language Model Tokenizers Introduce Unfairness Between Languages. *arXiv:2305.15425*.
- Rust et al. (2021). How Good is Your Tokenizer? *ACL 2021*.
