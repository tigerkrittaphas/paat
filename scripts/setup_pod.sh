#!/usr/bin/env bash
# Bootstrap a Prime Intellect (or any Linux+CUDA) pod for PAAT experiments.
#
# Assumes you have already rsynced the data bundle and (optionally) the repo
# tarball to the pod.  Run this once on the pod, then launch the experiment.
#
# Quick path (from your laptop, before SSHing in):
#   rsync -avP mc4_bundle.tar.zst user@pod:/workspace/
#   ssh user@pod
#   git clone https://github.com/<you>/paat.git /workspace/paat
#   cd /workspace/paat && bash scripts/setup_pod.sh /workspace/mc4_bundle.tar.zst
#
# The bundle is whatever you produced with:
#   tar -cf - data/raw/mc4_sampled data/raw/flores | zstd -T0 -19 > mc4_bundle.tar.zst

set -euo pipefail

BUNDLE="${1:-}"
REPO_DIR="${REPO_DIR:-$(pwd)}"
SAMPLED_DIR_NAME="${SAMPLED_DIR_NAME:-mc4_sampled}"

echo "================================================================"
echo "  PAAT pod bootstrap"
echo "  repo:    $REPO_DIR"
echo "  bundle:  ${BUNDLE:-<none — will skip data unpack>}"
echo "================================================================"

cd "$REPO_DIR"

# ── 1. System packages ────────────────────────────────────────────────────
# zstd for unpacking the bundle, tmux for detachable runs, git in case the
# image is minimal.  apt-get is silent-on-noop so this is cheap to re-run.
if command -v apt-get >/dev/null 2>&1; then
    echo ""
    echo "[1/5] Installing system packages (zstd, tmux, git, build-essential) ..."
    sudo -n apt-get update -qq 2>/dev/null || apt-get update -qq
    sudo -n apt-get install -y -qq zstd tmux git build-essential 2>/dev/null \
        || apt-get install -y -qq zstd tmux git build-essential
else
    echo "[1/5] Skipping apt — not a Debian-family image."
fi

# ── 2. uv (project manager) ────────────────────────────────────────────────
echo ""
echo "[2/5] Ensuring uv is installed ..."
if ! command -v uv >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    # uv installs to ~/.local/bin or ~/.cargo/bin depending on platform
    export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
uv --version

# ── 3. Python environment ──────────────────────────────────────────────────
echo ""
echo "[3/5] Syncing Python environment via uv ..."
uv sync                       # installs from pyproject + uv.lock into .venv
echo "  python: $(.venv/bin/python --version)"

# ── 4. CUDA / torch sanity ─────────────────────────────────────────────────
echo ""
echo "[4/5] Verifying CUDA + torch ..."
.venv/bin/python - <<'PY'
import torch, sys
print(f"torch: {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    sys.exit("[FATAL] no CUDA device visible — check `nvidia-smi` and pod image.")
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"  gpu{i}: {p.name}  {p.total_memory / 1e9:.1f} GB")
PY

# ── 5. Data bundle ─────────────────────────────────────────────────────────
echo ""
if [[ -n "$BUNDLE" ]]; then
    if [[ ! -f "$BUNDLE" ]]; then
        echo "[FATAL] bundle not found: $BUNDLE" >&2
        exit 1
    fi
    echo "[5/5] Unpacking data bundle ..."
    mkdir -p data/raw
    # The bundle was packed from the repo root, so contents land under
    # data/raw/{mc4_sampled,flores}/ automatically.
    tar --use-compress-program=unzstd -xf "$BUNDLE" -C "$REPO_DIR"

    # Match what scripts expect: data/raw/mc4/.  If the bundle used the
    # `_sampled` suffix, symlink it so --data-dir defaults still work.
    if [[ -d "data/raw/$SAMPLED_DIR_NAME" && ! -e "data/raw/mc4" ]]; then
        ln -s "$SAMPLED_DIR_NAME" data/raw/mc4
        echo "  symlinked data/raw/mc4 -> $SAMPLED_DIR_NAME"
    fi

    n_mc4=$(find data/raw/mc4 -maxdepth 1 -name '*.jsonl' 2>/dev/null | wc -l)
    n_flo=$(find data/raw/flores -maxdepth 1 -name '*.jsonl' 2>/dev/null | wc -l)
    size=$(du -sh data/raw 2>/dev/null | awk '{print $1}')
    echo "  mC4 langs:    $n_mc4"
    echo "  FLORES langs: $n_flo"
    echo "  total size:   $size"
else
    echo "[5/5] No bundle path given — skipping data unpack."
fi

# ── Done ───────────────────────────────────────────────────────────────────
cat <<'EOF'

================================================================
  Setup complete.

  Smoke test (a few minutes):
    .venv/bin/python scripts/run_paat.py \
        --output-dir models/tokenizers/paat_smoke \
        --initial-vocab 8000 --target-vocab 4000 --iterations 2 \
        --total-docs 50000 \
        --train-tokens-per-iter 1000000 \
        --eval-tokens-per-iter 200000 \
        --parity-alpha 1.0

  Full PAAT run (background, survives SSH disconnect):
    tmux new -d -s paat \
      ".venv/bin/python scripts/run_paat.py \
         --output-dir models/tokenizers/paat_full \
         --initial-vocab 32000 --target-vocab 16000 --iterations 3 \
         --parity-alpha 1.0 \
         2>&1 | tee logs/paat_full.log"
    tmux attach -t paat        # detach: Ctrl-b d

  Pull results back to your laptop afterwards:
    rsync -avP user@pod:/workspace/paat/models/tokenizers/paat_full ./
    rsync -avP user@pod:/workspace/paat/results ./
================================================================
EOF
