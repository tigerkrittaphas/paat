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
# Make sure $PATH already includes the canonical install dirs so a previous
# install (e.g. partial-success from a prior run) is detected.
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
if ! command -v uv >/dev/null 2>&1; then
    # UV_INSTALLER_NO_MODIFY_PATH=1 stops the installer from writing to
    # ~/.bashrc / ~/.config/fish/conf.d / etc.  On stripped pod images the
    # fish-config write fails ("Permission denied" on ~/.config/fish) and
    # the whole installer exits non-zero — even though uv itself installed
    # fine.  We don't need shell rc edits anyway since we just exported PATH.
    curl -LsSf https://astral.sh/uv/install.sh \
        | env UV_INSTALLER_NO_MODIFY_PATH=1 sh
fi
command -v uv >/dev/null 2>&1 || {
    echo "[FATAL] uv install completed but binary is not on PATH." >&2
    echo "        Looked under: $HOME/.local/bin and $HOME/.cargo/bin" >&2
    exit 1
}
uv --version

# ── 3. Python environment ──────────────────────────────────────────────────
echo ""
echo "[3/5] Syncing Python environment via uv ..."
uv sync                       # installs from pyproject + uv.lock into .venv
echo "  python: $(.venv/bin/python --version)"

# ── 4. CUDA / torch sanity ─────────────────────────────────────────────────
echo ""
echo "[4/5] Verifying CUDA + torch ..."

# Run the check; if cuda is unavailable AND nvidia-smi sees a GPU, it's
# almost always a torch/driver mismatch — uv pulled a torch wheel built
# for a newer CUDA than the pod's driver supports.  Auto-heal by
# reinstalling torch from the cu121 wheel index (forward-compatible with
# 12.x drivers).
torch_check() {
    .venv/bin/python - <<'PY'
import sys
try:
    import torch
    ok = torch.cuda.is_available()
    print(f"torch: {torch.__version__}  cuda: {ok}")
    if ok:
        for i in range(torch.cuda.device_count()):
            p = torch.cuda.get_device_properties(i)
            print(f"  gpu{i}: {p.name}  {p.total_memory / 1e9:.1f} GB")
        sys.exit(0)
except Exception as e:
    print(f"torch import failed: {e}")
sys.exit(1)
PY
}

if ! torch_check; then
    if command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
        echo ""
        echo "[heal] CUDA unavailable but nvidia-smi sees a GPU — reinstalling"
        echo "       torch from cu121 wheel index (compatible with driver 12.x) ..."
        uv pip install --reinstall torch \
            --index-url https://download.pytorch.org/whl/cu121
        echo ""
        echo "[heal] retrying CUDA check ..."
        torch_check || {
            echo "[FATAL] torch still cannot see CUDA after cu121 reinstall." >&2
            echo "        Check: nvidia-smi" >&2
            exit 1
        }
    else
        echo "[FATAL] no CUDA device visible AND nvidia-smi sees no GPU." >&2
        echo "        Check the pod image — it may not have GPU passthrough." >&2
        exit 1
    fi
fi

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
    n_tok=$(find models/tokenizers -maxdepth 3 -name 'tokenizer.json' 2>/dev/null | wc -l)
    n_par=$(find results/parity -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
    size=$(du -sh data/raw models/tokenizers results/parity 2>/dev/null | tail -1 | awk '{print $1}')
    echo "  mC4 langs:        $n_mc4"
    echo "  FLORES langs:     $n_flo"
    echo "  tokenizers:       $n_tok"
    echo "  parity reports:   $n_par"
    echo "  total bundle size: $(du -sh data/raw models/tokenizers results/parity 2>/dev/null | awk '{s+=$1} END {print s}' | numfmt --to=iec --suffix=B 2>/dev/null || echo "see du -sh")"
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
