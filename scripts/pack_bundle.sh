#!/usr/bin/env bash
# Build the cloud-upload bundle for Stage 2.
#
# Bundles:
#   - source code (src/, scripts/, pyproject.toml, uv.lock)
#   - sampled mC4 (data/raw/mc4_sampled/)  — sampled here if missing
#   - FLORES+    (data/raw/flores/)
#   - pre-trained tokenizers (models/tokenizers/) so Stage 1 doesn't repeat
#   - Stage 1 parity reports (results/parity/) for record-keeping
#
# Excludes:
#   - .git, .venv, __pycache__
#   - models/lm/ (cloud will build these)
#   - results/downstream*/ (cloud will build these)
#   - data/raw/mc4/ (the unsampled 345 GB original)
#
# Usage:
#   bash scripts/pack_bundle.sh                                  # default 10M docs
#   TOTAL_DOCS=15000000 bash scripts/pack_bundle.sh              # 15M (~50 GB)
#   BUNDLE_PATH=/tmp/paat_bundle.tar.zst bash scripts/pack_bundle.sh
#   SKIP_SAMPLE=1 bash scripts/pack_bundle.sh                    # use existing sample

set -euo pipefail

TOTAL_DOCS="${TOTAL_DOCS:-10000000}"
SAMPLED_DIR="${SAMPLED_DIR:-data/raw/mc4_sampled}"
FLORES_DIR="${FLORES_DIR:-data/raw/flores}"
TOK_DIR="${TOK_DIR:-models/tokenizers}"
PARITY_DIR="${PARITY_DIR:-results/parity}"
BUNDLE_PATH="${BUNDLE_PATH:-paat_bundle.tar.zst}"
SKIP_SAMPLE="${SKIP_SAMPLE:-0}"

PYTHON="${PYTHON:-.venv/bin/python}"

cat <<EOF

================================================================
  Pack Stage 2 cloud bundle
  total docs:    $(printf "%'d" $TOTAL_DOCS)
  sampled dir:   $SAMPLED_DIR
  bundle path:   $BUNDLE_PATH
================================================================
EOF

# ── Preflight ──────────────────────────────────────────────────────────────
err() { echo "[FATAL] $*" >&2; exit 1; }

# zstd is essential — fail fast if it's not available.
command -v zstd >/dev/null 2>&1 || err "zstd not installed.  apt-get install zstd"

# Tokenizers must exist locally (Stage 1 should already be done).
expected_tokenizers=(
    "$TOK_DIR/bpe/tokenizer.json"
    "$TOK_DIR/parity_bpe/tokenizer.json"
    "$TOK_DIR/adat/adat/tokenizer.json"
    "$TOK_DIR/adat/baseline/tokenizer.json"
    "$TOK_DIR/paat_a10/paat/tokenizer.json"
    "$TOK_DIR/paat_a100_l0/paat/tokenizer.json"
)
missing_toks=()
for t in "${expected_tokenizers[@]}"; do
    [[ -f "$t" ]] || missing_toks+=("$t")
done
if (( ${#missing_toks[@]} > 0 )); then
    echo "[WARN] some tokenizers missing — Stage 2 on the pod will skip them:"
    for t in "${missing_toks[@]}"; do echo "        $t"; done
    echo "       Run scripts/exp_1_tokenizers.sh first if you want all of them."
    echo "       Continuing with what's available ..."
fi

# FLORES is small but required for parity-eval rerun on the pod.
[[ -d "$FLORES_DIR" ]] || err "$FLORES_DIR not found — run scripts/download_flores.py"

# ── 1. Sample mC4 (skip if cached) ────────────────────────────────────────
if [[ "$SKIP_SAMPLE" == "1" ]]; then
    echo ""
    echo "[1/3] SKIP_SAMPLE=1 — assuming $SAMPLED_DIR is already populated"
    [[ -d "$SAMPLED_DIR" ]] || err "$SAMPLED_DIR missing.  Drop SKIP_SAMPLE=1 to sample."
elif [[ -d "$SAMPLED_DIR" && $(find "$SAMPLED_DIR" -maxdepth 1 -name '*.jsonl' | wc -l) -ge 90 ]]; then
    n_existing=$(wc -l "$SAMPLED_DIR"/*.jsonl 2>/dev/null | tail -1 | awk '{print $1}')
    echo ""
    echo "[1/3] $SAMPLED_DIR exists with ~$n_existing total docs — reusing"
    echo "       (delete the dir or set SKIP_SAMPLE=0 + remove it to resample)"
else
    echo ""
    echo "[1/3] Sampling mC4 to $SAMPLED_DIR ($(printf "%'d" $TOTAL_DOCS) docs) ..."
    "$PYTHON" scripts/sample_mc4.py \
        --total-docs "$TOTAL_DOCS" \
        --output-dir "$SAMPLED_DIR"
fi

# ── 2. Bundle ──────────────────────────────────────────────────────────────
echo ""
echo "[2/3] Building bundle (zstd -19) ..."

# We bundle the source code too so the pod doesn't need to git-clone — keeps
# the cloud workflow self-contained and avoids "did I push?" confusion.
# Listing the components by name so an accidental rename of an excluded path
# doesn't silently include the original.

components=(
    src
    scripts
    pyproject.toml
    uv.lock
    README.md
    "$SAMPLED_DIR"
    "$FLORES_DIR"
)
[[ -d "$TOK_DIR"    ]] && components+=("$TOK_DIR")
[[ -d "$PARITY_DIR" ]] && components+=("$PARITY_DIR")

# Fresh-build the bundle.  --owner/--group nuke the local UID so the
# extraction on the pod doesn't try to chown to a UID that doesn't exist.
tar --owner=0 --group=0 \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.DS_Store' \
    -cf - \
    "${components[@]}" \
  | zstd -T0 -19 -o "$BUNDLE_PATH" --force

# ── 3. Report + checksum ───────────────────────────────────────────────────
sha256sum "$BUNDLE_PATH" > "${BUNDLE_PATH}.sha256"
size_h=$(du -h "$BUNDLE_PATH" | awk '{print $1}')

echo ""
echo "[3/3] Bundle built."
echo "  path:       $BUNDLE_PATH"
echo "  size:       $size_h"
echo "  checksum:   ${BUNDLE_PATH}.sha256"
echo ""
echo "Next:  bash scripts/cloud_launch.sh    # uploads + launches Stage 2"
