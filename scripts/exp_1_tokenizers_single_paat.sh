#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Stage 1 (PAAT-only variant) — Train a single PAAT tokenizer and parity-eval.
#
#   PAAT (single α; default α = 1.0, λ = 0)  → models/tokenizers/paat_<tag>/
#
# Skips BPE, parity-aware BPE, ADAT, and the Unigram baseline.  Note that
# run_paat.py still trains an *initial* SentencePiece Unigram internally
# (the seed vocab that PAAT prunes from); that's part of PAAT itself, not a
# separate baseline tokenizer.
#
# Parity evaluation against FLORES+ runs on the resulting tokenizer:
#   results/parity/paat_<tag>.json
#
# ── Usage ─────────────────────────────────────────────────────────────────
#   bash scripts/exp_1_tokenizers_single_paat.sh             # default config
#   PRESET=smoke bash scripts/exp_1_tokenizers_single_paat.sh
#   ALPHA=0.5 bash scripts/exp_1_tokenizers_single_paat.sh   # different α
#   LAMBDA=1.0 bash scripts/exp_1_tokenizers_single_paat.sh  # re-enable LLM CE
#   SKIP_TRAIN=1 bash scripts/exp_1_tokenizers_single_paat.sh   # parity eval only
#   SKIP_PARITY=1 bash scripts/exp_1_tokenizers_single_paat.sh  # train only
#
# Re-running is idempotent — skips the PAAT training if it already exists.
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Presets ────────────────────────────────────────────────────────────────
PRESET="${PRESET:-standard}"
case "$PRESET" in
    smoke)
        : "${INITIAL_VOCAB:=8000}";   : "${TARGET_VOCAB:=4000}"
        : "${ITERATIONS:=2}";         : "${TOTAL_DOCS:=50000}"
        : "${TRAIN_TOK_PER_ITER:=1000000}"
        : "${EVAL_TOK_PER_ITER:=200000}"
        ;;
    standard)
        : "${INITIAL_VOCAB:=64000}";  : "${TARGET_VOCAB:=32000}"
        : "${ITERATIONS:=3}";         : "${TOTAL_DOCS:=500000}"
        : "${TRAIN_TOK_PER_ITER:=5000000}"
        : "${EVAL_TOK_PER_ITER:=2000000}"
        ;;
    *)
        echo "Unknown PRESET=$PRESET. Choose smoke|standard, or set knobs directly." >&2
        exit 1
        ;;
esac

# ── Config (override via env) ──────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-data/raw/mc4}"
FLORES_DIR="${FLORES_DIR:-data/raw/flores}"
TOK_ROOT="${TOK_ROOT:-models/tokenizers}"
PARITY_ROOT="${PARITY_ROOT:-results/parity}"
LOG_DIR="${LOG_DIR:-logs/exp_1}"

ALPHA="${ALPHA:-1.0}"
LAMBDA="${LAMBDA:-0}"      # λ = 0 → parity-bonus-only (no LLM CE term)

SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-3e-4}"
MODEL_SIZE="${MODEL_SIZE:-small}"
SEED="${SEED:-42}"

SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_PARITY="${SKIP_PARITY:-0}"

PYTHON="${PYTHON:-.venv/bin/python}"

mkdir -p "$LOG_DIR" "$PARITY_ROOT"

# alpha → tag helper (e.g. 0.33 -> a033)
alpha_tag() { printf "a%s" "$(echo "$1" | sed 's/[.]//g')"; }

PAAT_TAG="$(alpha_tag "$ALPHA")"
# Match the original script's λ=0 dir-naming convention when λ is zero.
if [[ "$(echo "$LAMBDA" | awk '{print ($1 == 0) ? 1 : 0}')" == "1" ]]; then
    PAAT_DIR="$TOK_ROOT/paat_${PAAT_TAG}_l0"
    PAAT_NAME="paat_${PAAT_TAG}_l0"
else
    PAAT_DIR="$TOK_ROOT/paat_${PAAT_TAG}"
    PAAT_NAME="paat_${PAAT_TAG}"
fi

# ── Banner ─────────────────────────────────────────────────────────────────
cat <<EOF

================================================================
  Stage 1 — PAAT-only run
  preset:        $PRESET
  vocab:         $INITIAL_VOCAB -> $TARGET_VOCAB in $ITERATIONS iters
  per-iter tok:  train=$TRAIN_TOK_PER_ITER  eval=$EVAL_TOK_PER_ITER
  alpha:         $ALPHA
  lambda:        $LAMBDA
  total docs:    $TOTAL_DOCS
  llm:           $MODEL_SIZE  bs=$BATCH_SIZE  lr=$LR  seq=$SEQ_LEN
  seed:          $SEED
  output:        $PAAT_DIR
================================================================
EOF

# ── PAAT (single run) ──────────────────────────────────────────────────────
train_paat() {
    local final="$PAAT_DIR/paat/tokenizer.json"
    if [[ -f "$final" ]]; then
        echo "[skip] PAAT already trained at $PAAT_DIR"
        return 0
    fi
    local log="$LOG_DIR/train_${PAAT_NAME}.log"
    echo ""
    echo "================================================================"
    echo "  Training PAAT α=$ALPHA λ=$LAMBDA  ->  $PAAT_DIR   (log: $log)"
    echo "================================================================"
    "$PYTHON" scripts/run_paat.py \
        --data-dir       "$DATA_DIR" \
        --flores-dir     "$FLORES_DIR" \
        --output-dir     "$PAAT_DIR" \
        --initial-vocab  "$INITIAL_VOCAB" \
        --target-vocab   "$TARGET_VOCAB" \
        --iterations     "$ITERATIONS" \
        --total-docs     "$TOTAL_DOCS" \
        --seq-len        "$SEQ_LEN" \
        --train-tokens-per-iter "$TRAIN_TOK_PER_ITER" \
        --eval-tokens-per-iter  "$EVAL_TOK_PER_ITER" \
        --model-size     "$MODEL_SIZE" \
        --batch-size     "$BATCH_SIZE" \
        --lr             "$LR" \
        --parity-alpha   "$ALPHA" \
        --balance-lambda "$LAMBDA" \
        --seed           "$SEED" \
        --skip-baseline \
        --skip-comparison \
        2>&1 | tee "$log"
}

if [[ "$SKIP_TRAIN" != "1" ]]; then
    train_paat
else
    echo ""
    echo "[SKIP_TRAIN=1] skipping PAAT training"
fi

# ── Parity evaluation ──────────────────────────────────────────────────────
if [[ "$SKIP_PARITY" == "1" ]]; then
    echo ""
    echo "[SKIP_PARITY=1] skipping parity evaluation"
    exit 0
fi

PAAT_TOK_DIR="$PAAT_DIR/paat"
report="$PARITY_ROOT/${PAAT_NAME}.json"
log="$LOG_DIR/parity_${PAAT_NAME}.log"

echo ""
echo "================================================================"
echo "  Parity evaluation"
echo "================================================================"

if [[ ! -f "$PAAT_TOK_DIR/tokenizer.json" ]]; then
    echo "[skip-parity] no tokenizer at $PAAT_TOK_DIR"
elif [[ -f "$report" ]]; then
    echo "[skip-parity] report already exists at $report"
else
    echo ""
    echo "  [parity] $PAAT_NAME  ->  $report"
    "$PYTHON" scripts/eval_parity.py \
        --tokenizer  "$PAAT_TOK_DIR" \
        --flores-dir "$FLORES_DIR" \
        --output     "$report" \
        2>&1 | tee "$log"
fi

echo ""
echo "Stage 1 complete."
echo "  Tokenizer:  $PAAT_DIR/"
echo "  Parity:     $report"
echo "  Logs:       $LOG_DIR/"
