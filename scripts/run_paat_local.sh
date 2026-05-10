#!/usr/bin/env bash
# Run PAAT locally and evaluate parity on the resulting tokenizer + the
# same-size Unigram baseline.
#
# Defaults are sized for a single A6000 / 24 GB consumer GPU and the
# 500 K-doc / 5 M-tok-per-iter budget that run_paat.py uses by default.
# Override any value via env var, e.g.:
#
#   PARITY_ALPHA=2.0 ITERATIONS=4 bash scripts/run_paat_local.sh
#   OUTPUT_DIR=models/tokenizers/paat_a05 PARITY_ALPHA=0.5 \
#       bash scripts/run_paat_local.sh
#
# Skip stages with SKIP_PAAT=1 (just re-run eval) or SKIP_EVAL=1 (just train).

set -euo pipefail

# ── Config (override via env) ──────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-data/raw/mc4}"
FLORES_DIR="${FLORES_DIR:-data/raw/flores}"
OUTPUT_DIR="${OUTPUT_DIR:-models/tokenizers/paat_local}"
RESULTS_DIR="${RESULTS_DIR:-results/parity}"
LOG_DIR="${LOG_DIR:-logs/paat}"

INITIAL_VOCAB="${INITIAL_VOCAB:-32000}"
TARGET_VOCAB="${TARGET_VOCAB:-16000}"
ITERATIONS="${ITERATIONS:-3}"
TOTAL_DOCS="${TOTAL_DOCS:-500000}"
TRAIN_TOK_PER_ITER="${TRAIN_TOK_PER_ITER:-5000000}"
EVAL_TOK_PER_ITER="${EVAL_TOK_PER_ITER:-2000000}"
SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-3e-4}"
MODEL_SIZE="${MODEL_SIZE:-small}"
PARITY_ALPHA="${PARITY_ALPHA:-1.0}"
SEED="${SEED:-42}"

SKIP_PAAT="${SKIP_PAAT:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

PYTHON="${PYTHON:-.venv/bin/python}"

mkdir -p "$LOG_DIR"
mkdir -p "$RESULTS_DIR"

cat <<EOF

================================================================
  PAAT local pipeline
  output:        $OUTPUT_DIR
  data:          $DATA_DIR  (docs=$TOTAL_DOCS)
  flores:        $FLORES_DIR
  vocab:         $INITIAL_VOCAB -> $TARGET_VOCAB in $ITERATIONS iters
  per-iter tok:  train=$TRAIN_TOK_PER_ITER  eval=$EVAL_TOK_PER_ITER
  parity alpha:  $PARITY_ALPHA
  llm:           $MODEL_SIZE  bs=$BATCH_SIZE  lr=$LR  seq=$SEQ_LEN
  seed:          $SEED
================================================================
EOF

# ── 1. Run PAAT ────────────────────────────────────────────────────────────
PAAT_TOK="$OUTPUT_DIR/paat/tokenizer.json"
BASE_TOK="$OUTPUT_DIR/baseline/tokenizer.json"

if [[ "$SKIP_PAAT" == "1" ]]; then
    echo ""
    echo "[SKIP_PAAT=1] skipping PAAT training stage"
elif [[ -f "$PAAT_TOK" && -f "$BASE_TOK" ]]; then
    echo ""
    echo "[skip] PAAT + baseline tokenizers already exist:"
    echo "        $PAAT_TOK"
    echo "        $BASE_TOK"
    echo "       (delete them or pass SKIP_PAAT=1 SKIP_EVAL=0 to rerun eval only)"
else
    log="$LOG_DIR/run_paat.log"
    echo ""
    echo "[1/2] Running PAAT  →  log: $log"
    "$PYTHON" scripts/run_paat.py \
        --data-dir       "$DATA_DIR" \
        --flores-dir     "$FLORES_DIR" \
        --output-dir     "$OUTPUT_DIR" \
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
        --parity-alpha   "$PARITY_ALPHA" \
        --seed           "$SEED" \
        --skip-comparison \
        2>&1 | tee "$log"
fi

# ── 2. Parity evaluation ───────────────────────────────────────────────────
if [[ "$SKIP_EVAL" == "1" ]]; then
    echo ""
    echo "[SKIP_EVAL=1] skipping parity evaluation"
    exit 0
fi

eval_one() {
    local tag="$1"            # paat | baseline
    local tok_path="$2"       # path to tokenizer.json
    local report="$RESULTS_DIR/${OUTPUT_DIR##*/}_${tag}.json"
    local log="$LOG_DIR/eval_parity_${tag}.log"

    if [[ ! -f "$tok_path" ]]; then
        echo "[skip-eval] $tag tokenizer not found at $tok_path"
        return 0
    fi

    echo ""
    echo "[parity:$tag] tokenizer: $tok_path"
    echo "[parity:$tag] report:    $report"
    "$PYTHON" scripts/eval_parity.py \
        --tokenizer "$(dirname "$tok_path")" \
        --flores-dir "$FLORES_DIR" \
        --output     "$report" \
        2>&1 | tee "$log"
}

echo ""
echo "[2/2] Parity evaluation"
eval_one "paat"     "$PAAT_TOK"
eval_one "baseline" "$BASE_TOK"

# ── Side-by-side summary ───────────────────────────────────────────────────
PAAT_REPORT="$RESULTS_DIR/${OUTPUT_DIR##*/}_paat.json"
BASE_REPORT="$RESULTS_DIR/${OUTPUT_DIR##*/}_baseline.json"

if [[ -f "$PAAT_REPORT" && -f "$BASE_REPORT" ]]; then
    echo ""
    echo "================================================================"
    echo "  Parity comparison  (lower Gini / lower max-tps = fairer)"
    echo "================================================================"
    "$PYTHON" - <<PY
import json, sys
from pathlib import Path

def summary(path):
    d = json.loads(Path(path).read_text())["summary"]
    return d

a = summary("$PAAT_REPORT")
b = summary("$BASE_REPORT")
fmt = lambda x: f"{x:>9.4f}"
rows = [
    ("gini_tokens_per_sentence", "Gini(tok/sent)  ↓"),
    ("gini_tokens_per_byte",     "Gini(tok/byte)  ↓"),
    ("mean_tokens_per_sentence", "mean tok/sent   ↓"),
    ("max_tokens_per_sentence",  "max  tok/sent   ↓"),
    ("tokens_per_sentence_ratio","max/min ratio   ↓"),
]
print(f"  {'metric':<20}  {'PAAT':>9}  {'baseline':>9}  {'Δ (paat-base)':>14}")
print("  " + "-" * 60)
for k, label in rows:
    av, bv = a[k], b[k]
    print(f"  {label:<20}  {fmt(av)}  {fmt(bv)}  {fmt(av-bv)}")
PY
fi

echo ""
echo "Done.  Reports:"
echo "  $PAAT_REPORT"
echo "  $BASE_REPORT"
