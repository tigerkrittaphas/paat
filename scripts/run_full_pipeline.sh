#!/usr/bin/env bash
# Pre-train pythia70m on each of the three tokenizers (ADAT, Unigram, BPE),
# then run downstream evaluation on all of them.
#
# ── Hyperparameter rationale ──────────────────────────────────────────────
# Defaults below are the recommended "standard" comparison budget. Tuned
# from the 300M-token pilot, where most downstream metrics sat at random
# chance (see docs/downstream-results.md).
#
#   TRAIN_TOKENS=1_000_000_000  ~3.3× the pilot, enough to lift PIQA / SciQ
#                               above chance for a meaningful three-way
#                               comparison. (Paper uses 15B; compute-bound
#                               here.)
#   TOTAL_DOCS=3_000_000        At ~350 tok/doc multilingual avg, 3M docs
#                               yields ~1B tokens with headroom — the
#                               tokenizer never starves before the budget.
#   BATCH_SIZE=64               pythia70m (~36M params) at seq 512 in bf16
#                               uses ~12 GB on an A6000; batch 64 keeps
#                               peak under 24 GB and ~doubles throughput
#                               vs the pilot's batch 32.
#   LEARNING_RATE=1e-3          Matches Pythia-70M Table 13 (Zheng 2024).
#   EVAL_BATCH_SIZE=32          Inference is cheaper than training.
#
# ── Usage ──────────────────────────────────────────────────────────────────
#   bash scripts/run_full_pipeline.sh                       # standard 1B-token run
#   PRESET=smoke bash scripts/run_full_pipeline.sh          # 30M tokens, ~5 min/model
#   PRESET=pilot bash scripts/run_full_pipeline.sh          # 300M tokens, original budget
#   PRESET=long bash scripts/run_full_pipeline.sh           # 3B tokens, paper-leaning
#   SKIP_PRETRAIN=1 bash scripts/run_full_pipeline.sh       # eval only
#   SKIP_EVAL=1     bash scripts/run_full_pipeline.sh       # pretrain only
#   MODELS="adat unigram" bash scripts/run_full_pipeline.sh # subset
#
# Approximate wall-clock per model on a single A6000 (bf16, batch 64):
#   smoke    ~5  min     pilot   ~3  h     standard ~10 h     long ~30 h
# Multiply by the number of MODELS (default 3) for total time.

set -euo pipefail

# ── Presets ────────────────────────────────────────────────────────────────
PRESET="${PRESET:-standard}"
case "$PRESET" in
    smoke)
        : "${TRAIN_TOKENS:=30000000}"
        : "${TOTAL_DOCS:=150000}"
        ;;
    pilot)
        : "${TRAIN_TOKENS:=300000000}"
        : "${TOTAL_DOCS:=1000000}"
        ;;
    standard)
        : "${TRAIN_TOKENS:=1000000000}"
        : "${TOTAL_DOCS:=3000000}"
        ;;
    long)
        : "${TRAIN_TOKENS:=3000000000}"
        : "${TOTAL_DOCS:=8000000}"
        ;;
    *)
        echo "Unknown PRESET=$PRESET. Choose smoke|pilot|standard|long, or set TRAIN_TOKENS / TOTAL_DOCS directly." >&2
        exit 1
        ;;
esac

# ── Config (override via env) ──────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-data/raw/mc4}"
MODEL_SIZE="${MODEL_SIZE:-pythia70m}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEQ_LEN="${SEQ_LEN:-512}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
SEED="${SEED:-42}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"

LM_OUTPUT_ROOT="${LM_OUTPUT_ROOT:-models/lm}"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-results/downstream}"
LOG_DIR="${LOG_DIR:-logs/pipeline}"

# Tokenizer paths (override individually if your layout differs)
ADAT_TOK="${ADAT_TOK:-models/tokenizers/adat_full_pythia/adat}"
UNIGRAM_TOK="${UNIGRAM_TOK:-models/tokenizers/adat_full/baseline}"
BPE_TOK="${BPE_TOK:-models/tokenizers/bpe_full}"

MODELS="${MODELS:-adat unigram bpe}"
SKIP_PRETRAIN="${SKIP_PRETRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

mkdir -p "$LOG_DIR"

# ── Banner ─────────────────────────────────────────────────────────────────
cat <<EOF

================================================================
  PAAT downstream pipeline
  preset:        $PRESET
  models:        $MODELS
  train tokens:  $(printf "%'d" $TRAIN_TOKENS)
  total docs:    $(printf "%'d" $TOTAL_DOCS)
  model size:    $MODEL_SIZE
  batch / seq:   $BATCH_SIZE / $SEQ_LEN
  learning rate: $LEARNING_RATE
  seed:          $SEED
================================================================
EOF

# ── Helpers ────────────────────────────────────────────────────────────────
get_tokenizer_path() {
    case "$1" in
        adat)    echo "$ADAT_TOK" ;;
        unigram) echo "$UNIGRAM_TOK" ;;
        bpe)     echo "$BPE_TOK" ;;
        *) echo "Unknown model: $1" >&2; exit 1 ;;
    esac
}

pretrain_one() {
    local name="$1"
    local tok_path
    tok_path="$(get_tokenizer_path "$name")"
    local out_dir="$LM_OUTPUT_ROOT/$name"
    local log="$LOG_DIR/pretrain_${name}.log"

    if [[ ! -f "$tok_path/tokenizer.json" ]]; then
        echo "[SKIP] $name: tokenizer.json not found at $tok_path"
        return 1
    fi
    if [[ -f "$out_dir/pretrain_summary.json" ]]; then
        echo "[SKIP] $name: already trained ($out_dir/pretrain_summary.json exists)"
        return 0
    fi

    echo ""
    echo "================================================================"
    echo "  Pretraining $name"
    echo "  tokenizer:  $tok_path"
    echo "  output:     $out_dir"
    echo "  log:        $log"
    echo "================================================================"

    python scripts/pretrain_lm.py \
        --tokenizer "$tok_path" \
        --output-dir "$out_dir" \
        --data-dir "$DATA_DIR" \
        --train-tokens "$TRAIN_TOKENS" \
        --total-docs "$TOTAL_DOCS" \
        --seq-len "$SEQ_LEN" \
        --batch-size "$BATCH_SIZE" \
        --learning-rate "$LEARNING_RATE" \
        --model-size "$MODEL_SIZE" \
        --seed "$SEED" \
        2>&1 | tee "$log"
}

# ── Pretraining ────────────────────────────────────────────────────────────
if [[ "$SKIP_PRETRAIN" != "1" ]]; then
    for m in $MODELS; do
        pretrain_one "$m" || echo "[WARN] pretrain failed for $m, continuing"
    done
else
    echo "[SKIP_PRETRAIN=1] skipping pretraining stage"
fi

# ── Downstream eval ────────────────────────────────────────────────────────
if [[ "$SKIP_EVAL" != "1" ]]; then
    eval_dirs=()
    for m in $MODELS; do
        d="$LM_OUTPUT_ROOT/$m"
        if [[ -f "$d/pretrain_summary.json" ]]; then
            eval_dirs+=("$d")
        else
            echo "[SKIP] eval: $d has no pretrain_summary.json"
        fi
    done

    if [[ ${#eval_dirs[@]} -eq 0 ]]; then
        echo "[ERROR] no trained models found, nothing to evaluate"
        exit 1
    fi

    log="$LOG_DIR/eval_downstream.log"
    echo ""
    echo "================================================================"
    echo "  Downstream evaluation"
    echo "  models:  ${eval_dirs[*]}"
    echo "  output:  $EVAL_OUTPUT_ROOT"
    echo "  log:     $log"
    echo "================================================================"

    python scripts/eval_downstream.py \
        --model-dir "${eval_dirs[@]}" \
        --output-dir "$EVAL_OUTPUT_ROOT" \
        --batch-size "$EVAL_BATCH_SIZE" \
        2>&1 | tee "$log"
else
    echo "[SKIP_EVAL=1] skipping evaluation stage"
fi

echo ""
echo "Pipeline complete."
echo "  Models:      $LM_OUTPUT_ROOT/{$(echo $MODELS | tr ' ' ',')}"
echo "  Comparison:  $EVAL_OUTPUT_ROOT/comparison.json"
echo "  Logs:        $LOG_DIR/"
