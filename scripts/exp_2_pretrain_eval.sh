#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Stage 2 — Pre-train one LM per tokenizer from Stage 1, then run
# downstream evaluation on every model.
#
#   Tokenizers → LMs:
#     bpe                    → models/lm/bpe/
#     adat                   → models/lm/adat/
#     unigram                → models/lm/unigram/
#     paat_a033              → models/lm/paat_a033/
#     paat_a067              → models/lm/paat_a067/
#     paat_a100              → models/lm/paat_a100/
#
#   Downstream eval per model:
#     results/downstream/<name>/{paper_tasks,xnli}/...
#     results/downstream/comparison.json
#
# Stage 2 expects Stage 1 to have produced the tokenizers under
# ``$TOK_ROOT``.  Run scripts/exp_1_tokenizers.sh first.
#
# ── Usage ─────────────────────────────────────────────────────────────────
#   bash scripts/exp_2_pretrain_eval.sh
#   PRESET=smoke bash scripts/exp_2_pretrain_eval.sh
#   PRESET=long  bash scripts/exp_2_pretrain_eval.sh
#   ALPHAS="0.5 1.0 2.0" bash scripts/exp_2_pretrain_eval.sh   # match Stage 1
#   MODELS="adat paat_a100" bash scripts/exp_2_pretrain_eval.sh
#   SKIP_PRETRAIN=1 bash scripts/exp_2_pretrain_eval.sh        # eval only
#   SKIP_EVAL=1     bash scripts/exp_2_pretrain_eval.sh        # pretrain only
#
# Approximate wall-clock per model on a single A6000 (bf16, batch 64):
#   smoke    ~5  min     pilot   ~3  h     standard ~10 h     long ~30 h
# Multiply by the number of MODELS for total time.
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Presets (match scripts/run_full_pipeline.sh) ───────────────────────────
PRESET="${PRESET:-standard}"
case "$PRESET" in
    smoke)
        : "${TRAIN_TOKENS:=30000000}";    : "${TOTAL_DOCS:=150000}"
        ;;
    pilot)
        : "${TRAIN_TOKENS:=300000000}";   : "${TOTAL_DOCS:=1000000}"
        ;;
    standard)
        : "${TRAIN_TOKENS:=1000000000}";  : "${TOTAL_DOCS:=3000000}"
        ;;
    long)
        : "${TRAIN_TOKENS:=3000000000}";  : "${TOTAL_DOCS:=8000000}"
        ;;
    *)
        echo "Unknown PRESET=$PRESET. Choose smoke|pilot|standard|long, "
        echo "or set TRAIN_TOKENS / TOTAL_DOCS directly." >&2
        exit 1
        ;;
esac

# ── Config (override via env) ──────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-data/raw/mc4}"
TOK_ROOT="${TOK_ROOT:-models/tokenizers}"
LM_ROOT="${LM_ROOT:-models/lm}"
EVAL_ROOT="${EVAL_ROOT:-results/downstream}"
LOG_DIR="${LOG_DIR:-logs/exp_2}"

MODEL_SIZE="${MODEL_SIZE:-pythia70m}"
BATCH_SIZE="${BATCH_SIZE:-64}"
SEQ_LEN="${SEQ_LEN:-512}"
LEARNING_RATE="${LEARNING_RATE:-1e-3}"
SEED="${SEED:-42}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-32}"

ALPHAS="${ALPHAS:-0.33 0.67 1.0}"

# Map logical name → tokenizer.json directory.  Override individually if
# Stage 1 was run with a custom layout.
BPE_TOK="${BPE_TOK:-$TOK_ROOT/bpe}"
PARITY_BPE_TOK="${PARITY_BPE_TOK:-$TOK_ROOT/parity_bpe}"
ADAT_TOK="${ADAT_TOK:-$TOK_ROOT/adat/adat}"
UNIGRAM_TOK="${UNIGRAM_TOK:-$TOK_ROOT/adat/baseline}"
PAAT_L0_TOK="${PAAT_L0_TOK:-$TOK_ROOT/paat_a100_l0/paat}"

alpha_tag() { printf "a%s" "$(echo "$1" | sed 's/[.]//g')"; }

# Build the default MODELS list from the presets unless user overrode it.
default_models="bpe parity_bpe adat unigram"
for a in $ALPHAS; do
    default_models="$default_models paat_$(alpha_tag "$a")"
done
default_models="$default_models paat_a100_l0"
MODELS="${MODELS:-$default_models}"

SKIP_PRETRAIN="${SKIP_PRETRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

mkdir -p "$LOG_DIR"

cat <<EOF

================================================================
  Stage 2 — Pre-train LMs + downstream evaluation
  preset:        $PRESET
  models:        $MODELS
  train tokens:  $(printf "%'d" $TRAIN_TOKENS)
  total docs:    $(printf "%'d" $TOTAL_DOCS)
  arch:          $MODEL_SIZE  bs=$BATCH_SIZE  seq=$SEQ_LEN  lr=$LEARNING_RATE
  seed:          $SEED
================================================================
EOF

# Resolve a logical model name to its tokenizer directory.
get_tokenizer_path() {
    case "$1" in
        bpe)           echo "$BPE_TOK" ;;
        parity_bpe)    echo "$PARITY_BPE_TOK" ;;
        adat)          echo "$ADAT_TOK" ;;
        unigram)       echo "$UNIGRAM_TOK" ;;
        paat_a100_l0)  echo "$PAAT_L0_TOK" ;;
        paat_*)        echo "$TOK_ROOT/${1}/paat" ;;
        *) echo "Unknown model name: $1" >&2; return 1 ;;
    esac
}

PYTHON="${PYTHON:-.venv/bin/python}"

# ── Pretraining ────────────────────────────────────────────────────────────
pretrain_one() {
    local name="$1"
    local tok_path; tok_path="$(get_tokenizer_path "$name")"
    local out_dir="$LM_ROOT/$name"
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

    "$PYTHON" scripts/pretrain_lm.py \
        --tokenizer    "$tok_path" \
        --output-dir   "$out_dir" \
        --data-dir     "$DATA_DIR" \
        --train-tokens "$TRAIN_TOKENS" \
        --total-docs   "$TOTAL_DOCS" \
        --seq-len      "$SEQ_LEN" \
        --batch-size   "$BATCH_SIZE" \
        --learning-rate "$LEARNING_RATE" \
        --model-size   "$MODEL_SIZE" \
        --seed         "$SEED" \
        2>&1 | tee "$log"
}

if [[ "$SKIP_PRETRAIN" != "1" ]]; then
    for m in $MODELS; do
        pretrain_one "$m" || echo "[WARN] pretrain failed for $m, continuing"
    done
else
    echo "[SKIP_PRETRAIN=1] skipping pretraining stage"
fi

# ── Downstream evaluation ──────────────────────────────────────────────────
if [[ "$SKIP_EVAL" == "1" ]]; then
    echo ""
    echo "[SKIP_EVAL=1] skipping evaluation stage"
    exit 0
fi

eval_dirs=()
for m in $MODELS; do
    d="$LM_ROOT/$m"
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
echo "  output:  $EVAL_ROOT"
echo "  log:     $log"
echo "================================================================"

"$PYTHON" scripts/eval_downstream.py \
    --model-dir "${eval_dirs[@]}" \
    --output-dir "$EVAL_ROOT" \
    --batch-size "$EVAL_BATCH_SIZE" \
    2>&1 | tee "$log"

echo ""
echo "Stage 2 complete."
echo "  LMs:         $LM_ROOT/{$(echo "$MODELS" | tr ' ' ',')}"
echo "  Downstream:  $EVAL_ROOT/comparison.json"
echo "  Logs:        $LOG_DIR/"
