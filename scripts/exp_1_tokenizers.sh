#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Stage 1 — Train all six tokenizers and run parity evaluation on each.
#
#   1. BPE                                   → models/tokenizers/bpe/
#   2. ADAT                                  → models/tokenizers/adat/
#   3. Unigram (same-size baseline)          → models/tokenizers/adat/baseline/
#      (produced as a side-effect of the ADAT run; reused by PAAT)
#   4. PAAT α = 0.33                         → models/tokenizers/paat_a033/
#   5. PAAT α = 0.67                         → models/tokenizers/paat_a067/
#   6. PAAT α = 1.00                         → models/tokenizers/paat_a100/
#
# Each tokenizer is then parity-evaluated against FLORES+:
#   results/parity/{bpe,adat,unigram,paat_a033,paat_a067,paat_a100}.json
#
# Plus a side-by-side summary table at:
#   results/parity/comparison.json
#
# ── Usage ─────────────────────────────────────────────────────────────────
#   bash scripts/exp_1_tokenizers.sh                  # default config
#   PRESET=smoke bash scripts/exp_1_tokenizers.sh     # tiny vocab, fast
#   ALPHAS="0.5 1.0 2.0" bash scripts/exp_1_tokenizers.sh
#   SKIP_TRAIN=1 bash scripts/exp_1_tokenizers.sh     # parity eval only
#
# Each stage is idempotent — re-running skips tokenizers that already exist.
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
        : "${BPE_TOTAL_DOCS:=50000}"
        ;;
    standard)
        : "${INITIAL_VOCAB:=64000}";  : "${TARGET_VOCAB:=32000}"
        : "${ITERATIONS:=3}";         : "${TOTAL_DOCS:=500000}"
        : "${TRAIN_TOK_PER_ITER:=5000000}"
        : "${EVAL_TOK_PER_ITER:=2000000}"
        : "${BPE_TOTAL_DOCS:=500000}"
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

ALPHAS="${ALPHAS:-0.33 0.67 1.0}"

SEQ_LEN="${SEQ_LEN:-512}"
BATCH_SIZE="${BATCH_SIZE:-32}"
LR="${LR:-3e-4}"
MODEL_SIZE="${MODEL_SIZE:-small}"
SEED="${SEED:-42}"

SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_PARITY="${SKIP_PARITY:-0}"

PYTHON="${PYTHON:-.venv/bin/python}"

mkdir -p "$LOG_DIR" "$PARITY_ROOT"

# ── Banner ─────────────────────────────────────────────────────────────────
cat <<EOF

================================================================
  Stage 1 — Tokenizer training + parity evaluation
  preset:        $PRESET
  vocab:         $INITIAL_VOCAB -> $TARGET_VOCAB in $ITERATIONS iters
  per-iter tok:  train=$TRAIN_TOK_PER_ITER  eval=$EVAL_TOK_PER_ITER
  alphas:        $ALPHAS
  total docs:    $TOTAL_DOCS  (BPE: $BPE_TOTAL_DOCS)
  llm:           $MODEL_SIZE  bs=$BATCH_SIZE  lr=$LR  seq=$SEQ_LEN
  seed:          $SEED
================================================================
EOF

# Tokenizer paths produced by this stage.
BPE_DIR="$TOK_ROOT/bpe"
ADAT_ROOT="$TOK_ROOT/adat"
ADAT_TOK="$ADAT_ROOT/adat"
UNIGRAM_TOK="$ADAT_ROOT/baseline"   # same-size Unigram from the ADAT run

# alpha → tag helper (e.g. 0.33 -> a033)
alpha_tag() { printf "a%s" "$(echo "$1" | sed 's/[.]//g')"; }

# ── 1. BPE ─────────────────────────────────────────────────────────────────
train_bpe() {
    local out="$BPE_DIR"
    if [[ -f "$out/tokenizer.json" ]]; then
        echo "[skip] BPE already trained at $out"
        return 0
    fi
    local log="$LOG_DIR/train_bpe.log"
    echo ""
    echo "================================================================"
    echo "  [1/6] Training BPE  ->  $out   (log: $log)"
    echo "================================================================"
    "$PYTHON" scripts/train_tokenizer.py \
        --data-dir   "$DATA_DIR" \
        --output-dir "$out" \
        --vocab-size "$TARGET_VOCAB" \
        --total-docs "$BPE_TOTAL_DOCS" \
        2>&1 | tee "$log"
}

# ── 2. ADAT (also produces the same-size Unigram baseline) ─────────────────
train_adat() {
    if [[ -f "$ADAT_TOK/tokenizer.json" && -f "$UNIGRAM_TOK/tokenizer.json" ]]; then
        echo "[skip] ADAT + Unigram baseline already trained at $ADAT_ROOT"
        return 0
    fi
    local log="$LOG_DIR/train_adat.log"
    echo ""
    echo "================================================================"
    echo "  [2/6] Training ADAT  ->  $ADAT_ROOT   (log: $log)"
    echo "================================================================"
    "$PYTHON" scripts/run_adat.py \
        --data-dir       "$DATA_DIR" \
        --output-dir     "$ADAT_ROOT" \
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
        --seed           "$SEED" \
        --skip-comparison \
        2>&1 | tee "$log"
}

# ── 3. PAAT (one run per alpha; reuse Unigram baseline from ADAT) ──────────
train_paat() {
    local alpha="$1"
    local tag; tag="$(alpha_tag "$alpha")"
    local out="$TOK_ROOT/paat_${tag}"
    local final="$out/paat/tokenizer.json"

    if [[ -f "$final" ]]; then
        echo "[skip] PAAT α=$alpha already trained at $out"
        return 0
    fi
    local log="$LOG_DIR/train_paat_${tag}.log"
    echo ""
    echo "================================================================"
    echo "  Training PAAT α=$alpha  ->  $out   (log: $log)"
    echo "================================================================"
    "$PYTHON" scripts/run_paat.py \
        --data-dir       "$DATA_DIR" \
        --flores-dir     "$FLORES_DIR" \
        --output-dir     "$out" \
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
        --parity-alpha   "$alpha" \
        --seed           "$SEED" \
        --skip-baseline \
        --skip-comparison \
        2>&1 | tee "$log"
}

# ── Training stage ─────────────────────────────────────────────────────────
if [[ "$SKIP_TRAIN" != "1" ]]; then
    train_bpe
    train_adat                     # also produces $UNIGRAM_TOK
    i=3
    for a in $ALPHAS; do
        echo ""
        echo "  [$i/6]"
        train_paat "$a"
        i=$((i + 1))
    done
else
    echo ""
    echo "[SKIP_TRAIN=1] skipping tokenizer training"
fi

# ── Parity evaluation ──────────────────────────────────────────────────────
if [[ "$SKIP_PARITY" == "1" ]]; then
    echo ""
    echo "[SKIP_PARITY=1] skipping parity evaluation"
    exit 0
fi

eval_parity_one() {
    local name="$1"           # bpe | adat | unigram | paat_a033 | ...
    local tok_dir="$2"        # directory containing tokenizer.json
    local report="$PARITY_ROOT/${name}.json"
    local log="$LOG_DIR/parity_${name}.log"

    if [[ ! -f "$tok_dir/tokenizer.json" ]]; then
        echo "[skip-parity] $name: no tokenizer at $tok_dir"
        return 0
    fi
    if [[ -f "$report" ]]; then
        echo "[skip-parity] $name: report already exists at $report"
        return 0
    fi
    echo ""
    echo "  [parity] $name  ->  $report"
    "$PYTHON" scripts/eval_parity.py \
        --tokenizer  "$tok_dir" \
        --flores-dir "$FLORES_DIR" \
        --output     "$report" \
        2>&1 | tee "$log"
}

echo ""
echo "================================================================"
echo "  Parity evaluation"
echo "================================================================"

eval_parity_one "bpe"     "$BPE_DIR"
eval_parity_one "adat"    "$ADAT_TOK"
eval_parity_one "unigram" "$UNIGRAM_TOK"
for a in $ALPHAS; do
    tag="$(alpha_tag "$a")"
    eval_parity_one "paat_${tag}" "$TOK_ROOT/paat_${tag}/paat"
done

# ── Side-by-side summary ───────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "  Parity comparison  (lower = fairer)"
echo "================================================================"
"$PYTHON" - <<PY
import json
from pathlib import Path

root = Path("$PARITY_ROOT")
alphas_str = "$ALPHAS".strip().split()
def tag(a): return "a" + a.replace(".", "")

names = ["bpe", "adat", "unigram"] + [f"paat_{tag(a)}" for a in alphas_str]
rows = []
for n in names:
    p = root / f"{n}.json"
    if not p.exists():
        continue
    s = json.loads(p.read_text())["summary"]
    rows.append((n, s))

if not rows:
    print("  no parity reports found")
else:
    metrics = [
        ("gini_tokens_per_sentence", "Gini(tok/sent)"),
        ("gini_tokens_per_byte",     "Gini(tok/byte)"),
        ("mean_tokens_per_sentence", "mean tok/sent"),
        ("max_tokens_per_sentence",  "max  tok/sent"),
        ("tokens_per_sentence_ratio","max/min ratio"),
    ]
    name_w = max(len(n) for n, _ in rows) + 2
    print(f"  {'tokenizer':<{name_w}}  " + "  ".join(f"{label:>15}" for _, label in metrics))
    print("  " + "-" * (name_w + 17 * len(metrics)))
    for n, s in rows:
        vals = "  ".join(f"{s[k]:>15.4f}" for k, _ in metrics)
        print(f"  {n:<{name_w}}  {vals}")

    out = root / "comparison.json"
    out.write_text(json.dumps({n: s for n, s in rows}, indent=2))
    print(f"\n  Wrote summary to {out}")
PY

echo ""
echo "Stage 1 complete."
echo "  Tokenizers:  $TOK_ROOT/{bpe,adat,paat_*}/"
echo "  Parity:      $PARITY_ROOT/"
echo "  Logs:        $LOG_DIR/"
