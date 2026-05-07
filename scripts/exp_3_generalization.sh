#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────
# Stage 3 — Off-FLORES held-out parity evaluation.
#
# Re-evaluates the eight frozen tokenizers from exp_1 on TWO non-FLORES
# held-out parallel corpora (NTREX-128 news, multilingual Bible) and
# produces a cross-domain parity table + Spearman rank-correlation figure
# that hardens the "PAAT generalises rather than fits FLORES style" claim.
#
# No retraining — tokenizers are frozen artifacts on disk.  Stage 1 must
# already have produced models/tokenizers/{bpe,parity_bpe,adat,paat_*}/.
#
#   1. Download NTREX-128 + Bible (~5 min, mostly bandwidth).
#   2. Per-tokenizer parity eval on each corpus (idempotent skip).
#   3. Per-corpus comparison.json (mirrors results/parity/comparison.json).
#   4. Cross-domain comparison: gini recomputed on the strict 3-way lang
#      intersection + Spearman rank correlation between domain pairs.
#
# Outputs:
#   data/raw/ntrex/<lang>.jsonl               # 1997 sentences × N langs
#   data/raw/bible/<lang>.jsonl               # ~1000 verses × M langs
#   results/parity_ntrex/<name>.json          # per-tokenizer NTREX report
#   results/parity_ntrex/comparison.json
#   results/parity_bible/<name>.json
#   results/parity_bible/comparison.json
#   results/parity_cross_domain.json          # headline cross-domain table
#
# ── Usage ─────────────────────────────────────────────────────────────────
#   bash scripts/exp_3_generalization.sh                  # standard preset
#   PRESET=smoke bash scripts/exp_3_generalization.sh     # bpe + paat_a100_l0
#   SKIP_DOWNLOAD=1 bash scripts/exp_3_generalization.sh  # already downloaded
#   SKIP_NTREX=1   bash scripts/exp_3_generalization.sh   # bible only
#   SKIP_BIBLE=1   bash scripts/exp_3_generalization.sh   # ntrex only
#
# Each stage is idempotent — re-running skips per-language downloads and
# per-tokenizer eval reports that already exist.
# ──────────────────────────────────────────────────────────────────────────

set -euo pipefail

# ── Presets — gate eval breadth, not downloads (downloads are small) ──────
PRESET="${PRESET:-standard}"
case "$PRESET" in
    smoke)
        # Headline contrast only: BPE vs the parity-only PAAT ablation.
        : "${EVAL_TOKENIZERS:=bpe paat_a100_l0}"
        ;;
    standard)
        : "${EVAL_TOKENIZERS:=bpe parity_bpe adat unigram paat_a033 paat_a067 paat_a100 paat_a100_l0}"
        ;;
    *)
        echo "Unknown PRESET=$PRESET. Choose smoke|standard." >&2
        exit 1
        ;;
esac

# ── Config (override via env) ──────────────────────────────────────────────
DATA_DIR="${DATA_DIR:-data/raw}"
NTREX_DIR="$DATA_DIR/ntrex"
BIBLE_DIR="$DATA_DIR/bible"
FLORES_DIR="${FLORES_DIR:-data/raw/flores}"

TOK_ROOT="${TOK_ROOT:-models/tokenizers}"
PARITY_FLORES_ROOT="${PARITY_FLORES_ROOT:-results/parity}"
PARITY_NTREX_ROOT="${PARITY_NTREX_ROOT:-results/parity_ntrex}"
PARITY_BIBLE_ROOT="${PARITY_BIBLE_ROOT:-results/parity_bible}"
LOG_DIR="${LOG_DIR:-logs/exp_3}"

NTREX_REPO="${NTREX_REPO:-vendor/NTREX-128}"

SKIP_DOWNLOAD="${SKIP_DOWNLOAD:-0}"
SKIP_NTREX="${SKIP_NTREX:-0}"
SKIP_BIBLE="${SKIP_BIBLE:-0}"
SKIP_PARITY="${SKIP_PARITY:-0}"
SKIP_COMPARE="${SKIP_COMPARE:-0}"

PYTHON="${PYTHON:-.venv/bin/python}"

mkdir -p "$LOG_DIR" "$PARITY_NTREX_ROOT" "$PARITY_BIBLE_ROOT"

# ── Tokenizer path map (must match exp_1's actual outputs) ────────────────
declare -A TOKENIZERS=(
    [bpe]="$TOK_ROOT/bpe"
    [parity_bpe]="$TOK_ROOT/parity_bpe"
    [adat]="$TOK_ROOT/adat/adat"
    [unigram]="$TOK_ROOT/adat/baseline"
    [paat_a033]="$TOK_ROOT/paat_a033/paat"
    [paat_a067]="$TOK_ROOT/paat_a067/paat"
    [paat_a100]="$TOK_ROOT/paat_a100/paat"
    [paat_a100_l0]="$TOK_ROOT/paat_a100_l0/paat"
)

# ── Banner ─────────────────────────────────────────────────────────────────
cat <<EOF

================================================================
  Stage 3 — Off-FLORES held-out parity evaluation
  preset:        $PRESET
  tokenizers:    $EVAL_TOKENIZERS
  ntrex dir:     $NTREX_DIR
  bible dir:     $BIBLE_DIR
  flores dir:    $FLORES_DIR
  parity out:    $PARITY_NTREX_ROOT, $PARITY_BIBLE_ROOT
  cross-domain:  results/parity_cross_domain.json
================================================================
EOF

# ── Pre-flight: every requested tokenizer must exist on disk ─────────────
missing=()
for tok in $EVAL_TOKENIZERS; do
    dir="${TOKENIZERS[$tok]:-}"
    if [[ -z "$dir" ]]; then
        echo "ERROR: unknown tokenizer name '$tok' (not in TOKENIZERS map)" >&2
        exit 1
    fi
    if [[ ! -f "$dir/tokenizer.json" ]]; then
        missing+=("$tok ($dir/tokenizer.json)")
    fi
done
if (( ${#missing[@]} > 0 )); then
    echo "" >&2
    echo "ERROR: missing tokenizers — run exp_1 first:" >&2
    for m in "${missing[@]}"; do echo "  - $m" >&2; done
    echo "" >&2
    echo "  bash scripts/exp_1_tokenizers.sh" >&2
    exit 1
fi

# ── 1. Download NTREX + Bible ─────────────────────────────────────────────
download_ntrex() {
    local log="$LOG_DIR/download_ntrex.log"
    echo ""
    echo "================================================================"
    echo "  [1a] Downloading NTREX-128  ->  $NTREX_DIR  (log: $log)"
    echo "================================================================"
    "$PYTHON" scripts/download_ntrex.py \
        --output-dir "$NTREX_DIR" \
        --ntrex-repo "$NTREX_REPO" \
        2>&1 | tee "$log"
}

download_bible() {
    local log="$LOG_DIR/download_bible.log"
    echo ""
    echo "================================================================"
    echo "  [1b] Downloading Bible      ->  $BIBLE_DIR  (log: $log)"
    echo "================================================================"
    "$PYTHON" scripts/download_bible.py \
        --output-dir "$BIBLE_DIR" \
        2>&1 | tee "$log"
}

if [[ "$SKIP_DOWNLOAD" != "1" ]]; then
    [[ "$SKIP_NTREX" == "1" ]] || download_ntrex
    [[ "$SKIP_BIBLE" == "1" ]] || download_bible
else
    echo ""
    echo "[SKIP_DOWNLOAD=1] skipping downloads"
fi

# ── 2. Per-tokenizer parity eval on NTREX + Bible ─────────────────────────
eval_parity_one() {
    local domain="$1"   # ntrex | bible
    local data_dir="$2"
    local out_root="$3"
    local name="$4"     # bpe | adat | paat_a100_l0 | ...
    local tok_dir="${TOKENIZERS[$name]}"

    local report="$out_root/${name}.json"
    local log="$LOG_DIR/parity_${domain}_${name}.log"

    if [[ -f "$report" ]]; then
        echo "[skip-parity] $domain/$name: report exists at $report"
        return 0
    fi
    if [[ ! -d "$data_dir" ]] || ! ls "$data_dir"/*.jsonl >/dev/null 2>&1; then
        echo "[skip-parity] $domain/$name: no data files in $data_dir"
        return 0
    fi
    echo ""
    echo "  [parity/$domain] $name  ->  $report"
    "$PYTHON" scripts/eval_parity.py \
        --tokenizer  "$tok_dir" \
        --flores-dir "$data_dir" \
        --output     "$report" \
        2>&1 | tee "$log"
}

if [[ "$SKIP_PARITY" != "1" ]]; then
    if [[ "$SKIP_NTREX" != "1" ]]; then
        echo ""
        echo "================================================================"
        echo "  [2a] Parity eval on NTREX-128"
        echo "================================================================"
        for tok in $EVAL_TOKENIZERS; do
            eval_parity_one ntrex "$NTREX_DIR" "$PARITY_NTREX_ROOT" "$tok"
        done
    fi

    if [[ "$SKIP_BIBLE" != "1" ]]; then
        echo ""
        echo "================================================================"
        echo "  [2b] Parity eval on Bible"
        echo "================================================================"
        for tok in $EVAL_TOKENIZERS; do
            eval_parity_one bible "$BIBLE_DIR" "$PARITY_BIBLE_ROOT" "$tok"
        done
    fi
else
    echo ""
    echo "[SKIP_PARITY=1] skipping per-tokenizer parity eval"
fi

# ── 3. Per-corpus comparison.json (mirrors exp_1's join logic) ────────────
write_comparison() {
    local domain="$1"
    local root="$2"
    local out="$root/comparison.json"
    "$PYTHON" - "$root" "$out" "$EVAL_TOKENIZERS" <<'PY'
import json, sys
from pathlib import Path

root = Path(sys.argv[1])
out_path = Path(sys.argv[2])
names = sys.argv[3].split()

rows = []
for n in names:
    p = root / f"{n}.json"
    if not p.exists():
        continue
    s = json.loads(p.read_text())["summary"]
    rows.append((n, s))

if not rows:
    print(f"  no parity reports found in {root}")
    sys.exit(0)

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

out_path.write_text(json.dumps({n: s for n, s in rows}, indent=2))
print(f"\n  Wrote summary to {out_path}")
PY
}

if [[ "$SKIP_COMPARE" != "1" ]]; then
    if [[ "$SKIP_NTREX" != "1" ]]; then
        echo ""
        echo "================================================================"
        echo "  [3a] NTREX per-corpus comparison"
        echo "================================================================"
        write_comparison ntrex "$PARITY_NTREX_ROOT"
    fi
    if [[ "$SKIP_BIBLE" != "1" ]]; then
        echo ""
        echo "================================================================"
        echo "  [3b] Bible per-corpus comparison"
        echo "================================================================"
        write_comparison bible "$PARITY_BIBLE_ROOT"
    fi

    # ── 4. Cross-domain comparison ───────────────────────────────────────
    echo ""
    echo "================================================================"
    echo "  [4] Cross-domain comparison (FLORES + NTREX + Bible)"
    echo "================================================================"
    log="$LOG_DIR/compare_cross_domain.log"
    "$PYTHON" scripts/compare_cross_domain.py \
        --tok-root          "$TOK_ROOT" \
        --flores-parity-dir "$PARITY_FLORES_ROOT" \
        --ntrex-parity-dir  "$PARITY_NTREX_ROOT" \
        --bible-parity-dir  "$PARITY_BIBLE_ROOT" \
        --flores-data       "$FLORES_DIR" \
        --ntrex-data        "$NTREX_DIR" \
        --bible-data        "$BIBLE_DIR" \
        --tokenizers $EVAL_TOKENIZERS \
        --output            "results/parity_cross_domain.json" \
        2>&1 | tee "$log"
else
    echo ""
    echo "[SKIP_COMPARE=1] skipping comparison stages"
fi

echo ""
echo "Stage 3 complete."
echo "  NTREX reports: $PARITY_NTREX_ROOT/"
echo "  Bible reports: $PARITY_BIBLE_ROOT/"
echo "  Cross-domain: results/parity_cross_domain.json"
echo "  Logs:         $LOG_DIR/"
