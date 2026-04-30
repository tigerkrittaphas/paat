"""
Evaluate a pre-trained LM on downstream benchmarks using lm-evaluation-harness.

Runs two benchmark suites:
  1. Paper tasks (Zheng et al. 2024): PIQA, ARC-Easy, ARC-Challenge, SciQ, Lambada — 5-shot.
  2. XNLI (multilingual NLI, 15 languages) — 0-shot. Tests cross-lingual fairness,
     directly complementing the tokenizer parity metrics.

Requires lm-evaluation-harness >= 0.4:
    pip install lm-eval

Usage:
    # Evaluate one model on both suites
    python scripts/eval_downstream.py \\
        --model-dir models/lm/adat \\
        --output-dir results/downstream/adat

    # Evaluate all three models in sequence
    python scripts/eval_downstream.py \\
        --model-dir models/lm/adat models/lm/unigram models/lm/bpe \\
        --output-dir results/downstream
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


# Tasks from the paper (Table 1, Zheng et al. 2024) — 5-shot.
PAPER_TASKS = [
    "piqa",
    "arc_easy",
    "arc_challenge",
    "sciq",
    "lambada_openai",
]

# XNLI languages available in lm-eval-harness.
# Covers ar, de, el, en, es, fr, hi, ru, sw, th, tr, ur, vi, zh
# — overlaps well with our 96-language mC4 set.
XNLI_TASKS = [
    "xnli_ar", "xnli_de", "xnli_en", "xnli_es",
    "xnli_fr", "xnli_hi", "xnli_ru", "xnli_sw",
    "xnli_th", "xnli_tr", "xnli_ur", "xnli_vi",
    "xnli_zh",
]


def run_lm_eval(
    model_dir: Path,
    tasks: list[str],
    output_path: Path,
    num_fewshot: int,
    batch_size: int = 16,
) -> dict:
    """Run lm_eval CLI and return the parsed results dict."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "lm_eval",
        "--model", "hf",
        "--model_args", f"pretrained={model_dir},dtype=bfloat16",
        "--tasks", ",".join(tasks),
        "--num_fewshot", str(num_fewshot),
        "--batch_size", str(batch_size),
        "--output_path", str(output_path),
        "--log_samples",
    ]
    print(f"\n{'='*60}")
    print(f"  Tasks: {tasks}")
    print(f"  Model: {model_dir}")
    print(f"  Output: {output_path}")
    print(f"{'='*60}\n")
    print("Running:", " ".join(cmd))

    result = subprocess.run(cmd, text=True)
    if result.returncode != 0:
        print(f"\n[WARN] lm_eval failed (exit {result.returncode}) for {model_dir} on {tasks}.")
        print("[WARN] Continuing with remaining models/tasks.\n")
        return {}
    # lm_eval may write either results.json (older) or results_<timestamp>.json
    # under either output_path or a model-named subdir. Pick the most recent.
    candidates = sorted(output_path.rglob("results*.json"), key=lambda p: p.stat().st_mtime)
    if candidates:
        with candidates[-1].open() as f:
            return json.load(f)
    print(f"[WARN] no results JSON found under {output_path}")
    return {}


def extract_summary(results: dict, tasks: list[str]) -> dict:
    """Pull per-task accuracy/acc_norm from lm_eval output."""
    summary = {}
    for task in tasks:
        task_results = results.get("results", {}).get(task, {})
        # lm_eval uses "acc,none" or "acc_norm,none" as keys depending on version.
        acc = (
            task_results.get("acc,none")
            or task_results.get("acc_norm,none")
            or task_results.get("acc")
            or task_results.get("acc_norm")
        )
        summary[task] = round(float(acc) * 100, 2) if acc is not None else None
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run lm-evaluation-harness on pre-trained PAAT models."
    )
    parser.add_argument(
        "--model-dir", type=Path, nargs="+", required=True,
        help="One or more model directories (HuggingFace format from pretrain_lm.py).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/downstream"),
        help="Root directory for results. Per-model subdirs are created automatically.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16,
        help="Batch size for lm_eval inference.",
    )
    parser.add_argument(
        "--skip-xnli", action="store_true",
        help="Skip the XNLI multilingual evaluation.",
    )
    args = parser.parse_args()

    all_summaries: dict[str, dict] = {}

    for model_dir in args.model_dir:
        if not model_dir.exists():
            print(f"WARNING: {model_dir} does not exist, skipping.")
            continue

        name = model_dir.name
        model_out = args.output_dir / name
        summaries: dict[str, dict] = {}

        # ── Paper tasks (5-shot) ───────────────────────────────────────────
        paper_out = model_out / "paper_tasks"
        paper_results = run_lm_eval(
            model_dir, PAPER_TASKS, paper_out,
            num_fewshot=5, batch_size=args.batch_size,
        )
        summaries["paper_tasks"] = extract_summary(paper_results, PAPER_TASKS)
        avg = [v for v in summaries["paper_tasks"].values() if v is not None]
        summaries["paper_tasks"]["avg"] = round(sum(avg) / len(avg), 2) if avg else None

        # ── XNLI (0-shot multilingual) ─────────────────────────────────────
        if not args.skip_xnli:
            xnli_out = model_out / "xnli"
            xnli_results = run_lm_eval(
                model_dir, XNLI_TASKS, xnli_out,
                num_fewshot=0, batch_size=args.batch_size,
            )
            summaries["xnli"] = extract_summary(xnli_results, XNLI_TASKS)
            xnli_scores = [v for v in summaries["xnli"].values() if v is not None]
            summaries["xnli"]["avg"] = round(sum(xnli_scores) / len(xnli_scores), 2) if xnli_scores else None

        all_summaries[name] = summaries

    # ── Comparison table ───────────────────────────────────────────────────
    print("\n\n" + "=" * 70)
    print("DOWNSTREAM RESULTS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<20}  {'PIQA':>6}  {'ARC-E':>6}  {'ARC-C':>6}  {'SciQ':>6}  {'Lam':>6}  {'Avg':>6}")
    print("-" * 70)
    for name, sums in all_summaries.items():
        pt = sums.get("paper_tasks", {})
        print(
            f"  {name:<18}  "
            f"{pt.get('piqa') or '—':>6}  "
            f"{pt.get('arc_easy') or '—':>6}  "
            f"{pt.get('arc_challenge') or '—':>6}  "
            f"{pt.get('sciq') or '—':>6}  "
            f"{pt.get('lambada_openai') or '—':>6}  "
            f"{pt.get('avg') or '—':>6}"
        )

    if not args.skip_xnli:
        print(f"\n{'Model':<20}  {'en':>6}  {'ar':>6}  {'hi':>6}  {'zh':>6}  {'sw':>6}  {'Avg':>6}")
        print("-" * 70)
        for name, sums in all_summaries.items():
            xl = sums.get("xnli", {})
            print(
                f"  {name:<18}  "
                f"{xl.get('xnli_en') or '—':>6}  "
                f"{xl.get('xnli_ar') or '—':>6}  "
                f"{xl.get('xnli_hi') or '—':>6}  "
                f"{xl.get('xnli_zh') or '—':>6}  "
                f"{xl.get('xnli_sw') or '—':>6}  "
                f"{xl.get('avg') or '—':>6}"
            )

    # Save combined summary.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "comparison.json"
    with summary_path.open("w") as f:
        json.dump(all_summaries, f, indent=2)
    print(f"\nFull summary saved to {summary_path}")


if __name__ == "__main__":
    main()
