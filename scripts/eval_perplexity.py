"""
Compute held-out perplexity across languages for all pretrained LMs.

Data: FLORES+ devtest split (~997 sentences/language, 96 languages)
      pre-downloaded to data/raw/flores/<lang>.jsonl

Models: models/lm_pi/{model}  (GPT-2 architecture, HuggingFace format)

Output: results_pi/perplexity/
          perplexity.json          — nested dict: model -> lang -> ppl
          perplexity_summary.json  — per-model mean/median/std across langs

Usage:
    python scripts/eval_perplexity.py
    python scripts/eval_perplexity.py --models bpe unigram --languages en zh fr
    python scripts/eval_perplexity.py --split dev
"""
from __future__ import annotations

import argparse
import json
import math
import statistics as st
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
FLORES_DIR = ROOT / "data" / "raw" / "flores"
MODEL_DIR = ROOT / "models" / "lm_pi"
OUT_DIR = ROOT / "results_pi" / "perplexity"

ALL_MODELS = ["bpe", "parity_bpe", "unigram", "adat", "paat_a10", "paat_a100_l0"]
MAX_LEN = 512
BATCH_SIZE = 64


def load_sentences(lang: str, split: str) -> list[str]:
    path = FLORES_DIR / f"{lang}.jsonl"
    if not path.exists():
        return []
    out = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec["split"] == split:
                out.append(rec["sentence"])
    return out


def compute_perplexity(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    sentences: list[str],
    device: torch.device,
) -> float:
    """Return perplexity (exp of mean per-token NLL) over the sentence list."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0

    for i in range(0, len(sentences), BATCH_SIZE):
        batch = sentences[i : i + BATCH_SIZE]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LEN,
            padding=True,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)

        # Mask padding positions in labels so they don't contribute to loss.
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100

        with torch.no_grad():
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # out.loss is mean NLL per non-padding token; recover total NLL.
        n_tokens = (labels != -100).sum().item()
        total_nll += out.loss.item() * n_tokens
        total_tokens += n_tokens

    mean_nll = total_nll / total_tokens
    return math.exp(mean_nll)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=ALL_MODELS)
    parser.add_argument(
        "--languages",
        nargs="+",
        default=None,
        help="Language codes (default: all available in FLORES dir)",
    )
    parser.add_argument("--split", default="devtest", choices=["dev", "devtest"])
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    langs = args.languages or sorted(p.stem for p in FLORES_DIR.glob("*.jsonl"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict[str, float]] = {}

    for model_name in args.models:
        model_path = MODEL_DIR / model_name
        if not model_path.exists():
            print(f"[{model_name}] model dir not found, skipping")
            continue

        print(f"\n=== {model_name} ===")
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float32)
        model.to(device)

        lang_ppls: dict[str, float] = {}
        for lang in langs:
            sentences = load_sentences(lang, args.split)
            if not sentences:
                continue
            ppl = compute_perplexity(model, tokenizer, sentences, device)
            lang_ppls[lang] = round(ppl, 3)
            print(f"  {lang}: {ppl:.1f}")

        results[model_name] = lang_ppls

        del model
        torch.cuda.empty_cache()

    # Save full results
    out_full = args.output_dir / "perplexity.json"
    with out_full.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_full}")

    # Summary: mean / median / std per model (geometric mean for ppl)
    summary = {}
    for model_name, lang_ppls in results.items():
        vals = list(lang_ppls.values())
        log_vals = [math.log(v) for v in vals]
        geo_mean = math.exp(st.mean(log_vals))
        summary[model_name] = {
            "n_languages": len(vals),
            "geo_mean_ppl": round(geo_mean, 2),
            "median_ppl": round(st.median(vals), 2),
            "std_ppl": round(st.stdev(vals), 2) if len(vals) > 1 else 0.0,
            "min_ppl": round(min(vals), 2),
            "max_ppl": round(max(vals), 2),
        }

    out_summary = args.output_dir / "perplexity_summary.json"
    with out_summary.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_summary}")

    # Print summary table
    print("\n--- Summary ---")
    header = f"{'model':<20} {'geo_mean':>10} {'median':>10} {'std':>8} {'min':>8} {'max':>8}"
    print(header)
    for m, s in summary.items():
        print(
            f"{m:<20} {s['geo_mean_ppl']:>10.1f} {s['median_ppl']:>10.1f} "
            f"{s['std_ppl']:>8.1f} {s['min_ppl']:>8.1f} {s['max_ppl']:>8.1f}"
        )


if __name__ == "__main__":
    main()
