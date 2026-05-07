"""
Cross-domain parity comparison: FLORES vs NTREX vs Bible.

Joins per-tokenizer parity reports across three held-out evaluation
corpora and produces the headline cross-domain table for exp_3.

Why this script exists:

* The per-corpus ``comparison.json`` files are computed over each corpus's
  own language coverage, which differs (NTREX covers ~85 PAAT langs, Bible
  ~60).  Comparing those headline gini values directly is invalid — the
  language sets are different.
* This script restricts every per-tokenizer gini to the STRICT INTERSECTION
  of languages present in all three corpora, so deltas and rank
  correlations are comparable across domains.

Usage:
    python scripts/compare_cross_domain.py
    python scripts/compare_cross_domain.py \
        --flores-parity-dir results/parity \
        --ntrex-parity-dir  results/parity_ntrex \
        --bible-parity-dir  results/parity_bible \
        --flores-data       data/raw/flores \
        --ntrex-data        data/raw/ntrex \
        --bible-data        data/raw/bible \
        --output            results/parity_cross_domain.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from paat.parity.metrics import compute_parity_report, gini, report_to_dict
from paat.tokenizer.train import load_tokenizer


DOMAINS = ("flores", "ntrex", "bible")

# Tokenizer name -> directory containing tokenizer.json.  Mirrors
# scripts/exp_1_tokenizers.sh; verify against your TOK_ROOT layout.
DEFAULT_TOKENIZERS: dict[str, str] = {
    "bpe":          "{tok_root}/bpe",
    "parity_bpe":   "{tok_root}/parity_bpe",
    "adat":         "{tok_root}/adat/adat",
    "unigram":      "{tok_root}/adat/baseline",
    "paat_a033":    "{tok_root}/paat_a033/paat",
    "paat_a067":    "{tok_root}/paat_a067/paat",
    "paat_a100":    "{tok_root}/paat_a100/paat",
    "paat_a100_l0": "{tok_root}/paat_a100_l0/paat",
}


def langs_in_report(path: Path) -> set[str]:
    """Return the set of language codes for which a per-corpus report has
    valid (non-empty) per-language stats."""
    if not path.exists():
        return set()
    rep = json.loads(path.read_text())
    return {lang["lang"] for lang in rep.get("languages", [])
            if lang.get("n_sentences", 0) > 0}


def lang_stats_in_report(path: Path) -> dict[str, dict]:
    """Map lang -> per-language stats dict from a per-corpus report."""
    if not path.exists():
        return {}
    rep = json.loads(path.read_text())
    return {lang["lang"]: lang for lang in rep.get("languages", [])}


def spearman(x: list[float], y: list[float]) -> float:
    """Spearman rank correlation. No scipy dependency."""
    if len(x) != len(y) or len(x) < 2:
        return float("nan")
    rx = _ranks(x)
    ry = _ranks(y)
    rx_a, ry_a = np.array(rx), np.array(ry)
    rx_a -= rx_a.mean()
    ry_a -= ry_a.mean()
    denom = float(np.sqrt((rx_a ** 2).sum() * (ry_a ** 2).sum()))
    if denom == 0.0:
        return float("nan")
    return float((rx_a * ry_a).sum() / denom)


def _ranks(values: list[float]) -> list[float]:
    """Average ranks (handles ties)."""
    indexed = sorted(enumerate(values), key=lambda p: p[1])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(indexed):
        j = i
        while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
            j += 1
        avg_rank = (i + j) / 2 + 1   # 1-indexed average
        for k in range(i, j + 1):
            ranks[indexed[k][0]] = avg_rank
        i = j + 1
    return ranks


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-domain parity comparison (FLORES + NTREX + Bible)."
    )
    parser.add_argument("--tok-root", default="models/tokenizers")
    parser.add_argument("--flores-parity-dir", type=Path, default=Path("results/parity"))
    parser.add_argument("--ntrex-parity-dir",  type=Path, default=Path("results/parity_ntrex"))
    parser.add_argument("--bible-parity-dir",  type=Path, default=Path("results/parity_bible"))
    parser.add_argument("--flores-data", type=Path, default=Path("data/raw/flores"))
    parser.add_argument("--ntrex-data",  type=Path, default=Path("data/raw/ntrex"))
    parser.add_argument("--bible-data",  type=Path, default=Path("data/raw/bible"))
    parser.add_argument("--tokenizers", nargs="+", default=list(DEFAULT_TOKENIZERS.keys()),
                        help="Subset of tokenizers to include (default: all 8 from exp_1).")
    parser.add_argument("--output", type=Path,
                        default=Path("results/parity_cross_domain.json"))
    parser.add_argument("--unk-threshold", type=float, default=0.01,
                        help="Flag (lang, tokenizer, domain) cells with unk_rate above this.")
    parser.add_argument("--tps-ratio-threshold", type=float, default=3.0,
                        help="Flag cells where tokens_per_sentence is more than "
                             "this multiple of the same lang's FLORES value.")
    args = parser.parse_args()

    parity_dirs = {
        "flores": args.flores_parity_dir,
        "ntrex":  args.ntrex_parity_dir,
        "bible":  args.bible_parity_dir,
    }
    data_dirs = {
        "flores": args.flores_data,
        "ntrex":  args.ntrex_data,
        "bible":  args.bible_data,
    }

    # ── Build strict language intersection across all (tokenizer, domain) ──
    # A language must appear with valid stats in every (tokenizer, domain)
    # to be included.  This is stricter than per-corpus intersection but
    # is the only way to get apples-to-apples gini comparisons.
    intersection: set[str] | None = None
    for tok in args.tokenizers:
        for dom in DOMAINS:
            langs = langs_in_report(parity_dirs[dom] / f"{tok}.json")
            if intersection is None:
                intersection = langs
            else:
                intersection &= langs
    intersection_langs = sorted(intersection or set())

    if not intersection_langs:
        raise RuntimeError(
            "Empty cross-domain language intersection.  Likely causes:\n"
            "  - one or more per-tokenizer parity reports are missing\n"
            "  - one corpus (probably Bible) covers very few PAAT langs\n"
            f"Checked tokenizers: {args.tokenizers}\n"
            f"Checked dirs: {parity_dirs}"
        )
    print(f"[cross-domain] {len(intersection_langs)} languages in strict intersection")
    print(f"  {' '.join(intersection_langs)}")

    # ── Recompute per-(tokenizer, domain) gini on the intersection ──
    # The headline gini in each per-corpus comparison.json is over that
    # corpus's full lang coverage; for cross-domain comparison we MUST
    # restrict to the intersection so gini values are comparable.
    per_tokenizer: dict[str, dict] = {}
    suspicious: list[dict] = []

    bpe_gini: dict[str, float] = {}   # filled when we hit "bpe"

    for tok in args.tokenizers:
        tok_dir = Path(DEFAULT_TOKENIZERS[tok].format(tok_root=args.tok_root))
        if not (tok_dir / "tokenizer.json").exists():
            print(f"  [skip] {tok}: no tokenizer at {tok_dir}")
            continue
        tokenizer = load_tokenizer(tok_dir)

        per_dom: dict[str, float] = {}
        per_dom_stats: dict[str, dict[str, dict]] = {}
        for dom in DOMAINS:
            print(f"  [{tok}/{dom}] recomputing gini on intersection ...")
            report = compute_parity_report(
                tokenizer,
                data_dirs[dom],
                intersection_langs,
            )
            per_dom[dom] = round(report.gini_tokens_per_sentence, 4)

            d = report_to_dict(report)
            per_dom_stats[dom] = {l["lang"]: l for l in d["languages"]}

        per_tokenizer[tok] = {
            "gini_flores": per_dom["flores"],
            "gini_ntrex":  per_dom["ntrex"],
            "gini_bible":  per_dom["bible"],
        }

        if tok == "bpe":
            bpe_gini = dict(per_dom)

        # Suspicious-lang flags: high unk rate or runaway fertility ratio
        # vs the same language's FLORES tok/sent.
        flores_stats = per_dom_stats["flores"]
        for dom in DOMAINS:
            for lang, stats in per_dom_stats[dom].items():
                if stats["unk_rate"] > args.unk_threshold:
                    suspicious.append({
                        "lang": lang, "tokenizer": tok, "domain": dom,
                        "unk_rate": stats["unk_rate"],
                    })
                if dom == "flores":
                    continue
                ftps = flores_stats.get(lang, {}).get("tokens_per_sentence", 0.0)
                if ftps > 0:
                    ratio = stats["tokens_per_sentence"] / ftps
                    if ratio > args.tps_ratio_threshold:
                        suspicious.append({
                            "lang": lang, "tokenizer": tok, "domain": dom,
                            "tok_per_sent_ratio": round(ratio, 3),
                            "tps_flores": ftps,
                            "tps_domain": stats["tokens_per_sentence"],
                        })

    # Per-tokenizer delta vs BPE per domain (sign and ordering are the
    # load-bearing numbers — magnitudes between domains are not comparable).
    if bpe_gini:
        for tok, row in per_tokenizer.items():
            row["delta_vs_bpe"] = {
                dom: round(row[f"gini_{dom}"] - bpe_gini[dom], 4)
                for dom in DOMAINS
            }

    # Spearman rank correlation between domain pairs (over tokenizers).
    tok_order = [t for t in args.tokenizers if t in per_tokenizer]
    rank_corr = {}
    for a, b in [("flores", "ntrex"), ("flores", "bible"), ("ntrex", "bible")]:
        xs = [per_tokenizer[t][f"gini_{a}"] for t in tok_order]
        ys = [per_tokenizer[t][f"gini_{b}"] for t in tok_order]
        rho = spearman(xs, ys)
        rank_corr[f"{a}_vs_{b}"] = round(rho, 4) if not np.isnan(rho) else None

    out = {
        "intersection_languages": intersection_langs,
        "n_languages": len(intersection_langs),
        "per_tokenizer": per_tokenizer,
        "rank_correlation": rank_corr,
        "suspicious_languages": suspicious,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nWrote {args.output}")

    # ── Console table (mirrors exp_1's parity comparison style) ──
    print()
    print("=" * 78)
    print(f"  Cross-domain parity (lower = fairer; gini on {len(intersection_langs)}-lang intersection)")
    print("=" * 78)
    name_w = max(len(n) for n in tok_order) + 2 if tok_order else 12
    headers = ["gini(FLORES)", "gini(NTREX)", "gini(Bible)",
               "Δ FLORES", "Δ NTREX", "Δ Bible"]
    print(f"  {'tokenizer':<{name_w}}  " + "  ".join(f"{h:>13}" for h in headers))
    print("  " + "-" * (name_w + 15 * len(headers)))
    for tok in tok_order:
        r = per_tokenizer[tok]
        d = r.get("delta_vs_bpe", {"flores": 0, "ntrex": 0, "bible": 0})
        vals = [r["gini_flores"], r["gini_ntrex"], r["gini_bible"],
                d["flores"], d["ntrex"], d["bible"]]
        print(f"  {tok:<{name_w}}  " + "  ".join(f"{v:>13.4f}" for v in vals))

    print()
    print("  Spearman rank correlation between domain pairs (over tokenizers):")
    for k, v in rank_corr.items():
        print(f"    {k:<22} {v}")

    if suspicious:
        print(f"\n  {len(suspicious)} suspicious cells flagged "
              f"(unk_rate > {args.unk_threshold} or tok/sent > {args.tps_ratio_threshold}× FLORES):")
        for s in suspicious[:15]:
            print(f"    {s}")
        if len(suspicious) > 15:
            print(f"    ... ({len(suspicious) - 15} more in {args.output})")


if __name__ == "__main__":
    main()
