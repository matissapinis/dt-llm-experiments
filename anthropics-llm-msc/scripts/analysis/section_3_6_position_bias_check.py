#!/usr/bin/env python3
"""RQ4 — Position (row-order) bias.

Pre-registered primary hypothesis (two-sided):
  For each (model, mode, question_type), P(A | row=12) + P(A | row=21) ≠ 1.

Under no position bias + any content preference of strength p_ssa:
  P(A | row=12) = p_ssa       (row=12 maps SSA→A)
  P(A | row=21) = 1 - p_ssa   (row=21 maps SSA→B, so SSA-preferring picks B = not A)
  Sum = 1.

Under letter-A bias of strength b (mixed with content preference):
  Sum = 1 + b.

Bias estimate = P(A|row=12) + P(A|row=21) - 1.
Bias > 0: A-biased. Bias < 0: B-biased. |Bias| > 0.10 flagged as substantively meaningful.

Statistical test: two-sided z-test on the bias estimate, with Bonferroni correction
within RQ4 (α = 0.05 / 48 ≈ 0.00104, where 48 = 12 (model,mode) × 4 q-types).
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

D = Path("experiment_results/main_run_20260516")
N_TESTS = 48  # 12 (model, mode) × 4 question types
ALPHA_FAMILY = 0.05
ALPHA_BONF = ALPHA_FAMILY / N_TESTS  # ≈ 0.00104
BIAS_THRESHOLD = 0.10  # |bias| > 10pp flagged as substantively meaningful


def parse_mode(filename: str) -> str:
    """Extract 'on' or 'off' from filename."""
    if filename.endswith("_on.json"):
        return "on"
    if filename.endswith("_off.json"):
        return "off"
    return "?"


def load_dataset() -> list[dict]:
    """Load all 13,824 cells into a list of dicts with derived fields.

    Derived fields added per cell:
      - 'mode': 'on' or 'off' (from filename suffix)
      - 'model_short': short model name without provider prefix or date suffix
      - '_filename': source filename
    """
    cells = []
    for f in sorted(D.glob("*.json")):
        d = json.load(open(f))
        d["_filename"] = f.name
        d["mode"] = parse_mode(f.name)
        model = d.get("model_id_openrouter") or ""
        d["model_short"] = model.split("/")[-1]
        cells.append(d)
    return cells


def z_test_bias(p12: float, n12: int, p21: float, n21: int) -> tuple[float, float, float]:
    """Two-sided z-test on bias = p12 + p21 - 1.

    Returns (bias_estimate, standard_error, p_value).
    """
    bias = p12 + p21 - 1.0
    se = math.sqrt(p12 * (1 - p12) / n12 + p21 * (1 - p21) / n21)
    if se == 0:
        return (bias, 0.0, 1.0 if bias == 0 else 0.0)
    z = bias / se
    # two-sided p-value via normal CDF approximation
    p_value = 2 * (1 - phi(abs(z)))
    return (bias, se, p_value)


def phi(z: float) -> float:
    """Standard normal CDF via erf approximation."""
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def main() -> int:
    cells = load_dataset()
    print(f"Loaded {len(cells)} cells from {D}")

    # Aggregate per (model_short, mode, question_type, row_order): count A, B, null
    counts: dict = defaultdict(lambda: {"A": 0, "B": 0, "null": 0})
    for c in cells:
        key = (c["model_short"], c["mode"], c.get("question_type"), c.get("row_order"))
        ch = c.get("extracted_choice")
        if ch in ("A", "B"):
            counts[key][ch] += 1
        else:
            counts[key]["null"] += 1

    # Compute per (model, mode, q-type) the position-bias test
    results = []
    keys_mmq = sorted({(k[0], k[1], k[2]) for k in counts.keys()})
    for model, mode, qt in keys_mmq:
        c12 = counts.get((model, mode, qt, "12"), {"A": 0, "B": 0, "null": 0})
        c21 = counts.get((model, mode, qt, "21"), {"A": 0, "B": 0, "null": 0})
        n12 = c12["A"] + c12["B"]
        n21 = c21["A"] + c21["B"]
        if n12 == 0 or n21 == 0:
            continue
        p12 = c12["A"] / n12
        p21 = c21["A"] / n21
        bias, se, p_value = z_test_bias(p12, n12, p21, n21)
        significant_bonf = p_value < ALPHA_BONF
        substantively_large = abs(bias) > BIAS_THRESHOLD
        results.append({
            "model": model, "mode": mode, "q_type": qt,
            "n12": n12, "n21": n21,
            "p12_A": p12, "p21_A": p21,
            "bias": bias, "se": se, "p_value": p_value,
            "sig_bonf": significant_bonf,
            "large_bias": substantively_large,
        })

    # Sort by |bias| descending for the report
    results.sort(key=lambda r: -abs(r["bias"]))

    # Print summary table
    print(f"\n=== RQ4 results — {len(results)} tests, Bonferroni α = {ALPHA_BONF:.5f} ===")
    print(f"{'model':<32} {'mode':<4} {'q-type':<22} {'n12/n21':<10} "
          f"{'P(A|12)':<8} {'P(A|21)':<8} {'bias':<8} {'SE':<7} {'p':<10} {'sig':<4} {'large':<5}")
    print("-" * 130)
    for r in results:
        sig_flag = "**" if r["sig_bonf"] else ""
        large_flag = "L" if r["large_bias"] else ""
        print(f"{r['model']:<32} {r['mode']:<4} {r['q_type']:<22} "
              f"{r['n12']}/{r['n21']:<6} "
              f"{r['p12_A']:<8.3f} {r['p21_A']:<8.3f} "
              f"{r['bias']:+.3f}   {r['se']:<7.4f} {r['p_value']:<10.6g} "
              f"{sig_flag:<4} {large_flag:<5}")

    # Summary statistics
    n_sig = sum(1 for r in results if r["sig_bonf"])
    n_large = sum(1 for r in results if r["large_bias"])
    n_both = sum(1 for r in results if r["sig_bonf"] and r["large_bias"])
    print(f"\n=== Summary ===")
    print(f"  Total tests: {len(results)}")
    print(f"  Bonferroni-significant (p < {ALPHA_BONF:.5f}): {n_sig}")
    print(f"  |bias| > {BIAS_THRESHOLD} (substantively meaningful): {n_large}")
    print(f"  Both significant AND substantively large: {n_both}")

    # Per-model summary: how many of the 4 q-types show large bias for each (model, mode)?
    print(f"\n=== Per-(model, mode) large-bias count out of 4 q-types ===")
    per_mm: dict = defaultdict(lambda: {"large": 0, "sig": 0, "total": 0})
    for r in results:
        per_mm[(r["model"], r["mode"])]["total"] += 1
        if r["large_bias"]:
            per_mm[(r["model"], r["mode"])]["large"] += 1
        if r["sig_bonf"]:
            per_mm[(r["model"], r["mode"])]["sig"] += 1
    for (model, mode), v in sorted(per_mm.items()):
        print(f"  {model:<32} {mode:<4}  large/total = {v['large']}/{v['total']}, "
              f"sig/total = {v['sig']}/{v['total']}")

    # Direction summary: A-biased vs B-biased
    n_A_biased = sum(1 for r in results if r["large_bias"] and r["bias"] > 0)
    n_B_biased = sum(1 for r in results if r["large_bias"] and r["bias"] < 0)
    print(f"\n=== Direction of large biases ===")
    print(f"  A-biased (bias > +{BIAS_THRESHOLD}): {n_A_biased}")
    print(f"  B-biased (bias < -{BIAS_THRESHOLD}): {n_B_biased}")

    # =================================================================
    # SECONDARY EXPLORATORY ANALYSIS A:
    # Per (model, mode, q-type, problem-base) — finer-grained bias
    # =================================================================
    print(f"\n\n{'=' * 100}")
    print("SECONDARY EXPLORATORY ANALYSIS A: per (model, mode, q-type, problem-base) bias")
    print(f"{'=' * 100}")

    # Re-aggregate counts at the finer grain
    fine_counts: dict = defaultdict(lambda: {"A": 0, "B": 0, "null": 0})
    for c in cells:
        # Problem-base = template_name without the row_order suffix (_12 or _21)
        tmpl = c.get("template_name") or ""
        base = re.sub(r"_(12|21)$", "", tmpl)
        key = (c["model_short"], c["mode"], c.get("question_type"), base, c.get("row_order"))
        ch = c.get("extracted_choice")
        if ch in ("A", "B"):
            fine_counts[key][ch] += 1
        else:
            fine_counts[key]["null"] += 1

    # For each (model, mode, q_type, problem-base), compute bias
    fine_results = []
    fine_keys = sorted({(k[0], k[1], k[2], k[3]) for k in fine_counts.keys()})
    for model, mode, qt, base in fine_keys:
        c12 = fine_counts.get((model, mode, qt, base, "12"), {"A": 0, "B": 0, "null": 0})
        c21 = fine_counts.get((model, mode, qt, base, "21"), {"A": 0, "B": 0, "null": 0})
        n12 = c12["A"] + c12["B"]
        n21 = c21["A"] + c21["B"]
        if n12 < 3 or n21 < 3:
            continue
        p12 = c12["A"] / n12
        p21 = c21["A"] / n21
        bias, se, p_value = z_test_bias(p12, n12, p21, n21)
        fine_results.append({
            "model": model, "mode": mode, "q_type": qt, "base": base,
            "n12": n12, "n21": n21,
            "p12_A": p12, "p21_A": p21,
            "bias": bias, "se": se, "p_value": p_value,
            "large_bias": abs(bias) > BIAS_THRESHOLD,
        })

    n_fine_tests = len(fine_results)
    alpha_fine_bonf = 0.05 / n_fine_tests
    n_fine_sig = sum(1 for r in fine_results if r["p_value"] < alpha_fine_bonf)
    n_fine_large = sum(1 for r in fine_results if r["large_bias"])
    n_fine_both = sum(1 for r in fine_results if r["large_bias"] and r["p_value"] < alpha_fine_bonf)
    print(f"  Total fine-grained tests: {n_fine_tests}")
    print(f"  Bonferroni-significant at α = {alpha_fine_bonf:.6f}: {n_fine_sig}")
    print(f"  |bias| > {BIAS_THRESHOLD}: {n_fine_large}")
    print(f"  Both: {n_fine_both}")

    # Show top-15 largest absolute biases
    fine_results.sort(key=lambda r: -abs(r["bias"]))
    print(f"\n  Top 15 largest |bias| at fine grain:")
    print(f"  {'model':<28} {'mode':<4} {'q-type':<22} {'problem-base':<40} "
          f"{'P(A|12)':<8} {'P(A|21)':<8} {'bias':<8} {'p':<10}")
    for r in fine_results[:15]:
        print(f"  {r['model']:<28} {r['mode']:<4} {r['q_type']:<22} {r['base'][:40]:<40} "
              f"{r['p12_A']:<8.3f} {r['p21_A']:<8.3f} "
              f"{r['bias']:+.3f}   {r['p_value']:<10.6g}")

    # =================================================================
    # SECONDARY EXPLORATORY ANALYSIS B:
    # Attitude-only letter preference per (model, mode) — pooled across attitude q-types
    # =================================================================
    print(f"\n\n{'=' * 100}")
    print("SECONDARY EXPLORATORY ANALYSIS B: attitude-only letter preference per (model, mode)")
    print(f"{'=' * 100}")

    # Pool attitude cells per (model, mode, row_order)
    att_counts: dict = defaultdict(lambda: {"A": 0, "B": 0, "null": 0})
    for c in cells:
        qt = c.get("question_type") or ""
        if qt not in ("normative_attitude", "personal_attitude"):
            continue
        key = (c["model_short"], c["mode"], c.get("row_order"))
        ch = c.get("extracted_choice")
        if ch in ("A", "B"):
            att_counts[key][ch] += 1
        else:
            att_counts[key]["null"] += 1

    att_results = []
    mm_keys = sorted({(k[0], k[1]) for k in att_counts.keys()})
    for model, mode in mm_keys:
        c12 = att_counts.get((model, mode, "12"), {"A": 0, "B": 0, "null": 0})
        c21 = att_counts.get((model, mode, "21"), {"A": 0, "B": 0, "null": 0})
        n12 = c12["A"] + c12["B"]
        n21 = c21["A"] + c21["B"]
        if n12 == 0 or n21 == 0:
            continue
        p12 = c12["A"] / n12
        p21 = c21["A"] / n21
        bias, se, p_value = z_test_bias(p12, n12, p21, n21)
        att_results.append({
            "model": model, "mode": mode,
            "n12": n12, "n21": n21,
            "p12_A": p12, "p21_A": p21,
            "bias": bias, "se": se, "p_value": p_value,
        })

    n_att = len(att_results)
    alpha_att_bonf = 0.05 / n_att
    att_results.sort(key=lambda r: r["bias"])  # ascending: most B-biased first
    print(f"  Total attitude (model, mode) tests: {n_att}, Bonferroni α = {alpha_att_bonf:.5f}")
    print(f"\n  {'model':<32} {'mode':<4} {'n12/n21':<10} {'P(A|12)':<8} {'P(A|21)':<8} "
          f"{'bias':<8} {'p':<12} {'Bonf-sig':<9} {'nominal-sig':<11} {'direction':<12}")
    print("  " + "-" * 130)
    for r in att_results:
        bonf_sig = "**" if r["p_value"] < alpha_att_bonf else ""
        nom_sig = "*" if r["p_value"] < 0.05 else ""
        direction = "B-biased" if r["bias"] < 0 else ("A-biased" if r["bias"] > 0 else "neutral")
        print(f"  {r['model']:<32} {r['mode']:<4} {r['n12']}/{r['n21']:<6} "
              f"{r['p12_A']:<8.3f} {r['p21_A']:<8.3f} "
              f"{r['bias']:+.3f}   {r['p_value']:<12.6g} "
              f"{bonf_sig:<9} {nom_sig:<11} {direction:<12}")

    # Population-level direction summary
    n_A_dir = sum(1 for r in att_results if r["bias"] > 0)
    n_B_dir = sum(1 for r in att_results if r["bias"] < 0)
    n_A_sig = sum(1 for r in att_results if r["bias"] > 0 and r["p_value"] < alpha_att_bonf)
    n_B_sig = sum(1 for r in att_results if r["bias"] < 0 and r["p_value"] < alpha_att_bonf)
    n_A_nom = sum(1 for r in att_results if r["bias"] > 0 and r["p_value"] < 0.05)
    n_B_nom = sum(1 for r in att_results if r["bias"] < 0 and r["p_value"] < 0.05)
    print(f"\n  === Direction summary across {n_att} (model, mode) pairs ===")
    print(f"  A-biased (bias > 0): {n_A_dir} (Bonferroni-sig: {n_A_sig}, nominal-sig: {n_A_nom})")
    print(f"  B-biased (bias < 0): {n_B_dir} (Bonferroni-sig: {n_B_sig}, nominal-sig: {n_B_nom})")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
