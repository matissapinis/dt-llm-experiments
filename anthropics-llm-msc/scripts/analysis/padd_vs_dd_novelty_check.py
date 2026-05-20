#!/usr/bin/env python3
"""RQ7 — PA-DD novelty effect.

Hypothesis (directional): PADD (less canonical in literature than standard DD)
shows novelty effects compared to DD, manifesting as:
  1. Higher within-cell variance / lower self-consistency on PADD
  2. Lower cross-model agreement on PADD
  3. Lower capability accuracy on PADD
  4. Higher off-menu / refusal rate on PADD (already partially shown in RQ8)

Tests:
  1. Within-cell entropy: paired sign test on (model, mode, theme, row, param, q_type)
     across DD and PADD problems.
  2. Cross-model agreement: per (theme, row, param, q_type), entropy of pooled
     model-mode response distribution. Paired comparison DD vs PADD.
  3. Capability accuracy: paired sign test per (model, mode) on DD vs PADD cap accuracy.
  4. Off-menu rate: chi-square on PADD vs DD off-menu rates.

Bonferroni within RQ7: α = 0.05 / 4 = 0.0125 across the 4 primary tests.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

D = Path("experiment_results/main_run_20260516")
N_PRIMARY_TESTS = 4
ALPHA_BONF = 0.05 / N_PRIMARY_TESTS  # = 0.0125
OFF_MENU_QUALITIES = {"empty_response", "no_final_answer", "off_menu_refusal"}


def parse_mode(filename: str) -> str:
    if filename.endswith("_on.json"):
        return "on"
    if filename.endswith("_off.json"):
        return "off"
    return "?"


def parse_sample(filename: str) -> int:
    m = re.search(r"_sample(\d+)_", filename)
    return int(m.group(1)) if m else -1


def parse_template_parts(template_name: str) -> dict:
    """Parse template like '20260516_standard_dd_civilization_scaled_12' into parts."""
    parts = (template_name or "").split("_")
    if len(parts) < 4:
        return {}
    out = {"problem_class": parts[2] if len(parts) > 2 else "?"}
    # parts: [date, "standard", class, theme, ?, ?, row]
    if "_scaled_" in (template_name or ""):
        out["parameterization"] = "DD-type"
        out["theme"] = parts[3] if len(parts) > 3 else "?"
        out["row_order"] = parts[-1]
    else:
        out["parameterization"] = "SB-type"
        out["theme"] = parts[3] if len(parts) > 3 else "?"
        out["row_order"] = parts[-1]
    return out


def binary_entropy(n_a: int, n_b: int) -> float:
    n = n_a + n_b
    if n == 0:
        return 0.0
    p = n_a / n
    if p == 0 or p == 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def chi2_2x2(a, b, c, d):
    n = a + b + c + d
    if n == 0:
        return (0.0, 1.0)
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    e_a = row1 * col1 / n
    e_b = row1 * col2 / n
    e_c = row2 * col1 / n
    e_d = row2 * col2 / n
    chi2 = 0.0
    for obs, exp in zip([a, b, c, d], [e_a, e_b, e_c, e_d]):
        if exp > 0:
            chi2 += (obs - exp) ** 2 / exp
    return (chi2, math.erfc(math.sqrt(chi2 / 2)))


def sign_test(differences: list[float]) -> tuple[int, int, int, float]:
    """Two-sided sign test on a list of paired differences.

    Returns (n_positive, n_negative, n_zero, p_value).
    """
    n_pos = sum(1 for d in differences if d > 0)
    n_neg = sum(1 for d in differences if d < 0)
    n_zero = sum(1 for d in differences if d == 0)
    n = n_pos + n_neg
    if n == 0:
        return (n_pos, n_neg, n_zero, 1.0)
    smaller = min(n_pos, n_neg)
    cdf = sum(math.comb(n, i) for i in range(0, smaller + 1)) * (0.5**n)
    return (n_pos, n_neg, n_zero, min(1.0, 2 * cdf))


def load_cells():
    cells = []
    for f in sorted(D.glob("*.json")):
        d = json.load(open(f))
        d["mode"] = parse_mode(f.name)
        d["sample_num"] = parse_sample(f.name)
        model = d.get("model_id_openrouter") or ""
        d["model_short"] = model.split("/")[-1]
        parts = parse_template_parts(d.get("template_name", ""))
        d.update(parts)
        cells.append(d)
    return cells


def main() -> int:
    cells = load_cells()
    print(f"Loaded {len(cells)} cells")
    print(f"Bonferroni α (4 primary tests): {ALPHA_BONF}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("TEST 1: Within-cell entropy (self-consistency) — paired DD vs PADD")
    print(f"{'=' * 100}")
    # Build per-cell entropy (cell = 9-sample group)
    cell_groups: dict = defaultdict(list)
    for c in cells:
        key = (c["model_short"], c["mode"], c["theme"], c["row_order"],
               c["parameterization"], c.get("question_type"), c.get("problem_class"))
        cell_groups[key].append(c)
    cell_entropy: dict = {}
    for key, samples in cell_groups.items():
        n_a = sum(1 for s in samples if s.get("extracted_choice") == "A")
        n_b = sum(1 for s in samples if s.get("extracted_choice") == "B")
        if n_a + n_b > 0:
            cell_entropy[key] = binary_entropy(n_a, n_b)

    # Pair cells: (model, mode, theme, row, param, q_type) × (dd, padd)
    pair_diffs = []
    paired_count = 0
    for key, ent in cell_entropy.items():
        m, mode, theme, row, param, qt, pc = key
        if pc != "dd":
            continue
        padd_key = (m, mode, theme, row, param, qt, "padd")
        if padd_key in cell_entropy:
            pair_diffs.append(cell_entropy[padd_key] - ent)
            paired_count += 1
    n_pos, n_neg, n_zero, p = sign_test(pair_diffs)
    avg_diff = sum(pair_diffs) / len(pair_diffs) if pair_diffs else 0
    print(f"\n  Paired (DD, PADD) cells: {paired_count}")
    print(f"  Differences (PADD entropy - DD entropy):")
    print(f"    n PADD > DD: {n_pos}   n PADD < DD: {n_neg}   n equal: {n_zero}")
    print(f"    avg difference: {avg_diff:+.4f}")
    print(f"    sign test p (two-sided): {p:.4g}")
    print(f"    {'**' if p < ALPHA_BONF else ('*' if p < 0.05 else '')} "
          f"{'PADD higher entropy' if avg_diff > 0 else 'DD higher entropy'}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("TEST 2: Cross-model agreement — paired DD vs PADD")
    print(f"{'=' * 100}")
    # For each (theme, row, param, q_type, problem_class): pool all 12 model-modes × 9 samples
    # → 108 responses. Compute entropy of pooled (A/B) distribution.
    # Then pair DD vs PADD.
    pooled_counts: dict = defaultdict(lambda: {"A": 0, "B": 0})
    for c in cells:
        ch = c.get("extracted_choice")
        if ch in ("A", "B"):
            key = (c["theme"], c["row_order"], c["parameterization"],
                   c.get("question_type"), c.get("problem_class"))
            pooled_counts[key][ch] += 1
    pooled_entropy: dict = {}
    for key, d in pooled_counts.items():
        pooled_entropy[key] = binary_entropy(d["A"], d["B"])
    # Pair DD vs PADD
    pair_diffs2 = []
    for key, ent in pooled_entropy.items():
        theme, row, param, qt, pc = key
        if pc != "dd":
            continue
        padd_key = (theme, row, param, qt, "padd")
        if padd_key in pooled_entropy:
            pair_diffs2.append(pooled_entropy[padd_key] - ent)
    n_pos, n_neg, n_zero, p = sign_test(pair_diffs2)
    avg_diff = sum(pair_diffs2) / len(pair_diffs2) if pair_diffs2 else 0
    print(f"\n  Paired (DD, PADD) pooled-cells (n_pairs={len(pair_diffs2)}):")
    print(f"  Differences (PADD pooled entropy - DD pooled entropy):")
    print(f"    n PADD > DD: {n_pos}   n PADD < DD: {n_neg}   n equal: {n_zero}")
    print(f"    avg difference: {avg_diff:+.4f}")
    print(f"    sign test p (two-sided): {p:.4g}")
    print(f"    {'**' if p < ALPHA_BONF else ('*' if p < 0.05 else '')} "
          f"{'PADD lower agreement' if avg_diff > 0 else 'DD lower agreement'}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("TEST 3: Capability accuracy — paired sign test (DD vs PADD per model-mode)")
    print(f"{'=' * 100}")
    cap_acc: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for c in cells:
        if not (c.get("question_type") or "").endswith("_capability"):
            continue
        if c.get("correct_capability_answer") is None:
            continue
        key = (c["model_short"], c["mode"], c.get("problem_class"))
        cap_acc[key]["total"] += 1
        if c["correct_capability_answer"]:
            cap_acc[key]["correct"] += 1

    print(f"\n  {'model':<32} {'mode':<5} {'DD acc':<14} {'PADD acc':<14} {'Δ (PADD-DD)':<13}")
    print("  " + "-" * 85)
    deltas = []
    mm_keys = sorted({(k[0], k[1]) for k in cap_acc.keys()})
    for (m, mode) in mm_keys:
        dd = cap_acc.get((m, mode, "dd"), {"correct": 0, "total": 0})
        padd = cap_acc.get((m, mode, "padd"), {"correct": 0, "total": 0})
        dd_acc = dd["correct"] / dd["total"] if dd["total"] > 0 else 0
        padd_acc = padd["correct"] / padd["total"] if padd["total"] > 0 else 0
        delta = padd_acc - dd_acc
        deltas.append(delta)
        print(f"  {m:<32} {mode:<5} {dd['correct']}/{dd['total']} ({dd_acc:.3f})  "
              f"{padd['correct']}/{padd['total']} ({padd_acc:.3f})  {delta:+.4f}")
    n_pos, n_neg, n_zero, p = sign_test(deltas)
    avg = sum(deltas) / len(deltas)
    print(f"\n  Sign test on per-(model, mode) deltas:")
    print(f"    PADD > DD: {n_pos}, PADD < DD: {n_neg}, equal: {n_zero}")
    print(f"    avg Δ: {avg:+.4f}  (positive = PADD higher accuracy)")
    print(f"    sign test p (two-sided): {p:.4g}")
    print(f"    {'**' if p < ALPHA_BONF else ('*' if p < 0.05 else '')} "
          f"{'PADD higher cap acc' if avg > 0 else 'DD higher cap acc' if avg < 0 else 'no diff'}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("TEST 4: Off-menu rate — chi-square DD vs PADD")
    print(f"{'=' * 100}")
    dd_off = sum(1 for c in cells if c.get("problem_class") == "dd"
                  and c.get("parse_quality") in OFF_MENU_QUALITIES)
    dd_total = sum(1 for c in cells if c.get("problem_class") == "dd")
    padd_off = sum(1 for c in cells if c.get("problem_class") == "padd"
                    and c.get("parse_quality") in OFF_MENU_QUALITIES)
    padd_total = sum(1 for c in cells if c.get("problem_class") == "padd")
    chi2, p = chi2_2x2(padd_off, padd_total - padd_off, dd_off, dd_total - dd_off)
    dd_rate = dd_off / dd_total
    padd_rate = padd_off / padd_total
    print(f"\n  DD:   {dd_off}/{dd_total} off-menu ({dd_rate:.4f})")
    print(f"  PADD: {padd_off}/{padd_total} off-menu ({padd_rate:.4f})")
    print(f"  Δ (PADD - DD): {(padd_rate-dd_rate)*100:+.2f}pp")
    print(f"  χ² = {chi2:.3f}, p = {p:.4g}")
    print(f"  {'**' if p < ALPHA_BONF else ('*' if p < 0.05 else '')} "
          f"{'PADD higher off-menu' if padd_rate > dd_rate else 'DD higher off-menu'}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("SUMMARY: PA-DD novelty effect across 4 pre-registered tests")
    print(f"{'=' * 100}")
    print(f"\n  Test                                       direction         Bonferroni-sig?")
    print(f"  ---------------------------------------------------------------------------")
    print(f"  1. Within-cell entropy (self-consistency)  see above         see above")
    print(f"  2. Cross-model agreement (pooled entropy)  see above         see above")
    print(f"  3. Capability accuracy                     see above         see above")
    print(f"  4. Off-menu rate                           see above         see above")
    print(f"\n  Note: all 4 tests are exploratory/secondary. Sign tests are conservative.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
