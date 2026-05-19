#!/usr/bin/env python3
"""RQ2 — Capability accuracy.

Pre-registered primary hypothesis (one-sided):
  Per (model, mode), accuracy on capability questions > 0.5 (chance).

Data subset: capability cells (ssa_capability + sia_capability) with parsed
extracted_choice and a derived correct_capability_answer.

Tests:
  - One-sided binomial test (normal approximation, plus exact for extremes).
  - 95% Wilson score confidence intervals.
  - Bonferroni correction: α = 0.05 / 12 ≈ 0.00417 (12 model-mode configurations).

Secondary breakdowns:
  - Per-model accuracy by question type (SSA-cap vs SIA-cap) — anchored to QT9.
  - Per-model accuracy by problem class (SB / INC / DD / PADD).
  - Per-model accuracy by cluster (SB-type / DD-type).
  - Sensitivity: parse_quality=strict_clean only vs all parsed.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

D = Path("experiment_results/main_run_20260516")
N_TESTS = 12  # 12 (model, mode) configurations
ALPHA_FAMILY = 0.05
ALPHA_BONF = ALPHA_FAMILY / N_TESTS  # ≈ 0.00417
NULL_P = 0.5
Z_95 = 1.96  # 95% CI two-sided z-score


def parse_mode(filename: str) -> str:
    if filename.endswith("_on.json"):
        return "on"
    if filename.endswith("_off.json"):
        return "off"
    return "?"


def parse_problem_class(template_name: str) -> str:
    m = re.match(r"\d+_standard_(sb|inc|dd|padd)_", template_name or "")
    return m.group(1) if m else "?"


def parse_cluster(problem_class: str) -> str:
    if problem_class in ("sb", "inc"):
        return "SB-type"
    if problem_class in ("dd", "padd"):
        return "DD-type"
    return "?"


def load_dataset() -> list[dict]:
    cells = []
    for f in sorted(D.glob("*.json")):
        d = json.load(open(f))
        d["_filename"] = f.name
        d["mode"] = parse_mode(f.name)
        model = d.get("model_id_openrouter") or ""
        d["model_short"] = model.split("/")[-1]
        d["problem_class"] = parse_problem_class(d.get("template_name", ""))
        d["cluster"] = parse_cluster(d["problem_class"])
        cells.append(d)
    return cells


def wilson_ci(k: int, n: int, z: float = Z_95) -> tuple[float, float]:
    """95% Wilson score interval for k/n binomial proportion."""
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half_width = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def chi2_2x2(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Chi-square test of independence on a 2×2 table.

    Returns (chi2_statistic, p_value).
    """
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
    p_value = math.erfc(math.sqrt(chi2 / 2))
    return (chi2, p_value)


def binomial_one_sided_p(k: int, n: int, p0: float) -> float:
    """One-sided binomial test: P(X >= k | n, p0).

    Exact for extreme accuracies; normal approximation otherwise.
    """
    if n == 0:
        return 1.0
    # Exact for extreme cases
    if k == n:
        return p0**n
    if k == 0:
        return 1.0
    # Normal approximation
    p_hat = k / n
    se = math.sqrt(p0 * (1 - p0) / n)
    if se == 0:
        return 1.0 if p_hat < p0 else 0.0
    z = (p_hat - p0) / se
    if z <= 0:
        return 1.0  # one-sided test, p_hat below null, no support for H1
    # Survival of standard normal at z
    return 0.5 * math.erfc(z / math.sqrt(2))


def main() -> int:
    cells = load_dataset()
    print(f"Loaded {len(cells)} cells")

    # Filter to capability cells with valid correct_capability_answer
    cap_cells = [
        c for c in cells
        if (c.get("question_type") or "").endswith("_capability")
        and c.get("correct_capability_answer") is not None
    ]
    print(f"Capability cells with valid correctness: {len(cap_cells)}\n")

    # =================================================================
    print(f"{'=' * 80}")
    print("PRIMARY: per-(model, mode) capability accuracy with 95% Wilson CI")
    print(f"Bonferroni α = {ALPHA_BONF:.5f}, one-sided null p_0 = {NULL_P}")
    print(f"{'=' * 80}")

    by_mm: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for c in cap_cells:
        key = (c["model_short"], c["mode"])
        by_mm[key]["total"] += 1
        if c["correct_capability_answer"]:
            by_mm[key]["correct"] += 1

    results = []
    for (m, mode), v in by_mm.items():
        k, n = v["correct"], v["total"]
        acc = k / n if n > 0 else 0.0
        lo, hi = wilson_ci(k, n)
        p_val = binomial_one_sided_p(k, n, NULL_P)
        results.append({
            "model": m, "mode": mode,
            "correct": k, "total": n,
            "acc": acc, "ci_lo": lo, "ci_hi": hi,
            "p_value": p_val,
            "sig_bonf": p_val < ALPHA_BONF,
        })

    # Sort by accuracy descending
    results.sort(key=lambda r: -r["acc"])
    print(f"\n  {'rank':<5} {'model':<32} {'mode':<5} {'correct/total':<14} {'acc':<8} "
          f"{'95% Wilson CI':<20} {'p':<12} {'Bonf-sig':<8}")
    print("  " + "-" * 110)
    for rank, r in enumerate(results, start=1):
        ci_str = f"[{r['ci_lo']:.3f}, {r['ci_hi']:.3f}]"
        sig = "**" if r["sig_bonf"] else ""
        print(f"  {rank:<5} {r['model']:<32} {r['mode']:<5} "
              f"{r['correct']}/{r['total']:<10} {r['acc']:.4f}   {ci_str:<20} "
              f"{r['p_value']:<12.4g} {sig:<8}")

    n_sig = sum(1 for r in results if r["sig_bonf"])
    print(f"\n  Summary: {n_sig}/{len(results)} model-modes Bonferroni-significantly > chance")

    # =================================================================
    print(f"\n\n{'=' * 80}")
    print("SECONDARY: per-(model, mode) accuracy split by SSA-cap vs SIA-cap (QT9)")
    print(f"{'=' * 80}")
    by_mm_qt: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for c in cap_cells:
        key = (c["model_short"], c["mode"], c["question_type"])
        by_mm_qt[key]["total"] += 1
        if c["correct_capability_answer"]:
            by_mm_qt[key]["correct"] += 1

    print(f"\n  {'model':<32} {'mode':<5} {'SSA-cap acc':<14} {'SIA-cap acc':<14} "
          f"{'SSA−SIA Δ':<10}")
    print("  " + "-" * 90)
    for r in results:  # iterate in same order as primary table
        m, mode = r["model"], r["mode"]
        ssa = by_mm_qt.get((m, mode, "ssa_capability"), {"correct": 0, "total": 0})
        sia = by_mm_qt.get((m, mode, "sia_capability"), {"correct": 0, "total": 0})
        ssa_acc = ssa["correct"] / ssa["total"] if ssa["total"] > 0 else 0
        sia_acc = sia["correct"] / sia["total"] if sia["total"] > 0 else 0
        delta = ssa_acc - sia_acc
        print(f"  {m:<32} {mode:<5} {ssa['correct']}/{ssa['total']} ({ssa_acc:.3f})  "
              f"{sia['correct']}/{sia['total']} ({sia_acc:.3f})  {delta:+.3f}")

    # =================================================================
    print(f"\n\n{'=' * 80}")
    print("SECONDARY: per-(model, mode) accuracy by problem class")
    print(f"{'=' * 80}")
    by_mm_pc: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for c in cap_cells:
        key = (c["model_short"], c["mode"], c["problem_class"])
        by_mm_pc[key]["total"] += 1
        if c["correct_capability_answer"]:
            by_mm_pc[key]["correct"] += 1

    classes = ["sb", "inc", "dd", "padd"]
    print(f"\n  {'model':<32} {'mode':<5} " + " ".join(f"{c.upper():<14}" for c in classes))
    print("  " + "-" * 110)
    for r in results:
        m, mode = r["model"], r["mode"]
        row = f"  {m:<32} {mode:<5} "
        for pc in classes:
            v = by_mm_pc.get((m, mode, pc), {"correct": 0, "total": 0})
            acc = v["correct"] / v["total"] if v["total"] > 0 else 0
            row += f"{v['correct']}/{v['total']} ({acc:.2f})  "
        print(row)

    # =================================================================
    print(f"\n\n{'=' * 80}")
    print("SECONDARY: per-(model, mode) accuracy by cluster")
    print(f"{'=' * 80}")
    by_mm_cl: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for c in cap_cells:
        key = (c["model_short"], c["mode"], c["cluster"])
        by_mm_cl[key]["total"] += 1
        if c["correct_capability_answer"]:
            by_mm_cl[key]["correct"] += 1

    print(f"\n  {'model':<32} {'mode':<5} {'SB-type cluster':<20} {'DD-type cluster':<20} "
          f"{'SB−DD Δ':<10}")
    print("  " + "-" * 100)
    for r in results:
        m, mode = r["model"], r["mode"]
        sb = by_mm_cl.get((m, mode, "SB-type"), {"correct": 0, "total": 0})
        dd = by_mm_cl.get((m, mode, "DD-type"), {"correct": 0, "total": 0})
        sb_acc = sb["correct"] / sb["total"] if sb["total"] > 0 else 0
        dd_acc = dd["correct"] / dd["total"] if dd["total"] > 0 else 0
        delta = sb_acc - dd_acc
        print(f"  {m:<32} {mode:<5} {sb['correct']}/{sb['total']} ({sb_acc:.3f})       "
              f"{dd['correct']}/{dd['total']} ({dd_acc:.3f})       {delta:+.3f}")

    # =================================================================
    print(f"\n\n{'=' * 80}")
    print("ADDITIONAL TESTS: SSA-cap vs SIA-cap, per-model and population-level")
    print(f"{'=' * 80}")
    # Chi-square per (model, mode) on SSA-cap vs SIA-cap correctness
    print(f"\n  Per-(model, mode) chi-square (SSA-cap vs SIA-cap accuracy):")
    print(f"  {'model':<32} {'mode':<5} {'SSA cor/wrong':<15} {'SIA cor/wrong':<15} "
          f"{'χ²':<7} {'p':<12} {'Bonf':<6}")
    print("  " + "-" * 100)
    alpha_bonf_12 = 0.05 / 12
    pop_ssa_correct = pop_ssa_wrong = pop_sia_correct = pop_sia_wrong = 0
    for r in results:
        m, mode = r["model"], r["mode"]
        ssa = by_mm_qt.get((m, mode, "ssa_capability"), {"correct": 0, "total": 0})
        sia = by_mm_qt.get((m, mode, "sia_capability"), {"correct": 0, "total": 0})
        ssa_w = ssa["total"] - ssa["correct"]
        sia_w = sia["total"] - sia["correct"]
        chi2, p = chi2_2x2(ssa["correct"], ssa_w, sia["correct"], sia_w)
        sig = "**" if p < alpha_bonf_12 else ("*" if p < 0.05 else "")
        print(f"  {m:<32} {mode:<5} {ssa['correct']}/{ssa_w:<10} "
              f"{sia['correct']}/{sia_w:<10} {chi2:<7.2f} {p:<12.4g} {sig:<6}")
        pop_ssa_correct += ssa["correct"]
        pop_ssa_wrong += ssa_w
        pop_sia_correct += sia["correct"]
        pop_sia_wrong += sia_w
    # Population-level
    chi2, p = chi2_2x2(pop_ssa_correct, pop_ssa_wrong, pop_sia_correct, pop_sia_wrong)
    ssa_pop_acc = pop_ssa_correct / (pop_ssa_correct + pop_ssa_wrong)
    sia_pop_acc = pop_sia_correct / (pop_sia_correct + pop_sia_wrong)
    print(f"\n  Population-level pooled across all model-modes:")
    print(f"    SSA-cap: {pop_ssa_correct}/{pop_ssa_correct + pop_ssa_wrong} ({ssa_pop_acc:.4f})")
    print(f"    SIA-cap: {pop_sia_correct}/{pop_sia_correct + pop_sia_wrong} ({sia_pop_acc:.4f})")
    print(f"    Δ (SSA - SIA): {ssa_pop_acc - sia_pop_acc:+.4f}")
    print(f"    χ² = {chi2:.2f}, p = {p:.4g}")

    # =================================================================
    print(f"\n\n{'=' * 80}")
    print("ADDITIONAL TESTS: SB-cluster vs DD-cluster, per-model and population-level")
    print(f"{'=' * 80}")
    print(f"\n  Per-(model, mode) chi-square (SB-cluster vs DD-cluster accuracy):")
    print(f"  {'model':<32} {'mode':<5} {'SB cor/wrong':<15} {'DD cor/wrong':<15} "
          f"{'χ²':<7} {'p':<12} {'Bonf':<6}")
    print("  " + "-" * 100)
    pop_sb_correct = pop_sb_wrong = pop_dd_correct = pop_dd_wrong = 0
    for r in results:
        m, mode = r["model"], r["mode"]
        sb = by_mm_cl.get((m, mode, "SB-type"), {"correct": 0, "total": 0})
        dd = by_mm_cl.get((m, mode, "DD-type"), {"correct": 0, "total": 0})
        sb_w = sb["total"] - sb["correct"]
        dd_w = dd["total"] - dd["correct"]
        chi2, p = chi2_2x2(sb["correct"], sb_w, dd["correct"], dd_w)
        sig = "**" if p < alpha_bonf_12 else ("*" if p < 0.05 else "")
        print(f"  {m:<32} {mode:<5} {sb['correct']}/{sb_w:<10} "
              f"{dd['correct']}/{dd_w:<10} {chi2:<7.2f} {p:<12.4g} {sig:<6}")
        pop_sb_correct += sb["correct"]
        pop_sb_wrong += sb_w
        pop_dd_correct += dd["correct"]
        pop_dd_wrong += dd_w
    chi2, p = chi2_2x2(pop_sb_correct, pop_sb_wrong, pop_dd_correct, pop_dd_wrong)
    sb_pop_acc = pop_sb_correct / (pop_sb_correct + pop_sb_wrong)
    dd_pop_acc = pop_dd_correct / (pop_dd_correct + pop_dd_wrong)
    print(f"\n  Population-level pooled across all model-modes:")
    print(f"    SB-cluster: {pop_sb_correct}/{pop_sb_correct + pop_sb_wrong} ({sb_pop_acc:.4f})")
    print(f"    DD-cluster: {pop_dd_correct}/{pop_dd_correct + pop_dd_wrong} ({dd_pop_acc:.4f})")
    print(f"    Δ (SB - DD): {sb_pop_acc - dd_pop_acc:+.4f}")
    print(f"    χ² = {chi2:.2f}, p = {p:.4g}")

    # =================================================================
    print(f"\n\n{'=' * 80}")
    print("ADDITIONAL TEST: pooled population-level capability accuracy vs chance (0.5)")
    print(f"{'=' * 80}")
    total_correct = sum(r["correct"] for r in results)
    total_n = sum(r["total"] for r in results)
    pop_acc = total_correct / total_n
    pop_p = binomial_one_sided_p(total_correct, total_n, 0.5)
    lo, hi = wilson_ci(total_correct, total_n)
    print(f"  Total capability cells with answer: {total_n}")
    print(f"  Total correct: {total_correct}")
    print(f"  Pooled accuracy: {pop_acc:.4f}")
    print(f"  95% Wilson CI: [{lo:.4f}, {hi:.4f}]")
    print(f"  One-sided p-value (vs 0.5): {pop_p:.4g}")

    # =================================================================
    print(f"\n\n{'=' * 80}")
    print("SENSITIVITY: strict_clean only vs all-parsed (parse_quality robustness)")
    print(f"{'=' * 80}")
    strict_cells = [c for c in cap_cells if c.get("parse_quality") == "strict_clean"]
    print(f"  All-parsed capability cells: {len(cap_cells)}")
    print(f"  strict_clean capability cells: {len(strict_cells)}")
    print(f"\n  {'model':<32} {'mode':<5} {'all acc':<12} {'strict acc':<14} {'Δ (strict − all)':<18}")
    print("  " + "-" * 90)
    by_mm_strict: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for c in strict_cells:
        key = (c["model_short"], c["mode"])
        by_mm_strict[key]["total"] += 1
        if c["correct_capability_answer"]:
            by_mm_strict[key]["correct"] += 1
    for r in results:
        m, mode = r["model"], r["mode"]
        st = by_mm_strict.get((m, mode), {"correct": 0, "total": 0})
        st_acc = st["correct"] / st["total"] if st["total"] > 0 else 0
        delta = st_acc - r["acc"]
        print(f"  {m:<32} {mode:<5} {r['acc']:.4f}        "
              f"{st['correct']}/{st['total']} ({st_acc:.3f})    {delta:+.4f}")

    # Statistical test: per (model, mode), is strict-only accuracy different from non-strict accuracy?
    # Compare strict-cell accuracy vs non-strict-cell accuracy via chi-square
    print(f"\n  Per-(model, mode) chi-square test: strict cells vs non-strict cells accuracy")
    print(f"  {'model':<32} {'mode':<5} {'strict cor/wrong':<17} {'non-strict cor/wrong':<22} "
          f"{'χ²':<6} {'p':<10} {'sig':<5}")
    print("  " + "-" * 110)
    by_mm_nonstrict: dict = defaultdict(lambda: {"correct": 0, "total": 0})
    for c in cap_cells:
        if c.get("parse_quality") != "strict_clean":
            key = (c["model_short"], c["mode"])
            by_mm_nonstrict[key]["total"] += 1
            if c["correct_capability_answer"]:
                by_mm_nonstrict[key]["correct"] += 1
    for r in results:
        m, mode = r["model"], r["mode"]
        st = by_mm_strict.get((m, mode), {"correct": 0, "total": 0})
        ns = by_mm_nonstrict.get((m, mode), {"correct": 0, "total": 0})
        if st["total"] == 0 or ns["total"] == 0:
            print(f"  {m:<32} {mode:<5} {st['correct']}/{st['total']-st['correct']:<13} "
                  f"{ns['correct']}/{ns['total']-ns['correct']:<18} (one side empty, skipped)")
            continue
        chi2, p = chi2_2x2(st["correct"], st["total"] - st["correct"],
                           ns["correct"], ns["total"] - ns["correct"])
        sig = "**" if p < alpha_bonf_12 else ("*" if p < 0.05 else "")
        print(f"  {m:<32} {mode:<5} {st['correct']}/{st['total']-st['correct']:<13} "
              f"{ns['correct']}/{ns['total']-ns['correct']:<18} {chi2:<6.2f} {p:<10.4g} {sig:<5}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
