#!/usr/bin/env python3
"""RQ8 — Off-menu / non-extracted-choice rate.

Definitions:
  - "Off-menu" cell = parse_quality in {empty_response, no_final_answer, off_menu_refusal}.
    These are the cells where extracted_choice is None.
  - Sub-categories:
      * empty_response: model returned no response text (often reasoning burnout)
      * no_final_answer: response present but no FINAL ANSWER format
      * off_menu_refusal: explicit "Neither / N/A / None" — substantive refusal

Pre-registered primary hypothesis for RQ8 (descriptive, exploratory):
  Quantify off-menu rate overall and by (model, mode), q-type, problem class, cluster.

Secondary hypothesis tests (also exploratory):
  - PADD shows higher off-menu rate than DD (sub-hypothesis from RQ7 novelty).
  - SSA-cap vs SIA-cap off-menu rate asymmetry (QT9 anchored here).
  - Attitude vs capability off-menu rate asymmetry.

Tests: chi-square / Fisher's exact for 2-way contingencies.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

D = Path("experiment_results/main_run_20260516")
OFF_MENU_QUALITIES = {"empty_response", "no_final_answer", "off_menu_refusal"}


def parse_mode(filename: str) -> str:
    if filename.endswith("_on.json"):
        return "on"
    if filename.endswith("_off.json"):
        return "off"
    return "?"


def parse_problem_class(template_name: str) -> str:
    """Extract the problem class (sb / inc / dd / padd) from template name."""
    m = re.match(r"\d+_standard_(sb|inc|dd|padd)_", template_name or "")
    return m.group(1) if m else "?"


def parse_cluster(problem_class: str) -> str:
    """Map problem class to cluster."""
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
        d["is_off_menu"] = d.get("parse_quality") in OFF_MENU_QUALITIES
        cells.append(d)
    return cells


def chi2_2x2(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """Chi-square test of independence on a 2×2 table.

    Table:
                Cat1 | Cat2
        Row1:    a   |  b
        Row2:    c   |  d

    Returns (chi2_statistic, p_value).
    """
    n = a + b + c + d
    if n == 0:
        return (0.0, 1.0)
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    # Expected counts
    e_a = row1 * col1 / n
    e_b = row1 * col2 / n
    e_c = row2 * col1 / n
    e_d = row2 * col2 / n
    # Chi-square statistic
    chi2 = 0.0
    for obs, exp in zip([a, b, c, d], [e_a, e_b, e_c, e_d]):
        if exp > 0:
            chi2 += (obs - exp) ** 2 / exp
    # p-value from chi-square distribution with df=1
    # Using survival function approximation: P(X > chi2) for df=1 is erfc(sqrt(chi2/2))
    p_value = math.erfc(math.sqrt(chi2 / 2))
    return (chi2, p_value)


def main() -> int:
    cells = load_dataset()
    n_total = len(cells)
    n_off = sum(1 for c in cells if c["is_off_menu"])
    print(f"Loaded {n_total} cells, of which {n_off} are off-menu ({100*n_off/n_total:.2f}%)")

    # =================================================================
    print(f"\n{'=' * 80}")
    print("SECTION 1: Overall off-menu breakdown by parse_quality category")
    print(f"{'=' * 80}")
    quality_counts: dict = defaultdict(int)
    for c in cells:
        quality_counts[c.get("parse_quality")] += 1
    for q in sorted(quality_counts):
        n = quality_counts[q]
        marker = " (off-menu)" if q in OFF_MENU_QUALITIES else ""
        print(f"  {q:<32} {n:>6}  ({100*n/n_total:6.2f}%){marker}")

    # =================================================================
    print(f"\n{'=' * 80}")
    print("SECTION 2: Off-menu rate by (model, mode)")
    print(f"{'=' * 80}")
    by_mm: dict = defaultdict(lambda: {"total": 0, "off": 0})
    for c in cells:
        key = (c["model_short"], c["mode"])
        by_mm[key]["total"] += 1
        if c["is_off_menu"]:
            by_mm[key]["off"] += 1
    print(f"  {'model':<32} {'mode':<5} {'off':<6} {'total':<6} {'%':<8}")
    print("  " + "-" * 60)
    for (m, mode), v in sorted(by_mm.items(), key=lambda x: -x[1]["off"]):
        pct = 100 * v["off"] / v["total"]
        print(f"  {m:<32} {mode:<5} {v['off']:<6} {v['total']:<6} {pct:.2f}%")

    # =================================================================
    print(f"\n{'=' * 80}")
    print("SECTION 3: Off-menu rate by question type")
    print(f"{'=' * 80}")
    by_qt: dict = defaultdict(lambda: {"total": 0, "off": 0})
    for c in cells:
        qt = c.get("question_type", "?")
        by_qt[qt]["total"] += 1
        if c["is_off_menu"]:
            by_qt[qt]["off"] += 1
    for qt, v in sorted(by_qt.items()):
        pct = 100 * v["off"] / v["total"]
        print(f"  {qt:<25} {v['off']:>4} / {v['total']:<5} ({pct:.2f}%)")

    # Test: attitude vs capability off-menu rate
    att_off = sum(v["off"] for qt, v in by_qt.items() if "attitude" in qt)
    att_tot = sum(v["total"] for qt, v in by_qt.items() if "attitude" in qt)
    cap_off = sum(v["off"] for qt, v in by_qt.items() if "capability" in qt)
    cap_tot = sum(v["total"] for qt, v in by_qt.items() if "capability" in qt)
    chi2, p = chi2_2x2(att_off, att_tot - att_off, cap_off, cap_tot - cap_off)
    print(f"\n  Attitude:   {att_off}/{att_tot} ({100*att_off/att_tot:.2f}%)")
    print(f"  Capability: {cap_off}/{cap_tot} ({100*cap_off/cap_tot:.2f}%)")
    print(f"  Chi-square test (attitude vs capability): χ² = {chi2:.3f}, p = {p:.5f}")

    # Test: SSA-cap vs SIA-cap (QT9 anchored here)
    ssa_v = by_qt.get("ssa_capability", {"total": 0, "off": 0})
    sia_v = by_qt.get("sia_capability", {"total": 0, "off": 0})
    chi2, p = chi2_2x2(ssa_v["off"], ssa_v["total"] - ssa_v["off"],
                       sia_v["off"], sia_v["total"] - sia_v["off"])
    print(f"  SSA-cap:    {ssa_v['off']}/{ssa_v['total']} ({100*ssa_v['off']/ssa_v['total']:.2f}%)")
    print(f"  SIA-cap:    {sia_v['off']}/{sia_v['total']} ({100*sia_v['off']/sia_v['total']:.2f}%)")
    print(f"  Chi-square test (SSA-cap vs SIA-cap): χ² = {chi2:.3f}, p = {p:.5f}")

    # =================================================================
    print(f"\n{'=' * 80}")
    print("SECTION 4: Off-menu rate by problem class")
    print(f"{'=' * 80}")
    by_pc: dict = defaultdict(lambda: {"total": 0, "off": 0})
    for c in cells:
        by_pc[c["problem_class"]]["total"] += 1
        if c["is_off_menu"]:
            by_pc[c["problem_class"]]["off"] += 1
    for pc, v in sorted(by_pc.items()):
        pct = 100 * v["off"] / v["total"]
        print(f"  {pc:<8} {v['off']:>4} / {v['total']:<5} ({pct:.2f}%)")

    # Test: PADD vs DD (RQ7 sub-hypothesis)
    dd_v = by_pc.get("dd", {"total": 0, "off": 0})
    padd_v = by_pc.get("padd", {"total": 0, "off": 0})
    chi2, p = chi2_2x2(padd_v["off"], padd_v["total"] - padd_v["off"],
                       dd_v["off"], dd_v["total"] - dd_v["off"])
    print(f"\n  PADD vs DD off-menu rate test:")
    print(f"    PADD: {padd_v['off']}/{padd_v['total']} ({100*padd_v['off']/padd_v['total']:.2f}%)")
    print(f"    DD:   {dd_v['off']}/{dd_v['total']} ({100*dd_v['off']/dd_v['total']:.2f}%)")
    print(f"    Chi-square: χ² = {chi2:.3f}, p = {p:.5f}")

    # Test: SB vs INC
    sb_v = by_pc.get("sb", {"total": 0, "off": 0})
    inc_v = by_pc.get("inc", {"total": 0, "off": 0})
    chi2, p = chi2_2x2(sb_v["off"], sb_v["total"] - sb_v["off"],
                       inc_v["off"], inc_v["total"] - inc_v["off"])
    print(f"\n  SB vs INC off-menu rate test:")
    print(f"    SB:  {sb_v['off']}/{sb_v['total']} ({100*sb_v['off']/sb_v['total']:.2f}%)")
    print(f"    INC: {inc_v['off']}/{inc_v['total']} ({100*inc_v['off']/inc_v['total']:.2f}%)")
    print(f"    Chi-square: χ² = {chi2:.3f}, p = {p:.5f}")

    # =================================================================
    print(f"\n{'=' * 80}")
    print("SECTION 5: Off-menu rate by cluster")
    print(f"{'=' * 80}")
    by_cluster: dict = defaultdict(lambda: {"total": 0, "off": 0})
    for c in cells:
        by_cluster[c["cluster"]]["total"] += 1
        if c["is_off_menu"]:
            by_cluster[c["cluster"]]["off"] += 1
    for cl, v in sorted(by_cluster.items()):
        pct = 100 * v["off"] / v["total"]
        print(f"  {cl:<10} {v['off']:>4} / {v['total']:<5} ({pct:.2f}%)")

    # =================================================================
    print(f"\n{'=' * 80}")
    print("SECTION 6: Off-menu sub-category breakdown by (model, mode)")
    print(f"{'=' * 80}")
    by_mm_sub: dict = defaultdict(lambda: defaultdict(int))
    for c in cells:
        if c["is_off_menu"]:
            by_mm_sub[(c["model_short"], c["mode"])][c.get("parse_quality")] += 1
    print(f"  {'model':<32} {'mode':<5} {'empty':<7} {'no_FA':<7} {'refusal':<8} {'total':<6}")
    print("  " + "-" * 70)
    for (m, mode), subs in sorted(by_mm_sub.items(), key=lambda x: -sum(x[1].values())):
        e = subs.get("empty_response", 0)
        n = subs.get("no_final_answer", 0)
        r = subs.get("off_menu_refusal", 0)
        print(f"  {m:<32} {mode:<5} {e:<7} {n:<7} {r:<8} {e+n+r:<6}")

    # =================================================================
    print(f"\n{'=' * 80}")
    print("SECTION 7: Summary headline numbers")
    print(f"{'=' * 80}")
    print(f"  Overall off-menu rate: {n_off}/{n_total} = {100*n_off/n_total:.3f}%")
    print(f"  Cells with parsed choice: {n_total - n_off} = {100*(n_total-n_off)/n_total:.3f}%")
    print(f"\n  Off-menu breakdown by sub-category:")
    for q in OFF_MENU_QUALITIES:
        c_q = quality_counts.get(q, 0)
        print(f"    {q:<25} {c_q:>4} ({100*c_q/n_total:.3f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
