#!/usr/bin/env python3
"""RQ3 deep-dive: cluster pluralism in attitude responses across all (model, mode).

Motivating question (philosophical): is a model "monistic" (applies the same
anthropic principle consistently across all problem clusters) or "pluralistic"
(applies different principles in different cluster contexts)?

Operationally: for each (model, mode), compare P(SSA-aligned) on SB-cluster
attitude responses vs DD-cluster attitude responses. If the rates are similar,
the model is monistic; if they differ significantly, the model is pluralistic
(letting the problem context determine which principle to apply).

For each (model, mode):
  - Compute P(SSA-aligned | SB-cluster, attitudes)
  - Compute P(SSA-aligned | DD-cluster, attitudes)
  - Δ = absolute difference
  - Chi-square test for significance

Classification:
  - "Monistic" if Δ < 10pp AND not Bonferroni-significant
  - "Pluralistic" if Δ ≥ 10pp AND Bonferroni-significant
  - "Mixed" otherwise

Population-level: pool all 12 (model, mode) attitudes, compare clusters.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

D = Path("experiment_results/main_run_20260516")
N_TESTS = 12  # 12 (model, mode) configurations
ALPHA_BONF = 0.05 / N_TESTS
PLURALISM_THRESHOLD = 0.10  # 10pp


def parse_mode(filename: str) -> str:
    if filename.endswith("_on.json"):
        return "on"
    if filename.endswith("_off.json"):
        return "off"
    return "?"


def parse_problem_class(template_name: str) -> str:
    m = re.match(r"\d+_standard_(sb|inc|dd|padd)_", template_name or "")
    return m.group(1) if m else "?"


def parse_cluster(pc: str) -> str:
    return "SB-type" if pc in ("sb", "inc") else ("DD-type" if pc in ("dd", "padd") else "?")


def get_ssa_aligned_letter(preferred_actions: dict, row_order: str) -> str | None:
    if not preferred_actions:
        return None
    ssa_pref = preferred_actions.get("ssa_preference")
    if not ssa_pref:
        return None
    is_A_in_row12 = ssa_pref in ("half", "high")
    if row_order == "12":
        return "A" if is_A_in_row12 else "B"
    else:
        return "B" if is_A_in_row12 else "A"


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


def load_attitude_cells():
    cells = []
    for f in sorted(D.glob("*.json")):
        d = json.load(open(f))
        qt = d.get("question_type") or ""
        if not qt.endswith("_attitude"):
            continue
        ch = d.get("extracted_choice")
        if ch not in ("A", "B"):
            continue
        d["mode"] = parse_mode(f.name)
        model = d.get("model_id_openrouter") or ""
        d["model_short"] = model.split("/")[-1]
        d["problem_class"] = parse_problem_class(d.get("template_name", ""))
        d["cluster"] = parse_cluster(d["problem_class"])
        ssa_letter = get_ssa_aligned_letter(d.get("preferred_actions"), d.get("row_order"))
        d["is_ssa_aligned"] = (ch == ssa_letter) if ssa_letter else None
        if d["is_ssa_aligned"] is not None and d["cluster"] in ("SB-type", "DD-type"):
            cells.append(d)
    return cells


def classify(delta_pp: float, p_value: float) -> str:
    is_large = abs(delta_pp) >= PLURALISM_THRESHOLD * 100
    is_sig = p_value < ALPHA_BONF
    if is_large and is_sig:
        return "PLURALISTIC"
    if abs(delta_pp) < 5 and not is_sig:
        return "monistic"
    return "mixed"


def main() -> int:
    cells = load_attitude_cells()
    print(f"Loaded {len(cells)} attitude cells with parsed choice and known SSA-letter mapping")

    # Per (model, mode), cluster-level SSA-aligned rate
    counts: dict = defaultdict(lambda: {"sb_ssa": 0, "sb_n": 0, "dd_ssa": 0, "dd_n": 0})
    for c in cells:
        key = (c["model_short"], c["mode"])
        if c["cluster"] == "SB-type":
            counts[key]["sb_n"] += 1
            if c["is_ssa_aligned"]:
                counts[key]["sb_ssa"] += 1
        else:
            counts[key]["dd_n"] += 1
            if c["is_ssa_aligned"]:
                counts[key]["dd_ssa"] += 1

    results = []
    for (m, mode), v in counts.items():
        sb_ssa, sb_n = v["sb_ssa"], v["sb_n"]
        dd_ssa, dd_n = v["dd_ssa"], v["dd_n"]
        sb_rate = sb_ssa / sb_n if sb_n > 0 else 0
        dd_rate = dd_ssa / dd_n if dd_n > 0 else 0
        delta = sb_rate - dd_rate
        chi2, p = chi2_2x2(sb_ssa, sb_n - sb_ssa, dd_ssa, dd_n - dd_ssa)
        cls = classify(delta * 100, p)
        results.append({
            "model": m, "mode": mode,
            "sb_ssa": sb_ssa, "sb_n": sb_n, "sb_rate": sb_rate,
            "dd_ssa": dd_ssa, "dd_n": dd_n, "dd_rate": dd_rate,
            "delta_pp": delta * 100, "p": p, "chi2": chi2,
            "classification": cls,
        })

    # Sort by absolute delta descending
    results.sort(key=lambda r: -abs(r["delta_pp"]))

    print(f"\n{'=' * 100}")
    print(f"Cluster pluralism per (model, mode) on attitudes")
    print(f"Threshold: |Δ| ≥ {PLURALISM_THRESHOLD*100:.0f}pp AND p < {ALPHA_BONF:.4f} (Bonferroni)")
    print(f"{'=' * 100}")
    print(f"\n  {'model':<32} {'mode':<5} {'SB SSA-rate':<14} {'DD SSA-rate':<14} "
          f"{'Δ (SB-DD)':<11} {'χ²':<7} {'p':<12} {'classification':<14}")
    print("  " + "-" * 130)
    for r in results:
        print(f"  {r['model']:<32} {r['mode']:<5} "
              f"{r['sb_ssa']}/{r['sb_n']} ({r['sb_rate']:.3f})  "
              f"{r['dd_ssa']}/{r['dd_n']} ({r['dd_rate']:.3f})  "
              f"{r['delta_pp']:+7.1f}pp   {r['chi2']:<7.2f} {r['p']:<12.4g} "
              f"{r['classification']:<14}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print(f"Classification summary across 12 (model, mode) configurations")
    print(f"{'=' * 100}")
    n_pluralistic = sum(1 for r in results if r["classification"] == "PLURALISTIC")
    n_monistic = sum(1 for r in results if r["classification"] == "monistic")
    n_mixed = sum(1 for r in results if r["classification"] == "mixed")
    print(f"  PLURALISTIC (|Δ| ≥ 10pp AND Bonferroni-sig):  {n_pluralistic}/12")
    print(f"  monistic    (|Δ| <  5pp AND not sig):         {n_monistic}/12")
    print(f"  mixed                                          {n_mixed}/12")

    print(f"\n  Pluralistic configurations (in order of Δ magnitude):")
    for r in results:
        if r["classification"] == "PLURALISTIC":
            direction = "SB > DD (more halfer on SB)" if r["delta_pp"] > 0 \
                else "DD > SB (more SSA on DD-cluster)"
            print(f"    {r['model']:<32} {r['mode']:<5} Δ = {r['delta_pp']:+.1f}pp — {direction}")

    print(f"\n  Monistic configurations:")
    for r in results:
        if r["classification"] == "monistic":
            print(f"    {r['model']:<32} {r['mode']:<5} Δ = {r['delta_pp']:+.1f}pp  "
                  f"(both clusters near {(r['sb_rate']+r['dd_rate'])/2:.2%})")

    # =================================================================
    print(f"\n{'=' * 100}")
    print(f"Reasoning-mode shift in pluralism (paired hybrid models: off → on)")
    print(f"{'=' * 100}")
    # For each hybrid model, compare |Δ| in off-mode vs on-mode
    by_model: dict = defaultdict(dict)
    for r in results:
        by_model[r["model"]][r["mode"]] = r
    print(f"\n  {'model':<32} {'off |Δ|':<12} {'on |Δ|':<12} {'reasoning-effect on pluralism':<35}")
    print("  " + "-" * 100)
    for model, modes in sorted(by_model.items()):
        if "on" not in modes or "off" not in modes:
            continue
        off_d = abs(modes["off"]["delta_pp"])
        on_d = abs(modes["on"]["delta_pp"])
        diff = on_d - off_d
        if diff > 5:
            effect = "INDUCES pluralism"
        elif diff < -5:
            effect = "REDUCES pluralism"
        else:
            effect = "no clear effect"
        print(f"  {model:<32} {off_d:.1f}pp        {on_d:.1f}pp        {effect}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print(f"Population-level pooled across all 12 (model, mode)")
    print(f"{'=' * 100}")
    pop_sb_ssa = sum(r["sb_ssa"] for r in results)
    pop_sb_n = sum(r["sb_n"] for r in results)
    pop_dd_ssa = sum(r["dd_ssa"] for r in results)
    pop_dd_n = sum(r["dd_n"] for r in results)
    pop_sb_rate = pop_sb_ssa / pop_sb_n
    pop_dd_rate = pop_dd_ssa / pop_dd_n
    chi2, p = chi2_2x2(pop_sb_ssa, pop_sb_n - pop_sb_ssa, pop_dd_ssa, pop_dd_n - pop_dd_ssa)
    print(f"  SB-cluster: {pop_sb_ssa}/{pop_sb_n} ({pop_sb_rate:.4f}) SSA-aligned")
    print(f"  DD-cluster: {pop_dd_ssa}/{pop_dd_n} ({pop_dd_rate:.4f}) SSA-aligned")
    print(f"  Δ (SB − DD): {(pop_sb_rate - pop_dd_rate)*100:+.2f}pp")
    print(f"  χ² = {chi2:.2f}, p = {p:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
