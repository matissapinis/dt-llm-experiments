#!/usr/bin/env python3
"""RQ3 deep-dive: is pluralism (cluster-dependent SSA-aligned rate) present
in capability responses, or only in attitude responses?

Attitudes deep-dive showed: 11/12 (model, mode) configurations are PLURALISTIC
on attitudes (DD-cluster more SSA-aligned than SB-cluster). Population-level
Δ = -66.65pp.

Question: does the same pattern hold on capability questions? If yes, then
even when explicitly asked about SSA or SIA, models apply the principles
inconsistently across clusters (literature-recognition runs deep). If no,
then models can apply principles consistently when explicitly asked but
default to literature-recognition only on free-choice attitude questions.

For each (model, mode, q_type), measure:
  - P(SSA-aligned answer | SB-cluster)
  - P(SSA-aligned answer | DD-cluster)
  - Δ = SB-rate - DD-rate

Compare attitude pluralism vs capability pluralism per-model and population-level.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

D = Path("experiment_results/main_run_20260516")


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


def load_cells():
    cells = []
    for f in sorted(D.glob("*.json")):
        d = json.load(open(f))
        qt = d.get("question_type") or ""
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


def cluster_pluralism_per_mmq(cells: list[dict]) -> dict:
    """For each (model, mode, q_type), compute SSA-aligned rate per cluster + chi-square test."""
    counts: dict = defaultdict(lambda: {"sb_ssa": 0, "sb_n": 0, "dd_ssa": 0, "dd_n": 0})
    for c in cells:
        key = (c["model_short"], c["mode"], c["question_type"])
        if c["cluster"] == "SB-type":
            counts[key]["sb_n"] += 1
            if c["is_ssa_aligned"]:
                counts[key]["sb_ssa"] += 1
        else:
            counts[key]["dd_n"] += 1
            if c["is_ssa_aligned"]:
                counts[key]["dd_ssa"] += 1
    results = []
    for (m, mode, qt), v in counts.items():
        sb_rate = v["sb_ssa"] / v["sb_n"] if v["sb_n"] > 0 else 0
        dd_rate = v["dd_ssa"] / v["dd_n"] if v["dd_n"] > 0 else 0
        chi2, p = chi2_2x2(v["sb_ssa"], v["sb_n"] - v["sb_ssa"],
                           v["dd_ssa"], v["dd_n"] - v["dd_ssa"])
        results.append({
            "model": m, "mode": mode, "q_type": qt,
            "sb_ssa": v["sb_ssa"], "sb_n": v["sb_n"], "sb_rate": sb_rate,
            "dd_ssa": v["dd_ssa"], "dd_n": v["dd_n"], "dd_rate": dd_rate,
            "delta_pp": (sb_rate - dd_rate) * 100,
            "p": p, "chi2": chi2,
        })
    return results


def main() -> int:
    cells = load_cells()
    print(f"Loaded {len(cells)} cells with parsed choice and known SSA-letter mapping")

    results = cluster_pluralism_per_mmq(cells)

    # =================================================================
    print(f"\n{'=' * 110}")
    print(f"Per-(model, mode, q_type) cluster pluralism on SSA-aligned rate")
    print(f"Bonferroni α (48 tests = 12 model-modes × 4 q-types): {0.05/48:.5f}")
    print(f"{'=' * 110}")
    # Sort by q-type, then by |Δ| descending
    results.sort(key=lambda r: (r["q_type"], -abs(r["delta_pp"])))
    print(f"\n  {'model':<32} {'mode':<5} {'q-type':<22} "
          f"{'SB rate':<8} {'DD rate':<8} {'Δ (pp)':<8} {'p':<12} {'sig':<5}")
    print("  " + "-" * 110)
    alpha_bonf = 0.05 / 48
    current_qt = None
    for r in results:
        if r["q_type"] != current_qt:
            print()
            current_qt = r["q_type"]
        sig = "**" if r["p"] < alpha_bonf else ("*" if r["p"] < 0.05 else "")
        print(f"  {r['model']:<32} {r['mode']:<5} {r['q_type']:<22} "
              f"{r['sb_rate']:.3f}   {r['dd_rate']:.3f}   "
              f"{r['delta_pp']:+7.1f}  {r['p']:<12.4g} {sig:<5}")

    # =================================================================
    print(f"\n{'=' * 110}")
    print(f"Average |Δ| per q-type — overall pluralism magnitude by question type")
    print(f"{'=' * 110}")
    by_qt: dict = defaultdict(list)
    for r in results:
        by_qt[r["q_type"]].append(abs(r["delta_pp"]))
    print(f"\n  {'q-type':<25} {'avg |Δ| (pp)':<14} {'max |Δ| (pp)':<14} {'min |Δ| (pp)':<14}")
    print("  " + "-" * 75)
    for qt, deltas in sorted(by_qt.items()):
        avg_d = sum(deltas) / len(deltas)
        print(f"  {qt:<25} {avg_d:<14.2f} {max(deltas):<14.2f} {min(deltas):<14.2f}")

    # =================================================================
    print(f"\n{'=' * 110}")
    print(f"Population-level cluster comparison per q-type")
    print(f"{'=' * 110}")
    pop_by_qt: dict = defaultdict(lambda: {"sb_ssa": 0, "sb_n": 0, "dd_ssa": 0, "dd_n": 0})
    for c in cells:
        qt = c["question_type"]
        if c["cluster"] == "SB-type":
            pop_by_qt[qt]["sb_n"] += 1
            if c["is_ssa_aligned"]:
                pop_by_qt[qt]["sb_ssa"] += 1
        else:
            pop_by_qt[qt]["dd_n"] += 1
            if c["is_ssa_aligned"]:
                pop_by_qt[qt]["dd_ssa"] += 1
    print(f"\n  {'q-type':<25} {'SB rate':<14} {'DD rate':<14} {'Δ (pp)':<10} {'χ²':<8} {'p':<12}")
    print("  " + "-" * 90)
    for qt, v in sorted(pop_by_qt.items()):
        sb_rate = v["sb_ssa"] / v["sb_n"]
        dd_rate = v["dd_ssa"] / v["dd_n"]
        chi2, p = chi2_2x2(v["sb_ssa"], v["sb_n"] - v["sb_ssa"],
                           v["dd_ssa"], v["dd_n"] - v["dd_ssa"])
        print(f"  {qt:<25} {v['sb_ssa']}/{v['sb_n']} ({sb_rate:.3f})  "
              f"{v['dd_ssa']}/{v['dd_n']} ({dd_rate:.3f})  "
              f"{(sb_rate-dd_rate)*100:+7.1f}  {chi2:<8.2f} {p:<12.4g}")

    # =================================================================
    print(f"\n{'=' * 110}")
    print(f"Synthesis: capability pluralism vs attitude pluralism per (model, mode)")
    print(f"{'=' * 110}")
    # For each model-mode, average |Δ| across the 2 capability q-types and 2 attitude q-types
    by_mm_qttype: dict = defaultdict(lambda: {"cap_deltas": [], "att_deltas": []})
    for r in results:
        m_mode = (r["model"], r["mode"])
        if r["q_type"].endswith("_capability"):
            by_mm_qttype[m_mode]["cap_deltas"].append(abs(r["delta_pp"]))
        elif r["q_type"].endswith("_attitude"):
            by_mm_qttype[m_mode]["att_deltas"].append(abs(r["delta_pp"]))
    print(f"\n  {'model':<32} {'mode':<5} {'capability |Δ|':<18} {'attitude |Δ|':<18} {'ratio cap/att':<14}")
    print("  " + "-" * 100)
    for (m, mode) in sorted(by_mm_qttype):
        v = by_mm_qttype[(m, mode)]
        cap_avg = sum(v["cap_deltas"]) / len(v["cap_deltas"]) if v["cap_deltas"] else 0
        att_avg = sum(v["att_deltas"]) / len(v["att_deltas"]) if v["att_deltas"] else 0
        ratio = cap_avg / att_avg if att_avg > 0 else float('inf')
        print(f"  {m:<32} {mode:<5} {cap_avg:<18.2f} {att_avg:<18.2f} {ratio:<14.3f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
