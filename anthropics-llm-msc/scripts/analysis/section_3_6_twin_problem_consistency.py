#!/usr/bin/env python3
"""RQ5 — Problem-family consistency: do models treat structural twins identically?

SB ↔ INC are structural twins (SB-type cluster, same formal SSA/SIA recommendations).
DD ↔ PADD are structural twins (DD-type cluster, same formal SSA/SIA recommendations).

Pre-registered hypothesis (two-sided):
  Within each twin pair, the population thirder rate (SIA-aligned letter choice)
  should not differ. Equivalently, models should treat SB and INC identically,
  and DD and PADD identically.

Tests:
  - Per (model, mode, q_type): chi-square test on P(thirder | SB) vs P(thirder | INC)
    and on P(thirder | DD) vs P(thirder | PADD).
  - 12 model-modes × 4 q-types × 2 twin-pairs = 96 tests.
  - Bonferroni α = 0.05 / 96 ≈ 0.00052.
  - Plus population-level pooled and per-q-type summaries.

Comparison: within-twin Δ should be small; between-cluster Δ (SB vs DD) we already
know is huge. RQ5 confirms whether the cluster boundary is sharp (large between,
small within) or whether problems are individually heterogeneous.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

D = Path("experiment_results/main_run_20260516")
N_TESTS = 96  # 12 × 4 × 2
ALPHA_BONF = 0.05 / N_TESTS  # ≈ 0.000521


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


def get_sia_aligned_letter(preferred_actions: dict, row_order: str) -> str | None:
    if not preferred_actions:
        return None
    sia_pref = preferred_actions.get("sia_preference")
    if not sia_pref:
        return None
    is_A_in_row12 = sia_pref in ("half", "high")
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
        ch = d.get("extracted_choice")
        if ch not in ("A", "B"):
            continue
        d["mode"] = parse_mode(f.name)
        model = d.get("model_id_openrouter") or ""
        d["model_short"] = model.split("/")[-1]
        d["problem_class"] = parse_problem_class(d.get("template_name", ""))
        d["cluster"] = parse_cluster(d["problem_class"])
        sia_letter = get_sia_aligned_letter(d.get("preferred_actions"), d.get("row_order"))
        d["is_thirder"] = (ch == sia_letter) if sia_letter else None
        if d["is_thirder"] is not None and d["problem_class"] in ("sb", "inc", "dd", "padd"):
            cells.append(d)
    return cells


def main() -> int:
    cells = load_cells()
    print(f"Loaded {len(cells)} cells with parsed choice and known SIA-letter mapping")

    # Aggregate: (model, mode, q_type, problem_class) → {thirder, total}
    counts: dict = defaultdict(lambda: {"thirder": 0, "total": 0})
    for c in cells:
        key = (c["model_short"], c["mode"], c.get("question_type"), c["problem_class"])
        counts[key]["total"] += 1
        if c["is_thirder"]:
            counts[key]["thirder"] += 1

    # =================================================================
    print(f"\n{'=' * 100}")
    print("PRIMARY: per (model, mode, q_type) within-twin difference tests")
    print(f"Bonferroni α = {ALPHA_BONF:.6f} (96 tests)")
    print(f"{'=' * 100}")
    print(f"\n  {'model':<32} {'mode':<5} {'q-type':<22} {'pair':<11} "
          f"{'P(thirder|A)':<13} {'P(thirder|B)':<13} {'|Δ|':<7} {'p':<12} {'Bonf-sig':<8}")
    print("  " + "-" * 130)

    twin_pairs = [("sb", "inc", "SB↔INC"), ("dd", "padd", "DD↔PADD")]
    n_sig_sb_pair = 0
    n_sig_dd_pair = 0
    results = []
    # Order: by model, mode, q-type, then pair
    keys_mmq = sorted({(k[0], k[1], k[2]) for k in counts.keys()})
    for (m, mode, qt) in keys_mmq:
        for a, b, label in twin_pairs:
            va = counts.get((m, mode, qt, a), {"thirder": 0, "total": 0})
            vb = counts.get((m, mode, qt, b), {"thirder": 0, "total": 0})
            if va["total"] == 0 or vb["total"] == 0:
                continue
            pa = va["thirder"] / va["total"]
            pb = vb["thirder"] / vb["total"]
            chi2, p = chi2_2x2(va["thirder"], va["total"] - va["thirder"],
                                vb["thirder"], vb["total"] - vb["thirder"])
            sig = "**" if p < ALPHA_BONF else ("*" if p < 0.05 else "")
            if p < ALPHA_BONF:
                if label == "SB↔INC":
                    n_sig_sb_pair += 1
                else:
                    n_sig_dd_pair += 1
            results.append({
                "model": m, "mode": mode, "q_type": qt, "pair": label,
                "pa": pa, "pb": pb, "delta_pp": abs(pa - pb) * 100,
                "p": p, "sig": p < ALPHA_BONF
            })
            print(f"  {m:<32} {mode:<5} {qt:<22} {label:<11} "
                  f"{pa:.3f} (n={va['total']})  {pb:.3f} (n={vb['total']})  "
                  f"{abs(pa-pb)*100:5.1f}pp {p:<12.4g} {sig:<8}")

    print(f"\n  Within-twin tests at Bonferroni:")
    print(f"    SB↔INC pair: {n_sig_sb_pair}/48 Bonferroni-significant differences")
    print(f"    DD↔PADD pair: {n_sig_dd_pair}/48 Bonferroni-significant differences")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("Average within-twin |Δ| per q-type — small Δ = twins treated identically")
    print(f"{'=' * 100}")
    by_qt: dict = defaultdict(lambda: {"sb_inc": [], "dd_padd": []})
    for r in results:
        if r["pair"] == "SB↔INC":
            by_qt[r["q_type"]]["sb_inc"].append(r["delta_pp"])
        else:
            by_qt[r["q_type"]]["dd_padd"].append(r["delta_pp"])
    print(f"\n  {'q-type':<25} {'avg |SB↔INC Δ|':<18} {'avg |DD↔PADD Δ|':<18}")
    print("  " + "-" * 70)
    for qt in sorted(by_qt):
        sb_avg = sum(by_qt[qt]["sb_inc"]) / len(by_qt[qt]["sb_inc"]) if by_qt[qt]["sb_inc"] else 0
        dd_avg = sum(by_qt[qt]["dd_padd"]) / len(by_qt[qt]["dd_padd"]) if by_qt[qt]["dd_padd"] else 0
        print(f"  {qt:<25} {sb_avg:<18.2f} {dd_avg:<18.2f}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("Population-level pooled: SB vs INC and DD vs PADD per q-type")
    print(f"{'=' * 100}")
    pop_by_qpc: dict = defaultdict(lambda: {"thirder": 0, "total": 0})
    for c in cells:
        key = (c.get("question_type"), c["problem_class"])
        pop_by_qpc[key]["total"] += 1
        if c["is_thirder"]:
            pop_by_qpc[key]["thirder"] += 1

    print(f"\n  {'q-type':<25} {'pair':<11} {'P(thirder|A)':<16} {'P(thirder|B)':<16} {'|Δ|':<7} {'p':<12}")
    print("  " + "-" * 95)
    for qt in sorted({k[0] for k in pop_by_qpc.keys()}):
        for a, b, label in twin_pairs:
            va = pop_by_qpc.get((qt, a), {"thirder": 0, "total": 0})
            vb = pop_by_qpc.get((qt, b), {"thirder": 0, "total": 0})
            pa = va["thirder"] / va["total"]
            pb = vb["thirder"] / vb["total"]
            chi2, p = chi2_2x2(va["thirder"], va["total"] - va["thirder"],
                                vb["thirder"], vb["total"] - vb["thirder"])
            print(f"  {qt:<25} {label:<11} {va['thirder']}/{va['total']} ({pa:.3f})   "
                  f"{vb['thirder']}/{vb['total']} ({pb:.3f})   {abs(pa-pb)*100:5.1f}pp {p:<12.4g}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("Comparison: within-twin Δ vs between-cluster Δ (sharpness of cluster boundary)")
    print(f"{'=' * 100}")
    print(f"\n  Between-cluster Δ = |P(thirder | SB-cluster) − P(thirder | DD-cluster)|")
    print(f"  Within-twin Δ = |P(thirder | SB) − P(thirder | INC)| or |P(thirder | DD) − P(thirder | PADD)|")
    print(f"\n  At population level per q-type:")
    print(f"  {'q-type':<25} {'within SB↔INC Δ':<18} {'within DD↔PADD Δ':<18} "
          f"{'between cluster Δ':<20} {'ratio (between/within)':<22}")
    print("  " + "-" * 110)
    for qt in sorted({k[0] for k in pop_by_qpc.keys()}):
        sb = pop_by_qpc.get((qt, "sb"), {"thirder": 0, "total": 0})
        inc = pop_by_qpc.get((qt, "inc"), {"thirder": 0, "total": 0})
        dd = pop_by_qpc.get((qt, "dd"), {"thirder": 0, "total": 0})
        padd = pop_by_qpc.get((qt, "padd"), {"thirder": 0, "total": 0})
        sb_r = sb["thirder"] / sb["total"] if sb["total"] > 0 else 0
        inc_r = inc["thirder"] / inc["total"] if inc["total"] > 0 else 0
        dd_r = dd["thirder"] / dd["total"] if dd["total"] > 0 else 0
        padd_r = padd["thirder"] / padd["total"] if padd["total"] > 0 else 0
        within_sb = abs(sb_r - inc_r) * 100
        within_dd = abs(dd_r - padd_r) * 100
        sb_cluster_r = (sb["thirder"] + inc["thirder"]) / (sb["total"] + inc["total"])
        dd_cluster_r = (dd["thirder"] + padd["thirder"]) / (dd["total"] + padd["total"])
        between = abs(sb_cluster_r - dd_cluster_r) * 100
        avg_within = (within_sb + within_dd) / 2
        ratio = between / avg_within if avg_within > 0 else float("inf")
        print(f"  {qt:<25} {within_sb:<18.2f} {within_dd:<18.2f} "
              f"{between:<20.2f} {ratio:<22.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
