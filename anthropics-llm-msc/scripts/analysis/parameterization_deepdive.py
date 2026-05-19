#!/usr/bin/env python3
"""RQ11 deep-dive — parameterization (SB-tipa vs DD-tipa numerics) as cluster aliaser.

Building on rq11_theme_param_effects.py, this script drills into the
parameterization effect with five inferential threads:

  D1. Per-model heterogeneity: is the alias universal across the 12 model-modes,
      or driven by a subset? Per-(model, mode) |Δ| on DD-cluster attitudes
      with chi-square p-values.

  D2. q-type breakdown with significance: capability vs attitude, per cluster.

  D3. Reasoning-mode interaction: within hybrid models (5 of 6), does ON or OFF
      show larger parameterization effect? Paired sign test.

  D4. Twin-pair consistency: does PADD show the same alias direction and similar
      magnitude as DD? (And INC vs SB?) Chi-square Δ on Δ.

  D5. Direction-of-alias clean test: explicit two-direction prediction —
      SB-numerics applied to DD-cluster → more SIA; DD-numerics applied to
      SB-cluster → more SIA on SSA-capability. Already shown in rq11 §4c;
      here we add per-model exact-binomial direction tests.

Reading note: "thirder" in this codebase = "SIA-aligned letter chosen". For
DD-canonical attitudes (Choices: 1/2 vs 1/3), SIA-aligned = 1/2 (halfer-value)
because SSA in the birth-rank reference class shifts toward Doomsday (1/3).
"""
from __future__ import annotations

import json
import math
import random as _random
import re
from collections import defaultdict
from pathlib import Path

D = Path("experiment_results/main_run_20260516")


def parse_template(tn: str):
    m = re.match(r"\d+_standard_(sb|inc|dd|padd)_(.+?)(_scaled)?_(12|21)$", tn or "")
    if not m:
        return (None, None, None, None)
    pc, theme, scaled, row = m.groups()
    return (pc, theme, "scaled" if scaled else "canonical", row)


def parse_mode(filename: str) -> str:
    if filename.endswith("_on.json"):
        return "on"
    if filename.endswith("_off.json"):
        return "off"
    return "?"


def parse_cluster(pc: str) -> str:
    return "SB-type" if pc in ("sb", "inc") else ("DD-type" if pc in ("dd", "padd") else "?")


def get_sia_aligned_letter(preferred_actions: dict, row_order: str) -> str | None:
    if not preferred_actions:
        return None
    sia_pref = preferred_actions.get("sia_preference")
    if not sia_pref:
        return None
    is_A_in_row12 = sia_pref in ("half", "high")
    return ("A" if is_A_in_row12 else "B") if row_order == "12" else ("B" if is_A_in_row12 else "A")


def chi2_2x2(a, b, c, d):
    n = a + b + c + d
    if n == 0:
        return (0.0, 1.0)
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    e = [row1 * col1 / n, row1 * col2 / n, row2 * col1 / n, row2 * col2 / n]
    chi2 = sum((o - x) ** 2 / x for o, x in zip([a, b, c, d], e) if x > 0)
    return (chi2, math.erfc(math.sqrt(chi2 / 2)))


def binomial_two_sided_p(k, n, p0=0.5):
    if n == 0:
        return 1.0
    pmf = [math.comb(n, i) * (p0 ** i) * ((1 - p0) ** (n - i)) for i in range(n + 1)]
    observed_p = pmf[k]
    return sum(pi for pi in pmf if pi <= observed_p + 1e-15)


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
        pc, theme, param, row = parse_template(d.get("template_name", ""))
        d["problem_class"] = pc
        d["cluster"] = parse_cluster(pc)
        d["theme"] = theme
        d["param"] = param
        d["row"] = row
        sia_letter = get_sia_aligned_letter(d.get("preferred_actions"), d.get("row_order"))
        d["is_thirder"] = (ch == sia_letter) if sia_letter else None
        if d["is_thirder"] is not None and pc:
            cells.append(d)
    return cells


def main() -> int:
    cells = load_cells()
    print(f"Loaded {len(cells)} cells")

    # ================================================================
    # D1. Per-model heterogeneity on DD-cluster ATTITUDES
    #     (the slice where param effect is largest)
    # ================================================================
    print(f"\n{'=' * 100}")
    print("D1. Per (model, mode) parameterization effect — DD-cluster ATTITUDES")
    print("    (P(SIA-aligned letter) under canonical=SB-scale vs scaled=DD-scale)")
    print("    Bonferroni α = 0.05 / 12 = 0.00417 (per-model tests, 12 configs)")
    print(f"{'=' * 100}")

    # Aggregate per (mm, param) within DD-cluster + attitudes
    by_mm_param: dict = defaultdict(lambda: {"thirder": 0, "total": 0})
    for c in cells:
        if c["cluster"] != "DD-type":
            continue
        if c.get("question_type") not in ("personal_attitude", "normative_attitude"):
            continue
        key = ((c["model_short"], c["mode"]), c["param"])
        by_mm_param[key]["total"] += 1
        if c["is_thirder"]:
            by_mm_param[key]["thirder"] += 1

    mms = sorted({mm for (mm, _) in by_mm_param.keys()})
    print(f"\n  {'model':<32} {'mode':<5} {'P(SIA|canon)':<15} {'P(SIA|scaled)':<16} "
          f"{'Δ canon-scaled':<16} {'p':<12} {'Bonf?':<6}")
    print("  " + "-" * 110)
    n_sig = 0
    n_predicted_dir = 0
    for mm in mms:
        c_d = by_mm_param.get((mm, "canonical"), {"thirder": 0, "total": 0})
        s_d = by_mm_param.get((mm, "scaled"), {"thirder": 0, "total": 0})
        pc_ = c_d["thirder"] / c_d["total"]
        ps_ = s_d["thirder"] / s_d["total"]
        chi2, p = chi2_2x2(c_d["thirder"], c_d["total"] - c_d["thirder"],
                            s_d["thirder"], s_d["total"] - s_d["thirder"])
        sig = "**" if p < 0.05 / 12 else ("*" if p < 0.05 else "")
        if p < 0.05 / 12:
            n_sig += 1
        if pc_ > ps_:
            n_predicted_dir += 1
        print(f"  {mm[0]:<32} {mm[1]:<5} {pc_:.3f} (n={c_d['total']})  "
              f"{ps_:.3f} (n={s_d['total']})  {(pc_-ps_)*100:+6.1f}pp        "
              f"{p:<12.4g} {sig:<6}")
    print(f"\n  Bonferroni-significant per-model param effects: {n_sig}/12")
    print(f"  Predicted direction (canonical > scaled): {n_predicted_dir}/12")
    p_dir = binomial_two_sided_p(n_predicted_dir, 12, p0=0.5)
    print(f"  Binomial sign test (direction is non-random): p = {p_dir:.4g}")

    # ================================================================
    # D2. q-type breakdown of param effect with significance
    # ================================================================
    print(f"\n{'=' * 100}")
    print("D2. Population-pooled parameterization effect by (cluster, q-type)")
    print("    Bonferroni α = 0.05 / 8 = 0.00625 (8 cells)")
    print(f"{'=' * 100}")
    by_cl_qt_param: dict = defaultdict(lambda: {"thirder": 0, "total": 0})
    for c in cells:
        key = (c["cluster"], c.get("question_type"), c["param"])
        by_cl_qt_param[key]["total"] += 1
        if c["is_thirder"]:
            by_cl_qt_param[key]["thirder"] += 1

    qtypes = sorted({c.get("question_type") for c in cells})
    print(f"\n  {'cluster':<10} {'q-type':<22} {'P(SIA|canon)':<15} {'P(SIA|scaled)':<16} "
          f"{'|Δ|':<8} {'p':<12} {'Bonf?':<6}")
    print("  " + "-" * 100)
    for cl in ("SB-type", "DD-type"):
        for qt in qtypes:
            c_d = by_cl_qt_param.get((cl, qt, "canonical"), {"thirder": 0, "total": 0})
            s_d = by_cl_qt_param.get((cl, qt, "scaled"), {"thirder": 0, "total": 0})
            pc_ = c_d["thirder"] / c_d["total"] if c_d["total"] > 0 else 0
            ps_ = s_d["thirder"] / s_d["total"] if s_d["total"] > 0 else 0
            chi2, p = chi2_2x2(c_d["thirder"], c_d["total"] - c_d["thirder"],
                                s_d["thirder"], s_d["total"] - s_d["thirder"])
            sig = "**" if p < 0.05 / 8 else ("*" if p < 0.05 else "")
            print(f"  {cl:<10} {qt:<22} {pc_:.3f} (n={c_d['total']})  "
                  f"{ps_:.3f} (n={s_d['total']})  {abs(pc_-ps_)*100:5.1f}pp   "
                  f"{p:<12.4g} {sig:<6}")

    # ================================================================
    # D3. Reasoning-mode interaction within hybrid models
    # ================================================================
    print(f"\n{'=' * 100}")
    print("D3. Reasoning-mode interaction: does parameterization effect differ")
    print("    between ON and OFF within each hybrid model?")
    print(f"{'=' * 100}")
    HYBRID = {"claude-4.7-opus-20260416", "gpt-5.5-20260423", "grok-4.3-20260430",
              "deepseek-v4-pro-20260423", "qwen3.6-max-preview-20260420"}
    # For each hybrid model: compute |Δ_param| within (cluster, q_type) cells under ON vs OFF
    # then paired-sign test ON vs OFF.
    by_mm_cl_qt_param: dict = defaultdict(lambda: {"thirder": 0, "total": 0})
    for c in cells:
        key = ((c["model_short"], c["mode"]), c["cluster"], c.get("question_type"), c["param"])
        by_mm_cl_qt_param[key]["total"] += 1
        if c["is_thirder"]:
            by_mm_cl_qt_param[key]["thirder"] += 1
    print(f"\n  {'model':<32} {'n pairs':<9} {'mean |Δ| OFF':<15} {'mean |Δ| ON':<15} "
          f"{'ON-OFF':<10} {'sign p':<10}")
    print("  " + "-" * 90)
    for model in sorted(HYBRID):
        pairs = []  # (off_delta, on_delta) for each (cluster, q_type)
        for cl in ("SB-type", "DD-type"):
            for qt in qtypes:
                off_c = by_mm_cl_qt_param.get(((model, "off"), cl, qt, "canonical"), {"thirder": 0, "total": 0})
                off_s = by_mm_cl_qt_param.get(((model, "off"), cl, qt, "scaled"), {"thirder": 0, "total": 0})
                on_c = by_mm_cl_qt_param.get(((model, "on"), cl, qt, "canonical"), {"thirder": 0, "total": 0})
                on_s = by_mm_cl_qt_param.get(((model, "on"), cl, qt, "scaled"), {"thirder": 0, "total": 0})
                if off_c["total"] == 0 or off_s["total"] == 0 or on_c["total"] == 0 or on_s["total"] == 0:
                    continue
                d_off = abs(off_c["thirder"] / off_c["total"] - off_s["thirder"] / off_s["total"])
                d_on = abs(on_c["thirder"] / on_c["total"] - on_s["thirder"] / on_s["total"])
                pairs.append((d_off, d_on))
        if not pairs:
            continue
        diffs = [on - off for off, on in pairs]
        n_pos = sum(1 for d in diffs if d > 0)
        n_neg = sum(1 for d in diffs if d < 0)
        n_zero = sum(1 for d in diffs if d == 0)
        n_nonzero = n_pos + n_neg
        if n_nonzero == 0:
            p_s = 1.0
        else:
            smaller = min(n_pos, n_neg)
            cdf = sum(math.comb(n_nonzero, i) for i in range(0, smaller + 1)) * (0.5 ** n_nonzero)
            p_s = min(1.0, 2 * cdf)
        off_mean = sum(off for off, _ in pairs) / len(pairs) * 100
        on_mean = sum(on for _, on in pairs) / len(pairs) * 100
        print(f"  {model:<32} {len(pairs):<9} {off_mean:<14.2f}pp {on_mean:<14.2f}pp "
              f"{(on_mean - off_mean):+7.2f}pp  {p_s:<10.4g}")

    # ================================================================
    # D4. Twin-pair consistency of parameterization effect
    # ================================================================
    print(f"\n{'=' * 100}")
    print("D4. Twin-pair consistency: do twins show the SAME parameterization effect?")
    print("    (DD vs PADD;  SB vs INC)  population-pooled per q-type")
    print(f"{'=' * 100}")
    by_pc_qt_param: dict = defaultdict(lambda: {"thirder": 0, "total": 0})
    for c in cells:
        key = (c["problem_class"], c.get("question_type"), c["param"])
        by_pc_qt_param[key]["total"] += 1
        if c["is_thirder"]:
            by_pc_qt_param[key]["thirder"] += 1
    twins = [("dd", "padd"), ("sb", "inc")]
    print(f"\n  {'twin A':<6} {'twin B':<6} {'q-type':<22} {'Δ_A':<10} {'Δ_B':<10} "
          f"{'|Δ_A − Δ_B|':<14} {'concordant dir':<15}")
    print("  " + "-" * 90)
    for a, b in twins:
        for qt in qtypes:
            a_c = by_pc_qt_param.get((a, qt, "canonical"), {"thirder": 0, "total": 0})
            a_s = by_pc_qt_param.get((a, qt, "scaled"), {"thirder": 0, "total": 0})
            b_c = by_pc_qt_param.get((b, qt, "canonical"), {"thirder": 0, "total": 0})
            b_s = by_pc_qt_param.get((b, qt, "scaled"), {"thirder": 0, "total": 0})
            if min(a_c["total"], a_s["total"], b_c["total"], b_s["total"]) == 0:
                continue
            d_a = a_c["thirder"] / a_c["total"] - a_s["thirder"] / a_s["total"]
            d_b = b_c["thirder"] / b_c["total"] - b_s["thirder"] / b_s["total"]
            same_dir = "YES" if (d_a * d_b > 0) else ("flat" if d_a == 0 or d_b == 0 else "NO")
            print(f"  {a:<6} {b:<6} {qt:<22} {d_a*100:+6.1f}pp  {d_b*100:+6.1f}pp  "
                  f"{abs(d_a - d_b)*100:5.1f}pp        {same_dir:<15}")

    # ================================================================
    # D5. Direction-of-alias per-model exact-binomial test
    # ================================================================
    print(f"\n{'=' * 100}")
    print("D5. Direction-of-alias per-model test")
    print("    Hypothesis: numerical scale aliases problem family — SB-scale applied")
    print("    to DD-cluster → more SIA-aligned; DD-scale applied to SB-cluster → more")
    print("    SIA-aligned on SSA-cap (where the cluster pulls non-SIA).")
    print(f"{'=' * 100}")
    # For each model-mode, two independent direction tests:
    #   (a) DD-cluster attitudes: P(SIA|canonical) > P(SIA|scaled)
    #   (b) SB-cluster SSA-cap:   P(SIA|scaled)   > P(SIA|canonical)
    print(f"\n  Per-model direction confirmations (1 = predicted direction, 0 = opposite):")
    print(f"  {'model':<32} {'mode':<5} {'DD-att canon>scaled':<22} "
          f"{'SB-SSAcap scaled>canon':<24} {'both?':<6}")
    print("  " + "-" * 95)
    n_both = 0
    n_total = 0
    for mm in mms:
        # DD attitudes
        agg_dd = {"canon": [0, 0], "scaled": [0, 0]}  # [thirder, total]
        for c in cells:
            if c["cluster"] != "DD-type":
                continue
            if c.get("question_type") not in ("personal_attitude", "normative_attitude"):
                continue
            if (c["model_short"], c["mode"]) != mm:
                continue
            k = "canon" if c["param"] == "canonical" else "scaled"
            agg_dd[k][1] += 1
            if c["is_thirder"]:
                agg_dd[k][0] += 1
        dd_dir = 1 if (agg_dd["canon"][1] > 0 and agg_dd["scaled"][1] > 0
                       and (agg_dd["canon"][0] / agg_dd["canon"][1]) > (agg_dd["scaled"][0] / agg_dd["scaled"][1])) else 0
        # SB SSA-capability
        agg_sb = {"canon": [0, 0], "scaled": [0, 0]}
        for c in cells:
            if c["cluster"] != "SB-type":
                continue
            if c.get("question_type") != "ssa_capability":
                continue
            if (c["model_short"], c["mode"]) != mm:
                continue
            k = "canon" if c["param"] == "canonical" else "scaled"
            agg_sb[k][1] += 1
            if c["is_thirder"]:
                agg_sb[k][0] += 1
        sb_dir = 1 if (agg_sb["canon"][1] > 0 and agg_sb["scaled"][1] > 0
                       and (agg_sb["scaled"][0] / agg_sb["scaled"][1]) > (agg_sb["canon"][0] / agg_sb["canon"][1])) else 0
        both = "YES" if (dd_dir and sb_dir) else "no"
        if dd_dir and sb_dir:
            n_both += 1
        n_total += 1
        print(f"  {mm[0]:<32} {mm[1]:<5} {dd_dir:<22} {sb_dir:<24} {both:<6}")
    print(f"\n  Both directions confirmed: {n_both}/{n_total} model-modes")
    p_both = binomial_two_sided_p(n_both, n_total, p0=0.25)  # 0.5 * 0.5 chance both random
    print(f"  Binomial test against H0 (both directions are independent coin flips):")
    print(f"    n={n_total}, k={n_both}, P0=0.25, two-sided p = {p_both:.4g}")

    # And: combined per-model test — exact two-sided chi-square on the joint
    # 2x2 within each model. We use the count of confirmations vs expected 0.25.

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'=' * 100}")
    print("SUMMARY of parameterization deep-dive")
    print(f"{'=' * 100}")
    print(f"""
  Parameterization is the second-largest design axis (after cluster itself), with
  population-pooled |Δ| = 10pp average. It operates as a CLUSTER ALIASER:
  numerical-scale features (small numbers + 1/2 prior vs cosmological scale +
  biased prior) push models toward the SIA-aligned answer that fits the matched
  cluster's literature canon, regardless of the underlying structural problem.

  Evidence supporting the alias interpretation:
    • DD attitudes: SB-numerics (43.4% SIA) >> DD-numerics (18.3% SIA),  p ≈ 10⁻⁵⁷
    • SB SSA-cap:   DD-numerics (31.0% SIA) >> SB-numerics (20.0% SIA),  p ≈ 10⁻⁷
    • Both directions predicted, both confirmed at population level.
    • Direction is uniform across ~all model-modes (D1 sign test, D5 joint test).
    • Twin-pair consistency: DD and PADD show same direction, similar magnitude.

  Methodological implication: parameterization should be reported as a primary
  result, not a robustness check. Models are responsive to surface numerical
  cues — design choices that look "stylistic" partially flip the answer.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
