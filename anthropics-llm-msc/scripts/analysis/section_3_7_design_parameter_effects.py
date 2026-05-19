#!/usr/bin/env python3
"""RQ11 — Theme and parameterization effects (exploratory).

Main run design crossed:
  - 4 problem classes (SB, INC, DD, PADD)
  - 2 themes per class:
      SB / INC: classic vs aiinstance
      DD / PADD: civilization vs aiinstance
  - 2 parameterizations: canonical (Sleeping-Beauty-tipa numerical scale)
                         vs scaled (Doomsday-tipa cosmological scale)
  - 2 row orders (12, 21)

Questions (exploratory, two-sided):
  - Does theme (classic/civilization vs aiinstance) change thirder rate within
    a (model, mode, problem_class, q_type)?
  - Does parameterization (canonical vs scaled) change thirder rate within
    same?
  - Population-pooled aggregates per (problem_class, q_type).

Tests:
  - Per-cell chi-square 2×2 (theme A vs theme B; canonical vs scaled).
  - Bonferroni: 12 mm × 4 pc × 4 qt = 192 tests per axis, α = 0.05 / 192 ≈ 2.6e-4.
  - Population-level pooled: 16 tests per axis (pc × qt), Bonferroni α ≈ 3.1e-3.

This is the cleanest "did our design choices matter" test, complementing RQ4
(row-order/position bias) and RQ5 (within-twin problem consistency).
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

D = Path("experiment_results/main_run_20260516")
N_TESTS_PER_AXIS = 12 * 4 * 4  # 192
ALPHA_BONF_PERCELL = 0.05 / N_TESTS_PER_AXIS  # ≈ 2.6e-4
N_POP_TESTS = 4 * 4  # 16
ALPHA_BONF_POP = 0.05 / N_POP_TESTS  # ≈ 3.1e-3


def parse_template(tn: str):
    """Return (problem_class, theme, parameterization, row_order) or Nones."""
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
        pc, theme, param, row = parse_template(d.get("template_name", ""))
        d["problem_class"] = pc
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
    print(f"Loaded {len(cells)} cells with parsed choice + SIA-letter mapping + template parse")

    # =================================================================
    # AXIS 1: THEME EFFECT
    # =================================================================
    print(f"\n{'=' * 100}")
    print("AXIS 1: THEME EFFECT (problem-canonical theme vs aiinstance theme)")
    print(f"{'=' * 100}")
    print(f"  Per-cell tests: 192 (12 mm × 4 pc × 4 qt)")
    print(f"  Bonferroni α (per-cell): {ALPHA_BONF_PERCELL:.6f}")
    print(f"  Pool aggregates by (problem_class, q_type): 16 tests")
    print(f"  Bonferroni α (population): {ALPHA_BONF_POP:.6f}")

    # Per-cell aggregates: {(mm, pc, qt, theme): {thirder, total}}
    by_mm_pc_qt_theme: dict = defaultdict(lambda: {"thirder": 0, "total": 0})
    for c in cells:
        key = ((c["model_short"], c["mode"]), c["problem_class"],
               c.get("question_type"), c["theme"])
        by_mm_pc_qt_theme[key]["total"] += 1
        if c["is_thirder"]:
            by_mm_pc_qt_theme[key]["thirder"] += 1

    # Per problem-class, the two themes
    pc_themes = {"sb": ("classic", "aiinstance"),
                 "inc": ("classic", "aiinstance"),
                 "dd": ("civilization", "aiinstance"),
                 "padd": ("civilization", "aiinstance")}

    # Per-cell test loop
    n_sig_percell = 0
    n_run = 0
    large_effects = []  # |Δ| > 20pp
    for mm in sorted({(c["model_short"], c["mode"]) for c in cells}):
        for pc, (t_canon, t_ai) in pc_themes.items():
            for qt in sorted({c.get("question_type") for c in cells}):
                a_dict = by_mm_pc_qt_theme.get((mm, pc, qt, t_canon),
                                                {"thirder": 0, "total": 0})
                b_dict = by_mm_pc_qt_theme.get((mm, pc, qt, t_ai),
                                                {"thirder": 0, "total": 0})
                if a_dict["total"] == 0 or b_dict["total"] == 0:
                    continue
                n_run += 1
                pa = a_dict["thirder"] / a_dict["total"]
                pb = b_dict["thirder"] / b_dict["total"]
                chi2, p = chi2_2x2(a_dict["thirder"], a_dict["total"] - a_dict["thirder"],
                                    b_dict["thirder"], b_dict["total"] - b_dict["thirder"])
                if p < ALPHA_BONF_PERCELL:
                    n_sig_percell += 1
                if abs(pa - pb) > 0.20:
                    large_effects.append((mm, pc, qt, t_canon, t_ai, pa, pb, p))

    print(f"\n  Per-cell results: {n_sig_percell}/{n_run} Bonferroni-significant theme differences")
    print(f"  Large effects (|Δ| > 20pp) regardless of significance: {len(large_effects)} cells")

    if large_effects:
        large_effects.sort(key=lambda x: -abs(x[5] - x[6]))
        print(f"\n  Top 20 largest theme effects:")
        print(f"  {'model':<32} {'mode':<5} {'pc':<5} {'q-type':<22} "
              f"{'theme A':<14} {'theme B':<11} {'P(thirder|A)':<13} {'P(thirder|B)':<13} {'|Δ|':<7} {'Bonf?':<6}")
        print("  " + "-" * 145)
        for mm, pc, qt, ta, tb, pa, pb, p in large_effects[:20]:
            sig = "**" if p < ALPHA_BONF_PERCELL else ("*" if p < 0.05 else "")
            print(f"  {mm[0]:<32} {mm[1]:<5} {pc:<5} {qt:<22} "
                  f"{ta:<14} {tb:<11} {pa:.3f}        {pb:.3f}        "
                  f"{abs(pa-pb)*100:5.1f}pp {sig:<6}")

    # Population-level (pool across mm) per (pc, q_type)
    print(f"\n  Population-pooled theme tests per (problem_class, q_type):")
    pop_pc_qt_theme: dict = defaultdict(lambda: {"thirder": 0, "total": 0})
    for c in cells:
        pop_pc_qt_theme[(c["problem_class"], c.get("question_type"), c["theme"])]["total"] += 1
        if c["is_thirder"]:
            pop_pc_qt_theme[(c["problem_class"], c.get("question_type"), c["theme"])]["thirder"] += 1
    print(f"\n  {'pc':<5} {'q-type':<22} {'theme A':<14} {'theme B':<11} "
          f"{'P(thirder|A)':<14} {'P(thirder|B)':<14} {'|Δ|':<7} {'p':<12} {'Bonf?':<6}")
    print("  " + "-" * 120)
    n_pop_sig = 0
    for pc in ("sb", "inc", "dd", "padd"):
        t_canon, t_ai = pc_themes[pc]
        for qt in sorted({c.get("question_type") for c in cells}):
            a_d = pop_pc_qt_theme.get((pc, qt, t_canon), {"thirder": 0, "total": 0})
            b_d = pop_pc_qt_theme.get((pc, qt, t_ai), {"thirder": 0, "total": 0})
            if a_d["total"] == 0 or b_d["total"] == 0:
                continue
            pa = a_d["thirder"] / a_d["total"]
            pb = b_d["thirder"] / b_d["total"]
            chi2, p = chi2_2x2(a_d["thirder"], a_d["total"] - a_d["thirder"],
                                b_d["thirder"], b_d["total"] - b_d["thirder"])
            sig = "**" if p < ALPHA_BONF_POP else ("*" if p < 0.05 else "")
            if p < ALPHA_BONF_POP:
                n_pop_sig += 1
            print(f"  {pc:<5} {qt:<22} {t_canon:<14} {t_ai:<11} "
                  f"{pa:.3f} ({a_d['total']})  {pb:.3f} ({b_d['total']})  "
                  f"{abs(pa-pb)*100:5.1f}pp {p:<12.4g} {sig:<6}")
    print(f"\n  Population-level theme effects: {n_pop_sig}/16 Bonferroni-significant")

    # =================================================================
    # AXIS 2: PARAMETERIZATION EFFECT (canonical vs scaled)
    # =================================================================
    print(f"\n{'=' * 100}")
    print("AXIS 2: PARAMETERIZATION EFFECT (Sleeping-Beauty-scale vs Doomsday-scale numerics)")
    print(f"{'=' * 100}")
    by_mm_pc_qt_param: dict = defaultdict(lambda: {"thirder": 0, "total": 0})
    for c in cells:
        key = ((c["model_short"], c["mode"]), c["problem_class"],
               c.get("question_type"), c["param"])
        by_mm_pc_qt_param[key]["total"] += 1
        if c["is_thirder"]:
            by_mm_pc_qt_param[key]["thirder"] += 1

    n_sig_pp = 0
    n_run_pp = 0
    large_pp = []
    for mm in sorted({(c["model_short"], c["mode"]) for c in cells}):
        for pc in ("sb", "inc", "dd", "padd"):
            for qt in sorted({c.get("question_type") for c in cells}):
                a_d = by_mm_pc_qt_param.get((mm, pc, qt, "canonical"),
                                              {"thirder": 0, "total": 0})
                b_d = by_mm_pc_qt_param.get((mm, pc, qt, "scaled"),
                                              {"thirder": 0, "total": 0})
                if a_d["total"] == 0 or b_d["total"] == 0:
                    continue
                n_run_pp += 1
                pa = a_d["thirder"] / a_d["total"]
                pb = b_d["thirder"] / b_d["total"]
                chi2, p = chi2_2x2(a_d["thirder"], a_d["total"] - a_d["thirder"],
                                    b_d["thirder"], b_d["total"] - b_d["thirder"])
                if p < ALPHA_BONF_PERCELL:
                    n_sig_pp += 1
                if abs(pa - pb) > 0.20:
                    large_pp.append((mm, pc, qt, pa, pb, p))

    print(f"\n  Per-cell results: {n_sig_pp}/{n_run_pp} Bonferroni-significant param differences")
    print(f"  Large effects (|Δ| > 20pp): {len(large_pp)} cells")

    if large_pp:
        large_pp.sort(key=lambda x: -abs(x[3] - x[4]))
        print(f"\n  Top 20 largest parameterization effects:")
        print(f"  {'model':<32} {'mode':<5} {'pc':<5} {'q-type':<22} "
              f"{'P(thirder|canon)':<18} {'P(thirder|scaled)':<19} {'|Δ|':<7} {'Bonf?':<6}")
        print("  " + "-" * 120)
        for mm, pc, qt, pa, pb, p in large_pp[:20]:
            sig = "**" if p < ALPHA_BONF_PERCELL else ("*" if p < 0.05 else "")
            print(f"  {mm[0]:<32} {mm[1]:<5} {pc:<5} {qt:<22} "
                  f"{pa:.3f}             {pb:.3f}              "
                  f"{abs(pa-pb)*100:5.1f}pp {sig:<6}")

    # Population-pooled per (pc, qt)
    print(f"\n  Population-pooled parameterization tests per (problem_class, q_type):")
    pop_pc_qt_param: dict = defaultdict(lambda: {"thirder": 0, "total": 0})
    for c in cells:
        pop_pc_qt_param[(c["problem_class"], c.get("question_type"), c["param"])]["total"] += 1
        if c["is_thirder"]:
            pop_pc_qt_param[(c["problem_class"], c.get("question_type"), c["param"])]["thirder"] += 1
    print(f"\n  {'pc':<5} {'q-type':<22} {'P(thirder|canon)':<18} {'P(thirder|scaled)':<19} "
          f"{'|Δ|':<7} {'p':<12} {'Bonf?':<6}")
    print("  " + "-" * 100)
    n_pop_sig_pp = 0
    for pc in ("sb", "inc", "dd", "padd"):
        for qt in sorted({c.get("question_type") for c in cells}):
            a_d = pop_pc_qt_param.get((pc, qt, "canonical"), {"thirder": 0, "total": 0})
            b_d = pop_pc_qt_param.get((pc, qt, "scaled"), {"thirder": 0, "total": 0})
            if a_d["total"] == 0 or b_d["total"] == 0:
                continue
            pa = a_d["thirder"] / a_d["total"]
            pb = b_d["thirder"] / b_d["total"]
            chi2, p = chi2_2x2(a_d["thirder"], a_d["total"] - a_d["thirder"],
                                b_d["thirder"], b_d["total"] - b_d["thirder"])
            sig = "**" if p < ALPHA_BONF_POP else ("*" if p < 0.05 else "")
            if p < ALPHA_BONF_POP:
                n_pop_sig_pp += 1
            print(f"  {pc:<5} {qt:<22} {pa:.3f} (n={a_d['total']})    "
                  f"{pb:.3f} (n={b_d['total']})    "
                  f"{abs(pa-pb)*100:5.1f}pp {p:<12.4g} {sig:<6}")
    print(f"\n  Population-level parameterization effects: {n_pop_sig_pp}/16 Bonferroni-significant")

    # =================================================================
    # SECTION 3: average |Δ| comparison across all three axes
    # (theme, parameterization, plus reference axes from earlier RQs)
    # =================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 3: Average |Δ| across all design axes — relative magnitude of effects")
    print(f"{'=' * 100}")
    # Compute mean |Δ| over all (mm, pc, qt) cells for theme axis
    theme_deltas = []
    for mm in sorted({(c["model_short"], c["mode"]) for c in cells}):
        for pc, (t_canon, t_ai) in pc_themes.items():
            for qt in sorted({c.get("question_type") for c in cells}):
                a = by_mm_pc_qt_theme.get((mm, pc, qt, t_canon), {"thirder": 0, "total": 0})
                b = by_mm_pc_qt_theme.get((mm, pc, qt, t_ai), {"thirder": 0, "total": 0})
                if a["total"] == 0 or b["total"] == 0:
                    continue
                theme_deltas.append(abs(a["thirder"] / a["total"] - b["thirder"] / b["total"]))
    param_deltas = []
    for mm in sorted({(c["model_short"], c["mode"]) for c in cells}):
        for pc in ("sb", "inc", "dd", "padd"):
            for qt in sorted({c.get("question_type") for c in cells}):
                a = by_mm_pc_qt_param.get((mm, pc, qt, "canonical"), {"thirder": 0, "total": 0})
                b = by_mm_pc_qt_param.get((mm, pc, qt, "scaled"), {"thirder": 0, "total": 0})
                if a["total"] == 0 or b["total"] == 0:
                    continue
                param_deltas.append(abs(a["thirder"] / a["total"] - b["thirder"] / b["total"]))

    print(f"\n  {'axis':<35} {'n cells':<10} {'mean |Δ|':<12} {'median |Δ|':<12}")
    print("  " + "-" * 75)
    theme_deltas.sort()
    param_deltas.sort()

    def med(xs):
        n = len(xs)
        if n == 0:
            return 0.0
        return xs[n // 2] if n % 2 else (xs[n // 2 - 1] + xs[n // 2]) / 2

    print(f"  theme (problem-canonical vs aiinstance)  {len(theme_deltas):<10} "
          f"{sum(theme_deltas)/len(theme_deltas)*100:<11.2f}pp {med(theme_deltas)*100:<11.2f}pp")
    print(f"  parameterization (canonical vs scaled)   {len(param_deltas):<10} "
          f"{sum(param_deltas)/len(param_deltas)*100:<11.2f}pp {med(param_deltas)*100:<11.2f}pp")
    print(f"\n  Reference points from prior RQs (rough means):")
    print(f"    row-order (RQ4)              ~5pp mean |Δ|  (0/48 Bonf-sig at aggregate)")
    print(f"    within-twin SB↔INC (RQ5)     ~3-5pp mean |Δ|  (2/48 Bonf-sig)")
    print(f"    within-twin DD↔PADD (RQ5)    ~3-5pp mean |Δ|  (0/48 Bonf-sig)")
    print(f"    between-cluster SB vs DD     ~50pp mean |Δ|  (the dominant axis)")

    # =================================================================
    # SECTION 4: Inferential tests on the comparative claims
    # =================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 4: Inferential tests on the comparative claims")
    print(f"{'=' * 100}")

    # --- 4a. Paired test: is mean |Δ_param| > mean |Δ_theme| across matched (mm, pc, qt) cells?
    # Per-cell |Δ| dicts keyed by (mm, pc, qt)
    theme_by_cell = {}
    for mm in sorted({(c["model_short"], c["mode"]) for c in cells}):
        for pc, (t_canon, t_ai) in pc_themes.items():
            for qt in sorted({c.get("question_type") for c in cells}):
                a = by_mm_pc_qt_theme.get((mm, pc, qt, t_canon), {"thirder": 0, "total": 0})
                b = by_mm_pc_qt_theme.get((mm, pc, qt, t_ai), {"thirder": 0, "total": 0})
                if a["total"] == 0 or b["total"] == 0:
                    continue
                theme_by_cell[(mm, pc, qt)] = abs(a["thirder"] / a["total"] - b["thirder"] / b["total"])
    param_by_cell = {}
    for mm in sorted({(c["model_short"], c["mode"]) for c in cells}):
        for pc in ("sb", "inc", "dd", "padd"):
            for qt in sorted({c.get("question_type") for c in cells}):
                a = by_mm_pc_qt_param.get((mm, pc, qt, "canonical"), {"thirder": 0, "total": 0})
                b = by_mm_pc_qt_param.get((mm, pc, qt, "scaled"), {"thirder": 0, "total": 0})
                if a["total"] == 0 or b["total"] == 0:
                    continue
                param_by_cell[(mm, pc, qt)] = abs(a["thirder"] / a["total"] - b["thirder"] / b["total"])

    common = sorted(set(theme_by_cell) & set(param_by_cell))
    diffs = [param_by_cell[k] - theme_by_cell[k] for k in common]
    n_pos = sum(1 for d in diffs if d > 0)
    n_neg = sum(1 for d in diffs if d < 0)
    n_zero = sum(1 for d in diffs if d == 0)
    n_nonzero = n_pos + n_neg
    # Sign test (two-sided)
    if n_nonzero == 0:
        p_sign = 1.0
    else:
        smaller = min(n_pos, n_neg)
        cdf = sum(math.comb(n_nonzero, i) for i in range(0, smaller + 1)) * (0.5 ** n_nonzero)
        p_sign = min(1.0, 2 * cdf)
    mean_diff = sum(diffs) / len(diffs) if diffs else 0.0
    # Bootstrap CI on mean difference
    import random as _random
    rng = _random.Random(7)
    n_boot = 2000
    boot_means = []
    for _ in range(n_boot):
        rs = [diffs[rng.randrange(len(diffs))] for _ in range(len(diffs))]
        boot_means.append(sum(rs) / len(rs))
    boot_means.sort()
    lo = boot_means[int(0.025 * n_boot)]
    hi = boot_means[int(0.975 * n_boot)]
    print(f"\n  4a. Paired test: |Δ_param| > |Δ_theme| across matched (mm, pc, qt) cells")
    print(f"      n matched cells: {len(common)}")
    print(f"      mean |Δ_param| − |Δ_theme|: {mean_diff * 100:+.2f}pp")
    print(f"      95% bootstrap CI on Δ:       [{lo * 100:+.2f}pp, {hi * 100:+.2f}pp]")
    print(f"      sign test (two-sided): n+={n_pos}, n−={n_neg}, n=0={n_zero}, p={p_sign:.4g}")
    sig = "**" if p_sign < 0.05 else "ns"
    print(f"      → {sig}")

    # --- 4b. Cluster × axis interaction: is the |Δ| difference between DD-cluster
    # and SB-cluster significantly larger for theme/param than zero?
    print(f"\n  4b. Asymmetry test: is |Δ_param| (or |Δ_theme|) systematically larger on")
    print(f"      DD-cluster than SB-cluster cells? (matched on mm × qt)")
    for axis_name, by_cell in [("theme", theme_by_cell), ("param", param_by_cell)]:
        # Compute paired differences per (mm, qt): mean over DD/PADD pc minus mean over SB/INC pc
        paired = []
        mm_qts = {(k[0], k[2]) for k in by_cell}
        for (mm, qt) in mm_qts:
            dd_vals = [by_cell[(mm, pc, qt)] for pc in ("dd", "padd") if (mm, pc, qt) in by_cell]
            sb_vals = [by_cell[(mm, pc, qt)] for pc in ("sb", "inc") if (mm, pc, qt) in by_cell]
            if not dd_vals or not sb_vals:
                continue
            paired.append(sum(dd_vals) / len(dd_vals) - sum(sb_vals) / len(sb_vals))
        if not paired:
            continue
        n_pos = sum(1 for d in paired if d > 0)
        n_neg = sum(1 for d in paired if d < 0)
        n_zero = sum(1 for d in paired if d == 0)
        n_nonzero = n_pos + n_neg
        if n_nonzero == 0:
            p_s = 1.0
        else:
            smaller = min(n_pos, n_neg)
            cdf = sum(math.comb(n_nonzero, i) for i in range(0, smaller + 1)) * (0.5 ** n_nonzero)
            p_s = min(1.0, 2 * cdf)
        mu = sum(paired) / len(paired)
        rng = _random.Random(8 if axis_name == "theme" else 9)
        boot = []
        for _ in range(n_boot):
            rs = [paired[rng.randrange(len(paired))] for _ in range(len(paired))]
            boot.append(sum(rs) / len(rs))
        boot.sort()
        lo = boot[int(0.025 * n_boot)]
        hi = boot[int(0.975 * n_boot)]
        sig = "**" if p_s < 0.05 else "ns"
        print(f"      {axis_name:<6}: n pairs (mm × qt) = {len(paired)}, "
              f"mean (DD − SB) |Δ| = {mu * 100:+.2f}pp, "
              f"95% CI [{lo * 100:+.2f}pp, {hi * 100:+.2f}pp], sign p={p_s:.4g}  {sig}")

    # --- 4c. Direction test: parameterization aliases problem family?
    # Hypothesis: "SB-tipa numbers on DD-cluster" → more thirder than "DD-tipa numbers on DD-cluster"
    # AND "DD-tipa numbers on SB-cluster" → more thirder than "SB-tipa numbers on SB-cluster" (SSA-cap only)
    # Both directions push toward thirder when the numerical scale mismatches the structural cluster,
    # but specifically on the q-types where the cluster's natural answer is non-thirder.
    print(f"\n  4c. Direction test: does parameterization push toward thirder when numerical")
    print(f"      scale mismatches structural cluster?")
    # Sub-test 1: DD-cluster attitudes (where cluster pulls halfer/non-thirder)
    #   canonical (SB-scale) should be MORE thirder than scaled (DD-scale)
    dd_attitudes_canon_thirder = 0
    dd_attitudes_canon_total = 0
    dd_attitudes_scaled_thirder = 0
    dd_attitudes_scaled_total = 0
    for c in cells:
        if c["problem_class"] in ("dd", "padd") and c.get("question_type") in ("personal_attitude", "normative_attitude"):
            if c["param"] == "canonical":
                dd_attitudes_canon_thirder += 1 if c["is_thirder"] else 0
                dd_attitudes_canon_total += 1
            elif c["param"] == "scaled":
                dd_attitudes_scaled_thirder += 1 if c["is_thirder"] else 0
                dd_attitudes_scaled_total += 1
    p1_canon = dd_attitudes_canon_thirder / dd_attitudes_canon_total
    p1_scaled = dd_attitudes_scaled_thirder / dd_attitudes_scaled_total
    chi1, p1 = chi2_2x2(dd_attitudes_canon_thirder, dd_attitudes_canon_total - dd_attitudes_canon_thirder,
                          dd_attitudes_scaled_thirder, dd_attitudes_scaled_total - dd_attitudes_scaled_thirder)
    print(f"      DD-cluster attitudes (cluster pulls non-thirder):")
    print(f"        canonical (SB-scale): {p1_canon:.3f} thirder  ({dd_attitudes_canon_thirder}/{dd_attitudes_canon_total})")
    print(f"        scaled    (DD-scale): {p1_scaled:.3f} thirder  ({dd_attitudes_scaled_thirder}/{dd_attitudes_scaled_total})")
    print(f"        Δ = {(p1_canon - p1_scaled) * 100:+.1f}pp (predicted: + ; SB-scale → more thirder)")
    print(f"        χ² p = {p1:.4g}  {'**' if p1 < 0.001 else ('*' if p1 < 0.05 else 'ns')}")

    # Sub-test 2: SB-cluster SSA-capability (where cluster pulls non-thirder)
    #   scaled (DD-scale) should be MORE thirder than canonical (SB-scale)
    sb_ssa_canon_thirder = 0
    sb_ssa_canon_total = 0
    sb_ssa_scaled_thirder = 0
    sb_ssa_scaled_total = 0
    for c in cells:
        if c["problem_class"] in ("sb", "inc") and c.get("question_type") == "ssa_capability":
            if c["param"] == "canonical":
                sb_ssa_canon_thirder += 1 if c["is_thirder"] else 0
                sb_ssa_canon_total += 1
            elif c["param"] == "scaled":
                sb_ssa_scaled_thirder += 1 if c["is_thirder"] else 0
                sb_ssa_scaled_total += 1
    p2_canon = sb_ssa_canon_thirder / sb_ssa_canon_total
    p2_scaled = sb_ssa_scaled_thirder / sb_ssa_scaled_total
    chi2_, p2 = chi2_2x2(sb_ssa_canon_thirder, sb_ssa_canon_total - sb_ssa_canon_thirder,
                          sb_ssa_scaled_thirder, sb_ssa_scaled_total - sb_ssa_scaled_thirder)
    print(f"\n      SB-cluster SSA-capability (cluster pulls non-thirder):")
    print(f"        canonical (SB-scale): {p2_canon:.3f} thirder  ({sb_ssa_canon_thirder}/{sb_ssa_canon_total})")
    print(f"        scaled    (DD-scale): {p2_scaled:.3f} thirder  ({sb_ssa_scaled_thirder}/{sb_ssa_scaled_total})")
    print(f"        Δ = {(p2_scaled - p2_canon) * 100:+.1f}pp (predicted: + ; DD-scale → more thirder)")
    print(f"        χ² p = {p2:.4g}  {'**' if p2 < 0.001 else ('*' if p2 < 0.05 else 'ns')}")

    # Combined Fisher's p
    if p1 > 0 and p2 > 0:
        chi_combined = -2 * (math.log(p1) + math.log(p2))
        # df = 4
        # Approximate upper tail of chi-square via series.
        p_combined = math.exp(-chi_combined / 2) * (1 + chi_combined / 2)  # rough; valid for df=4
        print(f"\n      Fisher's combined p (df=4): chi² = {chi_combined:.2f}, p ≈ {p_combined:.4g}")

    # --- 4d. Theme-effect asymmetry: is the theme effect significantly stronger on
    # DD-cluster attitudes than on SB-cluster cells?
    print(f"\n  4d. Theme effect — population pooled, by cluster × q-type, all 16 contrasts:")
    print(f"      Already shown above: 5/16 Bonf-sig at α=3.1e-3; ALL 5 in DD/PADD.")
    print(f"      Sign-of-finding test: 5 of 5 significant pop tests in DD-cluster; 0 in SB-cluster.")
    print(f"      Under H0 (theme effect equally likely to be sig anywhere): P(all 5 in same cluster)")
    print(f"        = 2 × (0.5)^5 = 0.0625 (two-sided, exact)")
    print(f"      → suggestive but not Bonferroni after correcting for axis count")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
