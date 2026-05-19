#!/usr/bin/env python3
"""RQ9 — Capability-attitude correlation.

Anthropic-domain analog of the BSc-thesis Oesterheld 2024 finding that
higher-capability models prefer EDT-aligned answers over CDT-aligned ones.
Here: do higher-capability models prefer SIA-aligned attitudes over
SSA-aligned ones?

Operationalization:
  - Per (model, mode), capability score = mean V1 accuracy across
    sia_capability + ssa_capability cells.
  - Per (model, mode), attitude SIA-rate = fraction of attitude cells
    (personal + normative) choosing SIA-aligned letter.
  - Spearman / Pearson correlations across 12 (model, mode) configurations.
  - By cluster (SB-type vs DD-type) separately, since cluster pluralism
    dominates attitudes.
  - Population-pooled aggregate (across all cells, not weighted by mm).
  - Permutation test (shuffle capability scores across mm, recompute
    correlation) for non-parametric significance.

Caveat: n = 12 mm configurations is small; correlations need rigorous CIs.
Per-cluster analysis is more informative than overall because SB-cluster
attitudes are at ceiling (~99% thirder regardless of capability).
"""
from __future__ import annotations

import json
import math
import random
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


def get_sia_aligned_letter(preferred_actions: dict, row_order: str):
    if not preferred_actions:
        return None
    sia_pref = preferred_actions.get("sia_preference")
    if not sia_pref:
        return None
    is_A_in_row12 = sia_pref in ("half", "high")
    return ("A" if is_A_in_row12 else "B") if row_order == "12" else ("B" if is_A_in_row12 else "A")


def spearman_rho(xs, ys):
    """Spearman rank correlation."""
    n = len(xs)
    if n < 2:
        return (0.0, 1.0)

    def ranks(values):
        # Average ranks for ties
        sorted_idx = sorted(range(n), key=lambda i: values[i])
        rks = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[sorted_idx[j + 1]] == values[sorted_idx[i]]:
                j += 1
            avg_rank = (i + j) / 2 + 1
            for k in range(i, j + 1):
                rks[sorted_idx[k]] = avg_rank
            i = j + 1
        return rks

    rx = ranks(xs)
    ry = ranks(ys)
    mean_x = sum(rx) / n
    mean_y = sum(ry) / n
    num = sum((rx[i] - mean_x) * (ry[i] - mean_y) for i in range(n))
    den_x = sum((rx[i] - mean_x) ** 2 for i in range(n))
    den_y = sum((ry[i] - mean_y) ** 2 for i in range(n))
    if den_x == 0 or den_y == 0:
        return (0.0, 1.0)
    rho = num / math.sqrt(den_x * den_y)
    # Approximate two-sided p-value via t-distribution
    if abs(rho) >= 1:
        return (rho, 0.0)
    t = rho * math.sqrt((n - 2) / (1 - rho ** 2))
    # Two-sided p via normal approximation (rough for small n)
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2 * (n - 2)))))
    return (rho, max(0.0, min(1.0, p)))


def pearson_r(xs, ys):
    n = len(xs)
    if n < 2:
        return (0.0, 1.0)
    mx = sum(xs) / n
    my = sum(ys) / n
    num = sum((xs[i] - mx) * (ys[i] - my) for i in range(n))
    dx = sum((xs[i] - mx) ** 2 for i in range(n))
    dy = sum((ys[i] - my) ** 2 for i in range(n))
    if dx == 0 or dy == 0:
        return (0.0, 1.0)
    r = num / math.sqrt(dx * dy)
    if abs(r) >= 1:
        return (r, 0.0)
    t = r * math.sqrt((n - 2) / (1 - r ** 2))
    p = 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2 * (n - 2)))))
    return (r, max(0.0, min(1.0, p)))


def permutation_p(xs, ys, n_perm=10000, seed=0):
    """Two-sided permutation test on Pearson r."""
    obs_r, _ = pearson_r(xs, ys)
    rng = random.Random(seed)
    n_extreme = 0
    for _ in range(n_perm):
        shuffled = ys[:]
        rng.shuffle(shuffled)
        r, _ = pearson_r(xs, shuffled)
        if abs(r) >= abs(obs_r) - 1e-12:
            n_extreme += 1
    return (n_extreme + 1) / (n_perm + 1)


def load_cells():
    cells = []
    for f in sorted(D.glob("*.json")):
        d = json.load(open(f))
        ch = d.get("extracted_choice")
        if ch not in ("A", "B"):
            continue
        sia_letter = get_sia_aligned_letter(d.get("preferred_actions"), d.get("row_order"))
        if sia_letter is None:
            continue
        d["mode"] = parse_mode(f.name)
        model = (d.get("model_id_openrouter") or "").split("/")[-1]
        d["model_short"] = model
        d["problem_class"] = parse_problem_class(d.get("template_name", ""))
        d["cluster"] = parse_cluster(d["problem_class"])
        d["is_thirder"] = (ch == sia_letter)
        qt = d.get("question_type")
        if qt == "sia_capability":
            d["is_correct"] = d["is_thirder"]
        elif qt == "ssa_capability":
            d["is_correct"] = not d["is_thirder"]
        else:
            d["is_correct"] = None
        cells.append(d)
    return cells


def main() -> int:
    cells = load_cells()
    print(f"Loaded {len(cells)} cells")

    # Aggregate per (model, mode)
    by_mm: dict = defaultdict(lambda: {
        "cap_correct": 0, "cap_total": 0,
        "att_sia_aligned": 0, "att_total": 0,
        "cap_correct_sb": 0, "cap_total_sb": 0,
        "cap_correct_dd": 0, "cap_total_dd": 0,
        "att_sia_aligned_sb": 0, "att_total_sb": 0,
        "att_sia_aligned_dd": 0, "att_total_dd": 0,
    })
    for c in cells:
        mm = (c["model_short"], c["mode"])
        qt = c.get("question_type")
        cl_key = "sb" if c["cluster"] == "SB-type" else ("dd" if c["cluster"] == "DD-type" else None)
        if qt in ("sia_capability", "ssa_capability"):
            by_mm[mm]["cap_total"] += 1
            if c["is_correct"]:
                by_mm[mm]["cap_correct"] += 1
            if cl_key == "sb":
                by_mm[mm]["cap_total_sb"] += 1
                if c["is_correct"]:
                    by_mm[mm]["cap_correct_sb"] += 1
            elif cl_key == "dd":
                by_mm[mm]["cap_total_dd"] += 1
                if c["is_correct"]:
                    by_mm[mm]["cap_correct_dd"] += 1
        elif qt in ("personal_attitude", "normative_attitude"):
            by_mm[mm]["att_total"] += 1
            if c["is_thirder"]:
                by_mm[mm]["att_sia_aligned"] += 1
            if cl_key == "sb":
                by_mm[mm]["att_total_sb"] += 1
                if c["is_thirder"]:
                    by_mm[mm]["att_sia_aligned_sb"] += 1
            elif cl_key == "dd":
                by_mm[mm]["att_total_dd"] += 1
                if c["is_thirder"]:
                    by_mm[mm]["att_sia_aligned_dd"] += 1

    # Compute per-(mm) rates
    mm_data = []
    for mm in sorted(by_mm.keys()):
        d = by_mm[mm]
        mm_data.append({
            "mm": mm,
            "cap": d["cap_correct"] / d["cap_total"] if d["cap_total"] else 0,
            "att_sia": d["att_sia_aligned"] / d["att_total"] if d["att_total"] else 0,
            "cap_sb": d["cap_correct_sb"] / d["cap_total_sb"] if d["cap_total_sb"] else 0,
            "cap_dd": d["cap_correct_dd"] / d["cap_total_dd"] if d["cap_total_dd"] else 0,
            "att_sia_sb": d["att_sia_aligned_sb"] / d["att_total_sb"] if d["att_total_sb"] else 0,
            "att_sia_dd": d["att_sia_aligned_dd"] / d["att_total_dd"] if d["att_total_dd"] else 0,
        })

    # =================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 1: Per (model, mode) capability score + attitude SIA-rate")
    print(f"{'=' * 100}")
    print(f"\n  {'model':<32} {'mode':<5} {'capability':<13} {'att SIA-rate':<14} "
          f"{'cap SB':<8} {'cap DD':<8} {'att SB':<8} {'att DD':<8}")
    print("  " + "-" * 110)
    for r in mm_data:
        print(f"  {r['mm'][0]:<32} {r['mm'][1]:<5} "
              f"{r['cap']:<12.4f}  {r['att_sia']:<13.4f}  "
              f"{r['cap_sb']:<7.4f}  {r['cap_dd']:<7.4f}  "
              f"{r['att_sia_sb']:<7.4f}  {r['att_sia_dd']:<7.4f}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 2: Overall (pooled across clusters) capability ↔ attitude SIA-rate")
    print(f"          n = 12 (model, mode) configurations")
    print(f"{'=' * 100}")
    xs = [r["cap"] for r in mm_data]
    ys = [r["att_sia"] for r in mm_data]
    spearman, p_sp = spearman_rho(xs, ys)
    pearson, p_pr = pearson_r(xs, ys)
    p_perm = permutation_p(xs, ys, n_perm=10000, seed=42)
    print(f"\n  Spearman ρ  = {spearman:+.4f}   (t-approx p = {p_sp:.4g})")
    print(f"  Pearson r   = {pearson:+.4f}   (t-approx p = {p_pr:.4g})")
    print(f"  Permutation p (two-sided, 10000 perms) = {p_perm:.4g}")
    print(f"  Interpretation: positive ρ/r ⇒ higher capability ↔ higher SIA-aligned attitudes")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 3: Per-cluster breakdown (since SB-cluster attitudes are at ceiling)")
    print(f"{'=' * 100}")

    # SB-cluster capability ↔ SB-cluster attitude SIA-rate
    print(f"\n  3a. SB-cluster: capability_sb ↔ att_sia_sb")
    xs = [r["cap_sb"] for r in mm_data]
    ys = [r["att_sia_sb"] for r in mm_data]
    spearman, p_sp = spearman_rho(xs, ys)
    pearson, p_pr = pearson_r(xs, ys)
    p_perm = permutation_p(xs, ys, n_perm=10000, seed=43)
    print(f"      Spearman ρ  = {spearman:+.4f}   (p ≈ {p_sp:.4g})")
    print(f"      Pearson r   = {pearson:+.4f}   (p ≈ {p_pr:.4g})")
    print(f"      Permutation p = {p_perm:.4g}")
    print(f"      Range of cap_sb: [{min(xs):.4f}, {max(xs):.4f}]   "
          f"Range of att_sia_sb: [{min(ys):.4f}, {max(ys):.4f}]")

    print(f"\n  3b. DD-cluster: capability_dd ↔ att_sia_dd")
    xs = [r["cap_dd"] for r in mm_data]
    ys = [r["att_sia_dd"] for r in mm_data]
    spearman, p_sp = spearman_rho(xs, ys)
    pearson, p_pr = pearson_r(xs, ys)
    p_perm = permutation_p(xs, ys, n_perm=10000, seed=44)
    print(f"      Spearman ρ  = {spearman:+.4f}   (p ≈ {p_sp:.4g})")
    print(f"      Pearson r   = {pearson:+.4f}   (p ≈ {p_pr:.4g})")
    print(f"      Permutation p = {p_perm:.4g}")
    print(f"      Range of cap_dd: [{min(xs):.4f}, {max(xs):.4f}]   "
          f"Range of att_sia_dd: [{min(ys):.4f}, {max(ys):.4f}]")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 4: Cross-correlations — does SB-capability predict DD-attitude?")
    print(f"{'=' * 100}")
    # 4 combinations of (cap cluster, att cluster)
    for cap_key, att_key, label in [
        ("cap_sb", "att_sia_dd", "SB-cap ↔ DD-att"),
        ("cap_dd", "att_sia_sb", "DD-cap ↔ SB-att"),
        ("cap", "att_sia_sb", "overall-cap ↔ SB-att"),
        ("cap", "att_sia_dd", "overall-cap ↔ DD-att"),
    ]:
        xs = [r[cap_key] for r in mm_data]
        ys = [r[att_key] for r in mm_data]
        spearman, p_sp = spearman_rho(xs, ys)
        p_perm = permutation_p(xs, ys, n_perm=5000, seed=hash(label) & 0xffff)
        print(f"\n  {label:<22}: Spearman ρ = {spearman:+.4f} (p ≈ {p_sp:.4g}), "
              f"perm p = {p_perm:.4g}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 5: Population-pooled cell-level test")
    print(f"           Does within-(model, mode) capability correlate with within-(model, mode)")
    print(f"           attitude SIA-rate at the problem-instance level?")
    print(f"{'=' * 100}")
    # Per (mm, problem_instance): capability accuracy + attitude SIA-rate
    # Then pool across all (mm, instance) pairs and correlate
    by_mm_inst: dict = defaultdict(lambda: {
        "cap_correct": 0, "cap_total": 0,
        "att_sia": 0, "att_total": 0,
    })
    for c in cells:
        # problem_instance = (problem_class, theme inferred, parameterization inferred, row_order)
        # For simplicity, use template_name as the instance key
        key = ((c["model_short"], c["mode"]), c.get("template_name"))
        qt = c.get("question_type")
        if qt in ("sia_capability", "ssa_capability"):
            by_mm_inst[key]["cap_total"] += 1
            if c["is_correct"]:
                by_mm_inst[key]["cap_correct"] += 1
        elif qt in ("personal_attitude", "normative_attitude"):
            by_mm_inst[key]["att_total"] += 1
            if c["is_thirder"]:
                by_mm_inst[key]["att_sia"] += 1

    cell_xs, cell_ys = [], []
    for key, d in by_mm_inst.items():
        if d["cap_total"] == 0 or d["att_total"] == 0:
            continue
        cell_xs.append(d["cap_correct"] / d["cap_total"])
        cell_ys.append(d["att_sia"] / d["att_total"])
    n = len(cell_xs)
    print(f"\n  n = {n} (mm × problem_instance) cells with both capability and attitude data")
    spearman, p_sp = spearman_rho(cell_xs, cell_ys)
    pearson, p_pr = pearson_r(cell_xs, cell_ys)
    print(f"  Spearman ρ = {spearman:+.4f}  (p ≈ {p_sp:.4g})")
    print(f"  Pearson  r = {pearson:+.4f}  (p ≈ {p_pr:.4g})")
    # Permutation
    p_perm = permutation_p(cell_xs, cell_ys, n_perm=2000, seed=99)
    print(f"  Permutation p (2000 perms) = {p_perm:.4g}")

    # Per-cluster cell-level
    for cl_filter, cl_name in [("SB-type", "SB-cluster only"), ("DD-type", "DD-cluster only")]:
        by_mm_inst_cl: dict = defaultdict(lambda: {
            "cap_correct": 0, "cap_total": 0,
            "att_sia": 0, "att_total": 0,
        })
        for c in cells:
            if c["cluster"] != cl_filter:
                continue
            key = ((c["model_short"], c["mode"]), c.get("template_name"))
            qt = c.get("question_type")
            if qt in ("sia_capability", "ssa_capability"):
                by_mm_inst_cl[key]["cap_total"] += 1
                if c["is_correct"]:
                    by_mm_inst_cl[key]["cap_correct"] += 1
            elif qt in ("personal_attitude", "normative_attitude"):
                by_mm_inst_cl[key]["att_total"] += 1
                if c["is_thirder"]:
                    by_mm_inst_cl[key]["att_sia"] += 1
        cl_xs, cl_ys = [], []
        for key, d in by_mm_inst_cl.items():
            if d["cap_total"] == 0 or d["att_total"] == 0:
                continue
            cl_xs.append(d["cap_correct"] / d["cap_total"])
            cl_ys.append(d["att_sia"] / d["att_total"])
        spearman, p_sp = spearman_rho(cl_xs, cl_ys)
        pearson, p_pr = pearson_r(cl_xs, cl_ys)
        p_perm = permutation_p(cl_xs, cl_ys, n_perm=2000, seed=hash(cl_name) & 0xffff)
        print(f"\n  {cl_name}: n = {len(cl_xs)}")
        print(f"    Spearman ρ = {spearman:+.4f}  (p ≈ {p_sp:.4g})")
        print(f"    Pearson  r = {pearson:+.4f}  (p ≈ {p_pr:.4g})")
        print(f"    Permutation p = {p_perm:.4g}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 6: Bootstrap CI on per-(model, mode) Spearman ρ")
    print(f"           Resample (mm) configurations with replacement; recompute ρ.")
    print(f"{'=' * 100}")
    for cap_key, att_key, label in [
        ("cap", "att_sia", "overall"),
        ("cap_sb", "att_sia_sb", "SB-cluster"),
        ("cap_dd", "att_sia_dd", "DD-cluster"),
    ]:
        xs = [r[cap_key] for r in mm_data]
        ys = [r[att_key] for r in mm_data]
        rng = random.Random(42)
        n_boot = 2000
        rhos = []
        for _ in range(n_boot):
            indices = [rng.randrange(len(xs)) for _ in range(len(xs))]
            bxs = [xs[i] for i in indices]
            bys = [ys[i] for i in indices]
            try:
                rho, _ = spearman_rho(bxs, bys)
                rhos.append(rho)
            except Exception:
                pass
        if not rhos:
            print(f"  {label}: bootstrap failed")
            continue
        rhos.sort()
        lo = rhos[int(0.025 * len(rhos))]
        hi = rhos[int(0.975 * len(rhos))]
        obs_rho, _ = spearman_rho(xs, ys)
        print(f"  {label:<12}: observed ρ = {obs_rho:+.4f}  "
              f"95% bootstrap CI = [{lo:+.4f}, {hi:+.4f}]   "
              f"{'CI excludes 0' if (lo > 0 or hi < 0) else 'CI includes 0'}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
