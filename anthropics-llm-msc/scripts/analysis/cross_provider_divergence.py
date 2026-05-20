#!/usr/bin/env python3
"""RQ6 — Cross-provider divergence patterns.

Pre-registered question (exploratory):
  How much do the 12 (model, mode) configurations agree on individual scenarios?
  Which model-modes are outliers? Which scenarios are most controversial?

Operationalization:
  A "scenario" = unique (template, theme, row_order, parameterization, q_type).
  For each (model, mode, scenario), the modal choice across 9 samples is taken
  as that configuration's "answer" for that scenario.
  Pairwise agreement between two model-modes = fraction of scenarios where
  their modal choices match.

Analyses:
  1. Pairwise agreement matrix (12 × 12).
  2. Per-configuration mean agreement with the other 11 (outlier detection).
  3. Permutation test: is observed mean agreement above chance (0.5)?
  4. Per-cluster and per-q-type agreement (given pluralism, agreement should
     be higher within q-type / cluster slice than overall).
  5. Controversy ranking: scenarios sorted by 12-vote entropy descending —
     most-controversial scenarios surface candidates for qualitative deep-dive.
"""
from __future__ import annotations

import json
import math
import random
import re
from collections import Counter, defaultdict
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


def binary_entropy(n_a: int, n_b: int) -> float:
    n = n_a + n_b
    if n == 0:
        return 0.0
    p = n_a / n
    if p in (0.0, 1.0):
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


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
        cells.append(d)
    return cells


def main() -> int:
    cells = load_cells()
    print(f"Loaded {len(cells)} cells with parsed A/B choice")

    # Group by (model_mode, scenario) → list of 9 sample choices, take modal.
    # Scenario key: (template, theme, row_order, parameterization, q_type).
    by_mm_scenario: dict = defaultdict(list)
    for c in cells:
        mm = (c["model_short"], c["mode"])
        scenario = (c.get("template_name"), c.get("theme"), c.get("row_order"),
                    c.get("parameterization"), c.get("question_type"))
        by_mm_scenario[(mm, scenario)].append(c["extracted_choice"])

    # Modal choice per (mm, scenario). Tie → skip (rare).
    mm_scenario_choice: dict = {}
    n_tie = 0
    for (mm, scenario), choices in by_mm_scenario.items():
        ct = Counter(choices)
        a, b = ct.get("A", 0), ct.get("B", 0)
        if a == b:
            n_tie += 1
            continue
        mm_scenario_choice[(mm, scenario)] = "A" if a > b else "B"
    print(f"Modal choices computed for {len(mm_scenario_choice)} (mm, scenario) pairs "
          f"({n_tie} tied across 9 samples — excluded)")

    # Universe of model-modes and scenarios.
    model_modes = sorted({mm for (mm, _) in mm_scenario_choice.keys()})
    scenarios = sorted({sc for (_, sc) in mm_scenario_choice.keys()})
    print(f"Model-modes: {len(model_modes)};  scenarios: {len(scenarios)}")

    # ================================================================
    # 1. Pairwise agreement matrix
    # ================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 1: Pairwise agreement matrix (fraction of scenarios with same modal choice)")
    print(f"{'=' * 100}")
    agreement: dict = {}
    overlap: dict = {}
    for i, mm_i in enumerate(model_modes):
        for mm_j in model_modes[i:]:
            n_match = 0
            n_overlap = 0
            for sc in scenarios:
                ci = mm_scenario_choice.get((mm_i, sc))
                cj = mm_scenario_choice.get((mm_j, sc))
                if ci is None or cj is None:
                    continue
                n_overlap += 1
                if ci == cj:
                    n_match += 1
            agr = n_match / n_overlap if n_overlap > 0 else 0.0
            agreement[(mm_i, mm_j)] = agr
            agreement[(mm_j, mm_i)] = agr
            overlap[(mm_i, mm_j)] = n_overlap
            overlap[(mm_j, mm_i)] = n_overlap

    def label(mm):
        m, mode = mm
        # Compact model name
        return f"{m[:18]:<18}|{mode}"

    print(f"\n  Diagonal omitted; off-diagonal = pairwise agreement (∈ [0, 1]).")
    print(f"  Headers truncated for layout; n in cell is overlap count (scenarios both saw).")
    print()
    # Header row
    hdr = "  " + " " * 22 + "  ".join(f"{i:>5d}" for i in range(len(model_modes)))
    print(hdr)
    for i, mm_i in enumerate(model_modes):
        row = f"  [{i:2d}] {label(mm_i):<22}"
        for j, mm_j in enumerate(model_modes):
            if i == j:
                row += "    -  "
            else:
                row += f"  {agreement[(mm_i, mm_j)]:.3f}"
        print(row)
    print("\n  Index → model-mode legend:")
    for i, mm in enumerate(model_modes):
        print(f"    [{i:2d}] {mm[0]:<32} {mm[1]}")

    # ================================================================
    # 2. Mean off-diagonal agreement per configuration (outlier detection)
    # ================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 2: Mean agreement of each (model, mode) with the other 11")
    print("           (low value = outlier configuration)")
    print(f"{'=' * 100}")
    mean_agr = []
    for mm in model_modes:
        others = [agreement[(mm, other)] for other in model_modes if other != mm]
        mean_agr.append((mm, sum(others) / len(others)))
    mean_agr.sort(key=lambda x: x[1])
    print(f"\n  {'rank':<5} {'model':<32} {'mode':<5} {'mean pairwise agr':<20}")
    print("  " + "-" * 65)
    for rank, (mm, mu) in enumerate(mean_agr, start=1):
        marker = " ← outlier" if mu < 0.55 else ""
        print(f"  {rank:<5} {mm[0]:<32} {mm[1]:<5} {mu:.4f}{marker}")

    # Population-level mean off-diagonal agreement.
    n_pairs = 0
    sum_agr = 0.0
    for i, mm_i in enumerate(model_modes):
        for mm_j in model_modes[i + 1:]:
            n_pairs += 1
            sum_agr += agreement[(mm_i, mm_j)]
    population_mean = sum_agr / n_pairs if n_pairs > 0 else 0.0
    print(f"\n  Overall mean pairwise agreement (66 unique pairs): {population_mean:.4f}")
    print(f"  Chance baseline (random binary choice): 0.5000")

    # ================================================================
    # 3. Permutation test: is observed mean agreement above chance?
    # ================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 3: Permutation test on overall mean pairwise agreement")
    print(f"{'=' * 100}")
    print(f"  Null: each model-mode picks A/B independently per scenario (p=0.5).")
    print(f"  Test stat: mean pairwise agreement across 66 pairs × scenarios.")

    # Build per-scenario matrix for permutation: list of dicts {mm: choice}
    scenario_choices = [{mm: mm_scenario_choice.get((mm, sc)) for mm in model_modes}
                        for sc in scenarios]
    # Observed stat = population_mean (already computed)
    obs = population_mean
    n_perm = 2000
    random.seed(42)
    n_ge = 0
    for _ in range(n_perm):
        # Under H0: each (mm, scenario) sample is i.i.d. Bernoulli(0.5).
        # Simulate by independently flipping each model-mode's choice per scenario.
        n_match = 0
        n_overlap = 0
        for i, mm_i in enumerate(model_modes):
            for mm_j in model_modes[i + 1:]:
                for sc_idx in range(len(scenarios)):
                    ci = scenario_choices[sc_idx][mm_i]
                    cj = scenario_choices[sc_idx][mm_j]
                    if ci is None or cj is None:
                        continue
                    n_overlap += 1
                    # Each side picks A with prob 0.5 → match with prob 0.5
                    if random.random() < 0.5:
                        n_match += 1
        stat = n_match / n_overlap if n_overlap > 0 else 0.0
        if stat >= obs:
            n_ge += 1
    p_perm = (n_ge + 1) / (n_perm + 1)
    print(f"\n  Observed mean pairwise agreement: {obs:.4f}")
    print(f"  Permutation p-value (one-sided, {n_perm} permutations): {p_perm:.4g}")
    print(f"  → {'reject null (agreement > chance)' if p_perm < 0.05 else 'fail to reject null'}")

    # ================================================================
    # 4. Per q-type and per-cluster agreement slices
    # ================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 4a: Per-q-type mean pairwise agreement")
    print(f"{'=' * 100}")
    qtypes = sorted({sc[4] for sc in scenarios})
    print(f"\n  {'q-type':<25} {'n scenarios':<13} {'mean pairwise agr':<20}")
    print("  " + "-" * 60)
    for qt in qtypes:
        sc_qt = [sc for sc in scenarios if sc[4] == qt]
        n_pairs_qt = 0
        sum_agr_qt = 0.0
        for i, mm_i in enumerate(model_modes):
            for mm_j in model_modes[i + 1:]:
                n_match = 0
                n_ovl = 0
                for sc in sc_qt:
                    ci = mm_scenario_choice.get((mm_i, sc))
                    cj = mm_scenario_choice.get((mm_j, sc))
                    if ci is None or cj is None:
                        continue
                    n_ovl += 1
                    if ci == cj:
                        n_match += 1
                if n_ovl > 0:
                    n_pairs_qt += 1
                    sum_agr_qt += n_match / n_ovl
        mu_qt = sum_agr_qt / n_pairs_qt if n_pairs_qt > 0 else 0.0
        print(f"  {qt:<25} {len(sc_qt):<13} {mu_qt:.4f}")

    print(f"\n{'=' * 100}")
    print("SECTION 4b: Per-cluster mean pairwise agreement")
    print(f"{'=' * 100}")
    clusters = sorted({parse_cluster(parse_problem_class(sc[0])) for sc in scenarios})
    print(f"\n  {'cluster':<12} {'n scenarios':<13} {'mean pairwise agr':<20}")
    print("  " + "-" * 50)
    for cl in clusters:
        sc_cl = [sc for sc in scenarios if parse_cluster(parse_problem_class(sc[0])) == cl]
        n_pairs_cl = 0
        sum_agr_cl = 0.0
        for i, mm_i in enumerate(model_modes):
            for mm_j in model_modes[i + 1:]:
                n_match = 0
                n_ovl = 0
                for sc in sc_cl:
                    ci = mm_scenario_choice.get((mm_i, sc))
                    cj = mm_scenario_choice.get((mm_j, sc))
                    if ci is None or cj is None:
                        continue
                    n_ovl += 1
                    if ci == cj:
                        n_match += 1
                if n_ovl > 0:
                    n_pairs_cl += 1
                    sum_agr_cl += n_match / n_ovl
        mu_cl = sum_agr_cl / n_pairs_cl if n_pairs_cl > 0 else 0.0
        print(f"  {cl:<12} {len(sc_cl):<13} {mu_cl:.4f}")

    # Finer: per (cluster, q_type)
    print(f"\n{'=' * 100}")
    print("SECTION 4c: Per (cluster, q_type) mean pairwise agreement")
    print(f"{'=' * 100}")
    print(f"\n  {'cluster':<12} {'q-type':<25} {'n scenarios':<13} {'mean pairwise agr':<20}")
    print("  " + "-" * 75)
    for cl in clusters:
        for qt in qtypes:
            sc_slice = [sc for sc in scenarios
                        if parse_cluster(parse_problem_class(sc[0])) == cl and sc[4] == qt]
            if not sc_slice:
                continue
            n_pairs_s = 0
            sum_agr_s = 0.0
            for i, mm_i in enumerate(model_modes):
                for mm_j in model_modes[i + 1:]:
                    n_match = 0
                    n_ovl = 0
                    for sc in sc_slice:
                        ci = mm_scenario_choice.get((mm_i, sc))
                        cj = mm_scenario_choice.get((mm_j, sc))
                        if ci is None or cj is None:
                            continue
                        n_ovl += 1
                        if ci == cj:
                            n_match += 1
                    if n_ovl > 0:
                        n_pairs_s += 1
                        sum_agr_s += n_match / n_ovl
            mu_s = sum_agr_s / n_pairs_s if n_pairs_s > 0 else 0.0
            print(f"  {cl:<12} {qt:<25} {len(sc_slice):<13} {mu_s:.4f}")

    # ================================================================
    # 5. Controversy ranking: scenarios by 12-vote entropy
    # ================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 5: Controversy ranking — scenarios sorted by 12-vote entropy desc")
    print("           (entropy 1.0 = perfect 6-6 split; entropy 0 = unanimous 12-0)")
    print(f"{'=' * 100}")
    scenario_entropy = []
    for sc in scenarios:
        votes = [mm_scenario_choice.get((mm, sc)) for mm in model_modes]
        n_a = sum(1 for v in votes if v == "A")
        n_b = sum(1 for v in votes if v == "B")
        e = binary_entropy(n_a, n_b)
        scenario_entropy.append((sc, n_a, n_b, e))
    scenario_entropy.sort(key=lambda x: -x[3])

    # Summary stats
    n_unan = sum(1 for x in scenario_entropy if x[3] == 0.0)
    n_split = sum(1 for x in scenario_entropy if x[3] > 0.9)
    mean_e = sum(x[3] for x in scenario_entropy) / len(scenario_entropy)
    print(f"\n  Total scenarios: {len(scenario_entropy)}")
    print(f"  Unanimous (entropy=0): {n_unan} ({100*n_unan/len(scenario_entropy):.2f}%)")
    print(f"  Highly split (entropy>0.9): {n_split} ({100*n_split/len(scenario_entropy):.2f}%)")
    print(f"  Mean cross-model entropy per scenario: {mean_e:.4f}")

    print(f"\n  Top 30 most-controversial scenarios:")
    print(f"  {'rank':<5} {'template':<32} {'theme':<14} {'row':<4} {'param':<10} "
          f"{'q-type':<22} {'A':<3} {'B':<3} {'H':<6}")
    print("  " + "-" * 110)
    for rank, (sc, n_a, n_b, e) in enumerate(scenario_entropy[:30], start=1):
        tpl, th, ro, pa, qt = sc
        tpl_s = (tpl or "")[:30]
        th_s = (th or "")[:12]
        pa_s = (pa or "")[:8]
        qt_s = (qt or "")[:20]
        print(f"  {rank:<5} {tpl_s:<32} {th_s:<14} {ro:<4} {pa_s:<10} "
              f"{qt_s:<22} {n_a:<3} {n_b:<3} {e:.3f}")

    # Distribution by problem-class for controversial scenarios (top 30)
    top_pc: Counter = Counter()
    top_qt: Counter = Counter()
    for sc, _, _, _ in scenario_entropy[:30]:
        top_pc[parse_problem_class(sc[0])] += 1
        top_qt[sc[4]] += 1
    print(f"\n  Top-30 breakdown by problem class: {dict(top_pc)}")
    print(f"  Top-30 breakdown by q-type:        {dict(top_qt)}")

    # And the inverse: most-unanimous scenarios (sanity check that any exist)
    n_show_unan = min(15, n_unan)
    print(f"\n  Sample {n_show_unan} unanimous scenarios (entropy=0):")
    print(f"  {'template':<32} {'theme':<14} {'row':<4} {'param':<10} "
          f"{'q-type':<22} {'A':<3} {'B':<3}")
    print("  " + "-" * 100)
    unan = [x for x in scenario_entropy if x[3] == 0.0]
    random.seed(11)
    sample = random.sample(unan, n_show_unan) if len(unan) >= n_show_unan else unan
    for sc, n_a, n_b, _ in sample:
        tpl, th, ro, pa, qt = sc
        print(f"  {(tpl or '')[:30]:<32} {(th or '')[:12]:<14} {ro:<4} "
              f"{(pa or '')[:8]:<10} {(qt or '')[:20]:<22} {n_a:<3} {n_b:<3}")

    # ================================================================
    # 6. Inferential tests on the descriptives above
    # ================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 6: Statistical-significance tests on RQ6 descriptives")
    print(f"{'=' * 100}")

    # Helper: compute pairwise agreement over a list of scenarios (returns mean over 66 pairs).
    def mean_pairwise_over(sc_list):
        n_pairs_s = 0
        sum_agr_s = 0.0
        for i, mm_i in enumerate(model_modes):
            for mm_j in model_modes[i + 1:]:
                n_match = 0
                n_ovl = 0
                for sc in sc_list:
                    ci = mm_scenario_choice.get((mm_i, sc))
                    cj = mm_scenario_choice.get((mm_j, sc))
                    if ci is None or cj is None:
                        continue
                    n_ovl += 1
                    if ci == cj:
                        n_match += 1
                if n_ovl > 0:
                    n_pairs_s += 1
                    sum_agr_s += n_match / n_ovl
            # close inner mm_j
        return sum_agr_s / n_pairs_s if n_pairs_s > 0 else 0.0

    def mean_agreement_of_config(mm, sc_list):
        """Mean agreement of one config with the other 11 over sc_list."""
        others = [o for o in model_modes if o != mm]
        n_pairs_c = 0
        sum_agr_c = 0.0
        for o in others:
            n_match = 0
            n_ovl = 0
            for sc in sc_list:
                ci = mm_scenario_choice.get((mm, sc))
                cj = mm_scenario_choice.get((o, sc))
                if ci is None or cj is None:
                    continue
                n_ovl += 1
                if ci == cj:
                    n_match += 1
            if n_ovl > 0:
                n_pairs_c += 1
                sum_agr_c += n_match / n_ovl
        return sum_agr_c / n_pairs_c if n_pairs_c > 0 else 0.0

    def bootstrap_ci(sc_list, fn, n_boot=2000, alpha=0.05, seed=0):
        """Bootstrap percentile CI for fn(resampled_sc_list)."""
        rng = random.Random(seed)
        n = len(sc_list)
        stats = []
        for _ in range(n_boot):
            resample = [sc_list[rng.randrange(n)] for _ in range(n)]
            stats.append(fn(resample))
        stats.sort()
        lo = stats[int((alpha / 2) * n_boot)]
        hi = stats[int((1 - alpha / 2) * n_boot)]
        return lo, hi

    # --- 6a. 95% bootstrap CI on overall mean pairwise agreement
    print(f"\n  6a. Overall mean pairwise agreement: 95% bootstrap CI (2000 resamples over scenarios)")
    lo, hi = bootstrap_ci(scenarios, mean_pairwise_over, n_boot=2000, seed=1)
    print(f"      Observed: {population_mean:.4f}    95% CI: [{lo:.4f}, {hi:.4f}]    "
          f"chance = 0.5000 → {'CI excludes chance' if lo > 0.5 else 'CI includes chance'}")

    # --- 6b. SB-cluster vs DD-cluster: paired bootstrap on Δ
    print(f"\n  6b. SB-cluster vs DD-cluster mean pairwise agreement: paired bootstrap on Δ")
    sb_sc = [sc for sc in scenarios if parse_cluster(parse_problem_class(sc[0])) == "SB-type"]
    dd_sc = [sc for sc in scenarios if parse_cluster(parse_problem_class(sc[0])) == "DD-type"]
    sb_mean = mean_pairwise_over(sb_sc)
    dd_mean = mean_pairwise_over(dd_sc)
    # Bootstrap Δ
    rng = random.Random(2)
    n_boot = 2000
    diffs = []
    for _ in range(n_boot):
        rs_sb = [sb_sc[rng.randrange(len(sb_sc))] for _ in range(len(sb_sc))]
        rs_dd = [dd_sc[rng.randrange(len(dd_sc))] for _ in range(len(dd_sc))]
        diffs.append(mean_pairwise_over(rs_sb) - mean_pairwise_over(rs_dd))
    diffs.sort()
    d_lo = diffs[int(0.025 * n_boot)]
    d_hi = diffs[int(0.975 * n_boot)]
    # Two-sided permutation: pool, shuffle assignment to cluster, count |Δ| ≥ observed.
    pool = sb_sc + dd_sc
    rng = random.Random(3)
    obs_abs = abs(sb_mean - dd_mean)
    n_ge = 0
    n_perm = 2000
    for _ in range(n_perm):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        a_set = shuffled[:len(sb_sc)]
        b_set = shuffled[len(sb_sc):]
        if abs(mean_pairwise_over(a_set) - mean_pairwise_over(b_set)) >= obs_abs:
            n_ge += 1
    p_cluster = (n_ge + 1) / (n_perm + 1)
    print(f"      SB-type mean: {sb_mean:.4f}    DD-type mean: {dd_mean:.4f}    "
          f"Δ = {sb_mean - dd_mean:+.4f}")
    print(f"      95% bootstrap CI on Δ: [{d_lo:+.4f}, {d_hi:+.4f}]")
    print(f"      Permutation p-value (two-sided, cluster-label shuffle): {p_cluster:.4g}")

    # --- 6c. SB-cluster SSA-capability: is it different from the other SB slices?
    print(f"\n  6c. SB-cluster: SSA-capability vs (SIA-capability + attitudes) — paired bootstrap")
    sb_ssa_cap = [sc for sc in sb_sc if sc[4] == "ssa_capability"]
    sb_other = [sc for sc in sb_sc if sc[4] != "ssa_capability"]
    ssa_mean = mean_pairwise_over(sb_ssa_cap)
    other_mean = mean_pairwise_over(sb_other)
    rng = random.Random(4)
    pool = sb_ssa_cap + sb_other
    obs_abs = abs(ssa_mean - other_mean)
    n_ge = 0
    for _ in range(n_perm):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        a_set = shuffled[:len(sb_ssa_cap)]
        b_set = shuffled[len(sb_ssa_cap):]
        if abs(mean_pairwise_over(a_set) - mean_pairwise_over(b_set)) >= obs_abs:
            n_ge += 1
    p_sb = (n_ge + 1) / (n_perm + 1)
    print(f"      SB ssa_capability: {ssa_mean:.4f}    SB other q-types: {other_mean:.4f}    "
          f"Δ = {ssa_mean - other_mean:+.4f}")
    print(f"      Permutation p-value (two-sided): {p_sb:.4g}")

    # --- 6d. DD-cluster attitudes vs DD-cluster capability
    print(f"\n  6d. DD-cluster: attitudes vs capability slices — permutation test")
    dd_att = [sc for sc in dd_sc if sc[4] in ("personal_attitude", "normative_attitude")]
    dd_cap = [sc for sc in dd_sc if sc[4] in ("ssa_capability", "sia_capability")]
    att_mean = mean_pairwise_over(dd_att)
    cap_mean = mean_pairwise_over(dd_cap)
    rng = random.Random(5)
    pool = dd_att + dd_cap
    obs_abs = abs(att_mean - cap_mean)
    n_ge = 0
    for _ in range(n_perm):
        shuffled = pool[:]
        rng.shuffle(shuffled)
        a_set = shuffled[:len(dd_att)]
        b_set = shuffled[len(dd_att):]
        if abs(mean_pairwise_over(a_set) - mean_pairwise_over(b_set)) >= obs_abs:
            n_ge += 1
    p_dd = (n_ge + 1) / (n_perm + 1)
    print(f"      DD attitudes: {att_mean:.4f}    DD capability: {cap_mean:.4f}    "
          f"Δ = {att_mean - cap_mean:+.4f}")
    print(f"      Permutation p-value (two-sided): {p_dd:.4g}")

    # --- 6e. Per-configuration: bootstrap 95% CI on mean agreement; flag outliers
    print(f"\n  6e. Per-(model, mode) bootstrap 95% CI on mean pairwise agreement")
    print(f"      (CI computed by resampling scenarios; n_boot=2000)")
    print(f"\n      {'model':<32} {'mode':<5} {'mean':<8} {'95% CI':<22} {'vs chance':<10}")
    print("      " + "-" * 80)
    config_cis = {}
    for mm in model_modes:
        mu_obs = mean_agreement_of_config(mm, scenarios)
        lo, hi = bootstrap_ci(scenarios, lambda s, mm=mm: mean_agreement_of_config(mm, s),
                              n_boot=2000, seed=hash(mm) & 0xffff)
        config_cis[mm] = (mu_obs, lo, hi)
        flag = "above" if lo > 0.5 else ("includes 0.5" if hi >= 0.5 else "below")
        print(f"      {mm[0]:<32} {mm[1]:<5} {mu_obs:.4f}  [{lo:.4f}, {hi:.4f}]   {flag}")

    # --- 6f. Pairwise contrast: grok-off vs each non-outlier (Δ permutation)
    print(f"\n  6f. Outlier contrasts: grok-off and gpt-5.5-off vs claude-off (top consensus)")
    outliers = [("grok-4.3-20260430", "off"), ("gpt-5.5-20260423", "off"),
                ("gemini-3.1-pro-preview-20260219", "on")]
    reference = ("claude-4.7-opus-20260416", "off")
    for out in outliers:
        if out not in model_modes:
            continue
        mu_out = mean_agreement_of_config(out, scenarios)
        mu_ref = mean_agreement_of_config(reference, scenarios)
        # Per-scenario paired diff: for each scenario, mean agr of out with other 11 minus
        # mean agr of ref with other 11 — bootstrap CI on Δ.
        def per_scenario_diff(sc_list):
            return mean_agreement_of_config(out, sc_list) - mean_agreement_of_config(reference, sc_list)
        lo, hi = bootstrap_ci(scenarios, per_scenario_diff, n_boot=2000, seed=hash(out) & 0xffff)
        # Sign of CI tells significance at α=0.05 two-sided
        sig = "**" if (lo > 0 or hi < 0) else "ns"
        print(f"      {out[0]:<32} {out[1]:<3}  mean {mu_out:.4f}  vs ref {mu_ref:.4f}  "
              f"Δ = {mu_out - mu_ref:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  {sig}")

    # --- 6g. Top-30 composition: is the (cluster, q-type) imbalance significant?
    print(f"\n  6g. Top-30 controversial scenarios — composition vs uniform expectation")
    # Expected: each of (cluster × q-type) cell has 16/128 = 12.5% of scenarios.
    # Observed counts in top-30:
    obs_cell: Counter = Counter()
    for sc, _, _, _ in scenario_entropy[:30]:
        cl = parse_cluster(parse_problem_class(sc[0]))
        obs_cell[(cl, sc[4])] += 1
    # Each cell has 16 scenarios out of 128 total → expected fraction 16/128 = 0.125.
    # Expected count in top 30 = 30 * 0.125 = 3.75 per cell.
    chi2 = 0.0
    n_cells_g = 0
    for cl in ("SB-type", "DD-type"):
        for qt in qtypes:
            obs = obs_cell.get((cl, qt), 0)
            exp = 30 * 16 / 128
            chi2 += (obs - exp) ** 2 / exp
            n_cells_g += 1
    # df = 8 - 1 = 7
    df = n_cells_g - 1
    # Approximate chi-square upper tail via series — for clean output, just print stat + df
    # and use the rule-of-thumb cutoff (df=7, α=0.05 cutoff = 14.07; α=0.001 cutoff = 24.32).
    print(f"      χ²({df}) = {chi2:.2f}    (df={df}; nominal cutoffs: α=0.05→14.07, α=0.001→24.32)")
    print(f"      Observed cell counts (cluster × q-type):")
    for cl in ("SB-type", "DD-type"):
        for qt in qtypes:
            obs = obs_cell.get((cl, qt), 0)
            print(f"        {cl:<10} {qt:<25} obs={obs:<3} exp=3.75")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
