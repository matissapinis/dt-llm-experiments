#!/usr/bin/env python3
"""QT3 (user's variant) — Endorsement coherence.

Question: When a model endorses a framework's prescription in an attitude
question, does it actually know that framework's prescription (per the
capability question)?

For each (model, mode, problem_class, theme, param, row_order), aggregate:
  - Modal attitude letter across 18 attitude samples (9 personal + 9 normative).
  - Modal SIA-capability correctness across 9 SIA-cap samples.
  - Modal SSA-capability correctness across 9 SSA-cap samples.

Classify each problem-instance into:
  - Coherent SIA-endorser: attitude = SIA-aligned AND SIA-cap correct
  - Coherent SSA-endorser: attitude = SSA-aligned AND SSA-cap correct
  - Incoherent SIA-endorser: attitude = SIA-aligned BUT SIA-cap wrong
  - Incoherent SSA-endorser: attitude = SSA-aligned BUT SSA-cap wrong
  - Tied / unclassifiable: no clear attitude majority

Aggregate per (model, mode) and per cluster.

Uses the V1 (standard SSA) grader: SSA-correct = doomsday on DD/PADD,
halfer on SB/INC.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
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


def load_cells():
    cells = []
    for f in sorted(D.glob("*.json")):
        d = json.load(open(f))
        ch = d.get("extracted_choice")
        if ch not in ("A", "B"):
            continue
        pc, theme, param, row = parse_template(d.get("template_name", ""))
        if pc is None:
            continue
        sia_letter = get_sia_aligned_letter(d.get("preferred_actions"), d.get("row_order"))
        if sia_letter is None:
            continue
        model = (d.get("model_id_openrouter") or "").split("/")[-1]
        d["mode"] = parse_mode(f.name)
        d["model_short"] = model
        d["problem_class"] = pc
        d["theme"] = theme
        d["param"] = param
        d["row"] = row
        d["cluster"] = parse_cluster(pc)
        d["sia_letter"] = sia_letter
        d["picked_sia_aligned"] = (ch == sia_letter)
        cells.append(d)
    return cells


def modal(lst):
    """Return modal value or None if tied/empty."""
    if not lst:
        return None
    c = Counter(lst)
    most = c.most_common()
    if len(most) > 1 and most[0][1] == most[1][1]:
        return None  # tie
    return most[0][0]


def main() -> int:
    cells = load_cells()
    print(f"Loaded {len(cells)} cells")

    # Group by (model_mode, problem_instance, q_type)
    # problem_instance = (problem_class, theme, param, row)
    # Within each (mm, pi, qt), pool samples
    by_key: dict = defaultdict(list)
    for c in cells:
        key = ((c["model_short"], c["mode"]),
               (c["problem_class"], c["theme"], c["param"], c["row"]),
               c.get("question_type"))
        by_key[key].append(c)

    # For each (mm, problem_instance), aggregate:
    #   - modal attitude letter across personal + normative
    #   - modal capability correctness for ssa and sia
    instance_classification: dict = defaultdict(list)  # (mm) → list of classifications
    by_inst: dict = defaultdict(dict)  # (mm, pi) → {q_type: [bool/letter list]}
    for (mm, pi, qt), lst in by_key.items():
        by_inst[(mm, pi)][qt] = lst

    classifications = []
    for (mm, pi), qdict in by_inst.items():
        personal = qdict.get("personal_attitude", [])
        normative = qdict.get("normative_attitude", [])
        sia_cap = qdict.get("sia_capability", [])
        ssa_cap = qdict.get("ssa_capability", [])
        if not (personal or normative) or not sia_cap or not ssa_cap:
            continue
        # Modal attitude letter (across pooled attitude samples)
        att_samples = personal + normative
        att_picks = [c["picked_sia_aligned"] for c in att_samples]
        # Convert to "SIA-aligned" or "SSA-aligned"
        n_sia_aligned = sum(att_picks)
        n_ssa_aligned = len(att_picks) - n_sia_aligned
        if n_sia_aligned > n_ssa_aligned:
            modal_attitude = "SIA-aligned"
        elif n_ssa_aligned > n_sia_aligned:
            modal_attitude = "SSA-aligned"
        else:
            modal_attitude = "TIED"
        # Modal capability correctness
        # SIA-cap correct = picked SIA-aligned letter
        sia_cap_correct_majority = (sum(c["picked_sia_aligned"] for c in sia_cap) > len(sia_cap) / 2)
        # SSA-cap correct (V1, standard) = picked NON-SIA-aligned letter
        ssa_cap_correct_majority = (sum(not c["picked_sia_aligned"] for c in ssa_cap) > len(ssa_cap) / 2)

        if modal_attitude == "TIED":
            classification = "tied_attitude"
        elif modal_attitude == "SIA-aligned":
            classification = "coherent_SIA" if sia_cap_correct_majority else "incoherent_SIA"
        else:  # SSA-aligned
            classification = "coherent_SSA" if ssa_cap_correct_majority else "incoherent_SSA"

        classifications.append({
            "mm": mm, "pi": pi, "cluster": parse_cluster(pi[0]),
            "modal_attitude": modal_attitude,
            "sia_cap_correct": sia_cap_correct_majority,
            "ssa_cap_correct": ssa_cap_correct_majority,
            "classification": classification,
            "n_att": len(att_samples),
            "n_sia_aligned": n_sia_aligned,
            "n_ssa_aligned": n_ssa_aligned,
        })

    print(f"\nTotal problem-instances classified: {len(classifications)}")
    print(f"(expected: 12 model-modes × 16 instances [4 pc × 2 theme × 2 param × 2 row] = 192)")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 1: Per (model, mode) classification counts")
    print(f"{'=' * 100}")
    by_mm = defaultdict(Counter)
    for c in classifications:
        by_mm[c["mm"]][c["classification"]] += 1

    print(f"\n  {'model':<32} {'mode':<5} {'coh-SIA':<9} {'coh-SSA':<9} "
          f"{'incoh-SIA':<11} {'incoh-SSA':<11} {'tied':<6} {'total':<6} "
          f"{'% incoherent':<14}")
    print("  " + "-" * 110)
    for mm in sorted(by_mm.keys()):
        c = by_mm[mm]
        n_total = sum(c.values())
        n_incoh = c["incoherent_SIA"] + c["incoherent_SSA"]
        pct_incoh = 100 * n_incoh / n_total if n_total else 0
        print(f"  {mm[0]:<32} {mm[1]:<5} {c['coherent_SIA']:<9} {c['coherent_SSA']:<9} "
              f"{c['incoherent_SIA']:<11} {c['incoherent_SSA']:<11} "
              f"{c['tied_attitude']:<6} {n_total:<6} {pct_incoh:<14.1f}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 2: Per (model, mode, cluster) classification")
    print(f"{'=' * 100}")
    by_mm_cl = defaultdict(Counter)
    for c in classifications:
        by_mm_cl[(c["mm"], c["cluster"])][c["classification"]] += 1
    for mm in sorted({c["mm"] for c in classifications}):
        for cl in ("SB-type", "DD-type"):
            cnt = by_mm_cl.get((mm, cl), Counter())
            n_total = sum(cnt.values())
            if n_total == 0:
                continue
            n_incoh = cnt["incoherent_SIA"] + cnt["incoherent_SSA"]
            n_coh = cnt["coherent_SIA"] + cnt["coherent_SSA"]
            pct_incoh = 100 * n_incoh / n_total if n_total else 0
            print(f"  {mm[0]:<32} {mm[1]:<5} {cl:<10} "
                  f"coh={n_coh:<3} incoh={n_incoh:<3} tied={cnt['tied_attitude']:<3} "
                  f"({pct_incoh:.0f}% incoh)")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 3: Population-pooled by cluster")
    print(f"{'=' * 100}")
    by_cl = defaultdict(Counter)
    for c in classifications:
        by_cl[c["cluster"]][c["classification"]] += 1
    for cl in ("SB-type", "DD-type"):
        cnt = by_cl[cl]
        n_total = sum(cnt.values())
        n_incoh = cnt["incoherent_SIA"] + cnt["incoherent_SSA"]
        n_coh = cnt["coherent_SIA"] + cnt["coherent_SSA"]
        print(f"\n  {cl}: {n_total} instances")
        print(f"    Coherent SIA-endorser:   {cnt['coherent_SIA']} ({100*cnt['coherent_SIA']/n_total:.1f}%)")
        print(f"    Coherent SSA-endorser:   {cnt['coherent_SSA']} ({100*cnt['coherent_SSA']/n_total:.1f}%)")
        print(f"    Incoherent SIA-endorser: {cnt['incoherent_SIA']} ({100*cnt['incoherent_SIA']/n_total:.1f}%)")
        print(f"    Incoherent SSA-endorser: {cnt['incoherent_SSA']} ({100*cnt['incoherent_SSA']/n_total:.1f}%)")
        print(f"    Tied attitude:           {cnt['tied_attitude']} ({100*cnt['tied_attitude']/n_total:.1f}%)")
        print(f"    → Overall % incoherent (endorses what it can't state): "
              f"{100*n_incoh/n_total:.1f}%")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 4: Stratified by (model, cluster, q-type-of-attitude)")
    print("           Does the pattern differ for personal vs normative attitudes?")
    print(f"{'=' * 100}")
    # Re-classify but separate personal and normative as different attitude groups
    classifications_split = []
    for (mm, pi), qdict in by_inst.items():
        sia_cap = qdict.get("sia_capability", [])
        ssa_cap = qdict.get("ssa_capability", [])
        if not sia_cap or not ssa_cap:
            continue
        sia_cap_correct_majority = (sum(c["picked_sia_aligned"] for c in sia_cap) > len(sia_cap) / 2)
        ssa_cap_correct_majority = (sum(not c["picked_sia_aligned"] for c in ssa_cap) > len(ssa_cap) / 2)
        for att_qt in ("personal_attitude", "normative_attitude"):
            att_samples = qdict.get(att_qt, [])
            if not att_samples:
                continue
            n_sia_aligned = sum(c["picked_sia_aligned"] for c in att_samples)
            n_ssa_aligned = len(att_samples) - n_sia_aligned
            if n_sia_aligned > n_ssa_aligned:
                modal_attitude = "SIA-aligned"
            elif n_ssa_aligned > n_sia_aligned:
                modal_attitude = "SSA-aligned"
            else:
                modal_attitude = "TIED"
            if modal_attitude == "TIED":
                cls = "tied_attitude"
            elif modal_attitude == "SIA-aligned":
                cls = "coherent_SIA" if sia_cap_correct_majority else "incoherent_SIA"
            else:
                cls = "coherent_SSA" if ssa_cap_correct_majority else "incoherent_SSA"
            classifications_split.append({"mm": mm, "cluster": parse_cluster(pi[0]),
                                          "att_qt": att_qt, "classification": cls})

    by_attqt_cl = defaultdict(Counter)
    for c in classifications_split:
        by_attqt_cl[(c["att_qt"], c["cluster"])][c["classification"]] += 1
    print(f"\n  {'att q-type':<22} {'cluster':<10} {'coh-SIA':<9} {'coh-SSA':<9} "
          f"{'incoh-SIA':<11} {'incoh-SSA':<11} {'% incoh':<8}")
    print("  " + "-" * 90)
    for att_qt in ("personal_attitude", "normative_attitude"):
        for cl in ("SB-type", "DD-type"):
            cnt = by_attqt_cl[(att_qt, cl)]
            n_total = sum(cnt.values())
            n_incoh = cnt["incoherent_SIA"] + cnt["incoherent_SSA"]
            pct_incoh = 100 * n_incoh / n_total if n_total else 0
            print(f"  {att_qt:<22} {cl:<10} {cnt['coherent_SIA']:<9} {cnt['coherent_SSA']:<9} "
                  f"{cnt['incoherent_SIA']:<11} {cnt['incoherent_SSA']:<11} {pct_incoh:<8.1f}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 5: Sample 'incoherent' instances — what does this look like?")
    print(f"{'=' * 100}")
    incoherent = [c for c in classifications
                  if c["classification"] in ("incoherent_SIA", "incoherent_SSA")]
    print(f"\n  Total incoherent instances: {len(incoherent)}")
    if incoherent:
        import random
        random.seed(42)
        sample = random.sample(incoherent, min(10, len(incoherent)))
        for i, c in enumerate(sample, 1):
            print(f"\n  [{i}] {c['mm'][0]:<32} {c['mm'][1]} | pc={c['pi'][0]} theme={c['pi'][1]} "
                  f"param={c['pi'][2]} row={c['pi'][3]}")
            print(f"      attitude: {c['n_sia_aligned']}/{c['n_att']} SIA-aligned, "
                  f"{c['n_ssa_aligned']}/{c['n_att']} SSA-aligned → modal: {c['modal_attitude']}")
            print(f"      sia-cap correct: {c['sia_cap_correct']}    "
                  f"ssa-cap correct: {c['ssa_cap_correct']}")
            print(f"      classification: {c['classification']}")
            print(f"      interpretation: model endorses {c['modal_attitude']} prescription in attitudes,")
            ep = "SIA" if c["classification"] == "incoherent_SIA" else "SSA"
            cap_correct = c["sia_cap_correct"] if ep == "SIA" else c["ssa_cap_correct"]
            print(f"                      but {ep}-capability shows it doesn't reliably state "
                  f"{ep}'s prescription (correct={cap_correct}).")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 6: Inferential tests on the coherence claims")
    print(f"{'=' * 100}")

    import math
    import random as _random

    def wilson_ci(k, n, z=1.96):
        if n == 0:
            return (0.0, 0.0)
        p = k / n
        den = 1 + z**2 / n
        center = (p + z**2 / (2 * n)) / den
        half = (z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))) / den
        return (max(0.0, center - half), min(1.0, center + half))

    def chi2_2x2(a, b, c, d):
        n = a + b + c + d
        if n == 0:
            return (0.0, 1.0)
        row1, row2 = a + b, c + d
        col1, col2 = a + c, b + d
        e = [row1 * col1 / n, row1 * col2 / n, row2 * col1 / n, row2 * col2 / n]
        chi2 = sum((o - x) ** 2 / x for o, x in zip([a, b, c, d], e) if x > 0)
        return (chi2, math.erfc(math.sqrt(chi2 / 2)))

    # ----- 6a. Observed coherence vs binomial null
    n_classified = sum(1 for c in classifications if c["classification"] != "tied_attitude")
    n_coherent = sum(1 for c in classifications
                     if c["classification"] in ("coherent_SIA", "coherent_SSA"))
    n_incoherent = n_classified - n_coherent
    print(f"\n  6a. Wilson 95% CI on coherent rate:")
    lo, hi = wilson_ci(n_coherent, n_classified)
    print(f"      Coherent: {n_coherent}/{n_classified} = {n_coherent/n_classified:.4f}  "
          f"95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"      Incoherent: {n_incoherent}/{n_classified} = {n_incoherent/n_classified:.4f}")

    # ----- 6b. Permutation test: shuffle attitude labels within each (mm, cluster),
    # re-compute incoherence rate, compare to observed.
    print(f"\n  6b. Permutation test: shuffle attitude direction within (mm, cluster)")
    print(f"      Null: attitude direction is independent of capability correctness")

    # Build per (mm, cluster) lists of:
    #   (attitude_direction, sia_cap_correct, ssa_cap_correct)
    # Then shuffle attitude_direction; recompute incoherence.
    per_mmcl: dict = defaultdict(list)
    for c in classifications:
        if c["modal_attitude"] == "TIED":
            continue
        per_mmcl[(c["mm"], c["cluster"])].append(
            (c["modal_attitude"], c["sia_cap_correct"], c["ssa_cap_correct"])
        )

    def compute_incoherent(records):
        n_inc = 0
        for att, sc, sscc in records:
            if att == "SIA-aligned" and not sc:
                n_inc += 1
            elif att == "SSA-aligned" and not sscc:
                n_inc += 1
        return n_inc

    obs_inc_total = sum(compute_incoherent(lst) for lst in per_mmcl.values())
    n_perm = 5000
    rng = _random.Random(42)
    perm_inc_counts = []
    for _ in range(n_perm):
        total = 0
        for (mmcl, lst) in per_mmcl.items():
            atts = [r[0] for r in lst]
            rng.shuffle(atts)
            shuffled = [(atts[i], lst[i][1], lst[i][2]) for i in range(len(lst))]
            total += compute_incoherent(shuffled)
        perm_inc_counts.append(total)
    perm_inc_counts.sort()
    # One-sided: P(perm incoherence ≤ observed)
    n_le = sum(1 for x in perm_inc_counts if x <= obs_inc_total)
    p_one_sided = (n_le + 1) / (n_perm + 1)
    perm_mean = sum(perm_inc_counts) / len(perm_inc_counts)
    print(f"      Observed incoherent count: {obs_inc_total}")
    print(f"      Permutation distribution (n={n_perm}): mean={perm_mean:.2f}, "
          f"5%={perm_inc_counts[int(0.05*n_perm)]}, 95%={perm_inc_counts[int(0.95*n_perm)]}")
    print(f"      One-sided p (P[perm ≤ obs]): {p_one_sided:.4g}")
    print(f"      → {'observed is significantly LESS incoherent than chance' if p_one_sided < 0.05 else 'no significant difference from chance'}")

    # ----- 6c. SB vs DD: difference in SIA-endorsement rate
    print(f"\n  6c. Cluster contrast: SIA-endorsement rate SB vs DD")
    n_sb_sia = sum(1 for c in classifications
                   if c["cluster"] == "SB-type" and c["modal_attitude"] == "SIA-aligned")
    n_sb_total = sum(1 for c in classifications
                     if c["cluster"] == "SB-type" and c["modal_attitude"] != "TIED")
    n_dd_sia = sum(1 for c in classifications
                   if c["cluster"] == "DD-type" and c["modal_attitude"] == "SIA-aligned")
    n_dd_total = sum(1 for c in classifications
                     if c["cluster"] == "DD-type" and c["modal_attitude"] != "TIED")
    p_sb = n_sb_sia / n_sb_total if n_sb_total else 0
    p_dd = n_dd_sia / n_dd_total if n_dd_total else 0
    chi2, p_chi = chi2_2x2(n_sb_sia, n_sb_total - n_sb_sia, n_dd_sia, n_dd_total - n_dd_sia)
    print(f"      SB SIA-endorsement: {n_sb_sia}/{n_sb_total} = {p_sb:.4f}")
    print(f"      DD SIA-endorsement: {n_dd_sia}/{n_dd_total} = {p_dd:.4f}")
    print(f"      χ² = {chi2:.2f}, p = {p_chi:.4g}")

    # ----- 6d. Within-DD: SSA-endorsement vs 50/50
    print(f"\n  6d. DD-cluster: is SSA-endorsement different from 50/50?")
    n_dd_ssa = n_dd_total - n_dd_sia
    # Binomial test against H0 p=0.5
    def binomial_two_sided_p(k, n, p0=0.5):
        if n == 0:
            return 1.0
        pmf = [math.comb(n, i) * (p0**i) * ((1-p0)**(n-i)) for i in range(n+1)]
        p_obs = pmf[k]
        return min(1.0, sum(pi for pi in pmf if pi <= p_obs + 1e-15))
    p_50 = binomial_two_sided_p(n_dd_sia, n_dd_total, 0.5)
    print(f"      DD SIA-endorsement: {n_dd_sia}/{n_dd_total} ({p_dd:.4f})")
    print(f"      DD SSA-endorsement: {n_dd_ssa}/{n_dd_total} ({1-p_dd:.4f})")
    print(f"      Binomial two-sided p (vs 0.5): {p_50:.4g}")

    # ----- 6e. Per-model: is each model's coherence rate significantly > 50%?
    print(f"\n  6e. Per (model, mode) coherence rates with Wilson CIs:")
    print(f"      {'model':<32} {'mode':<5} {'coherent rate':<16} {'95% CI':<22} "
          f"{'binom p (vs 0.5)':<18}")
    print("      " + "-" * 95)
    for mm in sorted(by_mm.keys()):
        c = by_mm[mm]
        n_classified_mm = sum(c.values()) - c["tied_attitude"]
        n_coh_mm = c["coherent_SIA"] + c["coherent_SSA"]
        rate = n_coh_mm / n_classified_mm if n_classified_mm else 0
        lo, hi = wilson_ci(n_coh_mm, n_classified_mm)
        p_b = binomial_two_sided_p(n_coh_mm, n_classified_mm, 0.5)
        print(f"      {mm[0]:<32} {mm[1]:<5} {n_coh_mm}/{n_classified_mm} = {rate:.4f}    "
              f"[{lo:.3f}, {hi:.3f}]    {p_b:.4g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
