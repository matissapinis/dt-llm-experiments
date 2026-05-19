#!/usr/bin/env python3
"""QT8 + QT9 — Information-leakage citation patterns + SSA/SIA capability asymmetry.

QT8: For each (problem_class, model, mode), count traces that cite named
anthropic-reasoning literature (Sleeping Beauty / Elga / Bostrom / Doomsday
Argument / Carter / Gott / Presumptuous Philosopher / Simulation Argument /
specific paper titles). Tests the hypothesis: SB has the highest citation
rate, PADD the lowest — consistent with training-data frequency. Per-cluster
and per-problem-class comparisons with significance tests.

QT9: For each (model, mode, problem_instance) where the model has at least 5
SIA-cap and 5 SSA-cap samples, test paired difference: P(SIA-cap correct) vs
P(SSA-cap correct). Population-pooled and per-(model, mode) McNemar-like
tests. Tests the hypothesis: differential training-data salience between SSA
and SIA shows up as differential accuracy.

Note: V1 grader for capability correctness (the standard SSA per cluster:
doomsday for DD/PADD, halfer for SB/INC).
"""
from __future__ import annotations

import json
import math
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


def parse_template_instance(template_name: str):
    """(pc, theme, param, row) — the 4-tuple identifying a problem instance."""
    m = re.match(r"\d+_standard_(sb|inc|dd|padd)_(.+?)(_scaled)?_(12|21)$", template_name or "")
    if not m:
        return None
    pc, theme, scaled, row = m.groups()
    return (pc, theme, "scaled" if scaled else "canonical", row)


def get_sia_aligned_letter(preferred_actions: dict, row_order: str):
    if not preferred_actions:
        return None
    sia_pref = preferred_actions.get("sia_preference")
    if not sia_pref:
        return None
    is_A_in_row12 = sia_pref in ("half", "high")
    return ("A" if is_A_in_row12 else "B") if row_order == "12" else ("B" if is_A_in_row12 else "A")


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


def mcnemar_p(b, c):
    """Two-sided McNemar test on discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    smaller = min(b, c)
    cdf = sum(math.comb(n, i) for i in range(0, smaller + 1)) * (0.5 ** n)
    return min(1.0, 2 * cdf)


# ----- QT8 citation patterns -----
# Direct literature mentions (specific named problems, authors, paper titles)
LITERATURE_CITATION_PATTERNS = [
    # Named problems
    (r"\bsleeping\s+beauty\b", "sleeping-beauty"),
    (r"\bdoomsday\s+argument\b", "doomsday-argument"),
    (r"\bpresumptuous\s+philosopher\b", "presumptuous-philosopher"),
    (r"\bsimulation\s+(argument|hypothesis)\b", "simulation-argument"),
    (r"\bincubator\s+(problem|gedanken|thought\s+experiment)\b", "incubator-problem"),
    (r"\bgod['’]?s\s+coin\s+toss\b", "gods-coin-toss"),
    (r"\bmonty\s+hall\b", "monty-hall"),
    (r"\btwo[-\s]urn\s+problem\b", "two-urn"),
    (r"\bbertrand['’]?s?\s+box\b", "bertrand-box"),
    # Authors
    (r"\bbostrom\b", "Bostrom"),
    (r"\belga\b", "Elga"),
    (r"\bleslie\b", "Leslie"),
    (r"\bcarter\b", "Carter"),
    (r"\bgott\b", "Gott"),
    (r"\b(neal\b|radford\s+neal)", "Neal"),
    (r"\b(yudkowsky|eliezer)\b", "Yudkowsky"),
    (r"\bconitzer\b", "Conitzer"),
    (r"\bolum\b", "Olum"),
    (r"\bchalmers\b", "Chalmers"),
    (r"\blewis\b", "Lewis"),
    (r"\badelstein\b", "Adelstein"),
    # Book / paper titles
    (r"\banthropic\s+bias\b", "AnthropicBias-book"),
    (r"\banthropic\s+decision\s+theory\b", "ADT-paper"),
    (r"\bgreat\s+filter\b", "great-filter"),
    # Specific named frameworks (broader; partial overlap with QT1c)
    (r"\bFNC\b|\bfull\s+non[-\s]indexical\s+conditioning\b", "FNC"),
    (r"\bUDT\b|\bupdateless\s+decision\s+theory\b", "UDT"),
]


def has_any_match(text, patterns):
    text_l = text.lower()
    for pat, _ in patterns:
        if re.search(pat, text_l, flags=re.IGNORECASE):
            return True
    return False


def find_matches(text, patterns):
    text_l = text.lower()
    labels = set()
    for pat, label in patterns:
        if re.search(pat, text_l, flags=re.IGNORECASE):
            labels.add(label)
    return labels


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
        d["instance"] = parse_template_instance(d.get("template_name", ""))
        d["is_thirder"] = (ch == sia_letter)
        qt = d.get("question_type")
        if qt == "sia_capability":
            d["is_correct"] = d["is_thirder"]
        elif qt == "ssa_capability":
            d["is_correct"] = not d["is_thirder"]
        else:
            d["is_correct"] = None
        rt = d.get("reasoning_trace") or ""
        resp = d.get("response") or ""
        d["_trace"] = rt if rt.strip() else resp
        cells.append(d)
    return cells


def main() -> int:
    cells = load_cells()
    print(f"Loaded {len(cells)} cells\n")

    # =================================================================
    # QT8: literature citation rate per problem class
    # =================================================================
    print(f"\n{'=' * 100}")
    print("QT8: LITERATURE CITATION rate per problem class")
    print("     (Sleeping Beauty / Doomsday Argument / Bostrom / Elga / etc. named in trace)")
    print(f"{'=' * 100}")

    # Per-cell: does it cite ≥1 literature item?
    cells_by_pc = defaultdict(list)
    for c in cells:
        cells_by_pc[c["problem_class"]].append(c)

    print(f"\n  Population-pooled by problem class:")
    print(f"  {'problem class':<8} {'cells':<8} {'with ≥1 citation':<18} {'%':<8} "
          f"{'Wilson 95% CI':<22}")
    print("  " + "-" * 75)
    pc_results = {}
    for pc in ("sb", "inc", "dd", "padd"):
        lst = cells_by_pc.get(pc, [])
        n_total = len(lst)
        n_cite = sum(1 for c in lst if has_any_match(c["_trace"], LITERATURE_CITATION_PATTERNS))
        rate = n_cite / n_total if n_total else 0
        lo, hi = wilson_ci(n_cite, n_total)
        pc_results[pc] = (n_cite, n_total)
        print(f"  {pc:<8} {n_total:<8} {n_cite:<18} {rate*100:<7.2f}% [{lo*100:.2f}, {hi*100:.2f}]")

    # Hypothesis test: SB > PADD?
    print(f"\n  Headline test: SB citation rate vs PADD citation rate")
    sb_cite, sb_total = pc_results["sb"]
    padd_cite, padd_total = pc_results["padd"]
    chi2, p = chi2_2x2(sb_cite, sb_total - sb_cite, padd_cite, padd_total - padd_cite)
    print(f"  SB:   {sb_cite}/{sb_total} = {sb_cite/sb_total*100:.2f}%")
    print(f"  PADD: {padd_cite}/{padd_total} = {padd_cite/padd_total*100:.2f}%")
    print(f"  Δ = {(sb_cite/sb_total - padd_cite/padd_total)*100:+.2f}pp")
    print(f"  χ² = {chi2:.2f}, p = {p:.4g}")

    # All pairwise problem-class comparisons
    print(f"\n  All pairwise (problem class) citation-rate contrasts:")
    print(f"  {'pair':<14} {'Δ (pp)':<10} {'p':<12}")
    print("  " + "-" * 45)
    pcs = ["sb", "inc", "dd", "padd"]
    for i in range(len(pcs)):
        for j in range(i + 1, len(pcs)):
            a_cite, a_total = pc_results[pcs[i]]
            b_cite, b_total = pc_results[pcs[j]]
            delta_pp = (a_cite/a_total - b_cite/b_total) * 100
            _, p_val = chi2_2x2(a_cite, a_total - a_cite, b_cite, b_total - b_cite)
            print(f"  {pcs[i]+' vs '+pcs[j]:<14} {delta_pp:+8.2f}   {p_val:.4g}")

    # By specific framework cited, per problem class
    print(f"\n  Top citation labels per problem class (which named items appear most):")
    label_by_pc = defaultdict(Counter)
    for c in cells:
        labels = find_matches(c["_trace"], LITERATURE_CITATION_PATTERNS)
        for lab in labels:
            label_by_pc[c["problem_class"]][lab] += 1
    for pc in ("sb", "inc", "dd", "padd"):
        print(f"\n    {pc} (n={pc_results[pc][1]}):")
        for label, count in label_by_pc[pc].most_common(10):
            rate = count / pc_results[pc][1] * 100
            print(f"      {label:<28} {count:<5} ({rate:.2f}%)")

    # Per (model, mode) citation rate broken down by cluster
    print(f"\n  Per (model, mode, cluster) citation rate:")
    print(f"  {'model':<32} {'mode':<5} {'SB cluster':<14} {'DD cluster':<14} {'Δ (SB-DD)':<10}")
    print("  " + "-" * 80)
    mm_cluster: dict = defaultdict(lambda: defaultdict(lambda: {"cite": 0, "total": 0}))
    for c in cells:
        mm_cluster[(c["model_short"], c["mode"])][c["cluster"]]["total"] += 1
        if has_any_match(c["_trace"], LITERATURE_CITATION_PATTERNS):
            mm_cluster[(c["model_short"], c["mode"])][c["cluster"]]["cite"] += 1
    for mm in sorted(mm_cluster.keys()):
        sb = mm_cluster[mm]["SB-type"]
        dd = mm_cluster[mm]["DD-type"]
        sb_rate = sb["cite"] / sb["total"] if sb["total"] else 0
        dd_rate = dd["cite"] / dd["total"] if dd["total"] else 0
        print(f"  {mm[0]:<32} {mm[1]:<5} "
              f"{sb_rate*100:6.2f}% ({sb['cite']:>3}/{sb['total']:>3})  "
              f"{dd_rate*100:6.2f}% ({dd['cite']:>3}/{dd['total']:>3})  "
              f"{(sb_rate - dd_rate)*100:+6.2f}pp")

    # =================================================================
    # QT9: SSA-cap vs SIA-cap accuracy asymmetry
    # =================================================================
    print(f"\n\n{'=' * 100}")
    print("QT9: SSA-cap vs SIA-cap accuracy ASYMMETRY")
    print("     V1 grader (doomsday-SSA per cluster for SSA-cap; SIA-cancellation for SIA-cap)")
    print(f"{'=' * 100}")

    # Per (model, mode), pool capability cells by q-type
    by_mm_qt: dict = defaultdict(lambda: defaultdict(list))
    for c in cells:
        qt = c.get("question_type")
        if qt not in ("sia_capability", "ssa_capability"):
            continue
        by_mm_qt[(c["model_short"], c["mode"])][qt].append(c)

    print(f"\n  Per (model, mode) capability accuracy by q-type (paired contrast):")
    print(f"  {'model':<32} {'mode':<5} {'SIA-cap':<18} {'SSA-cap':<18} {'Δ (SIA-SSA)':<12} "
          f"{'χ² p':<10}")
    print("  " + "-" * 100)
    for mm in sorted(by_mm_qt.keys()):
        sia_cells = by_mm_qt[mm].get("sia_capability", [])
        ssa_cells = by_mm_qt[mm].get("ssa_capability", [])
        n_sia = len(sia_cells)
        n_ssa = len(ssa_cells)
        n_sia_correct = sum(1 for c in sia_cells if c["is_correct"])
        n_ssa_correct = sum(1 for c in ssa_cells if c["is_correct"])
        if n_sia == 0 or n_ssa == 0:
            continue
        sia_rate = n_sia_correct / n_sia
        ssa_rate = n_ssa_correct / n_ssa
        delta = sia_rate - ssa_rate
        _, p = chi2_2x2(n_sia_correct, n_sia - n_sia_correct,
                         n_ssa_correct, n_ssa - n_ssa_correct)
        print(f"  {mm[0]:<32} {mm[1]:<5} "
              f"{sia_rate*100:6.2f}% ({n_sia_correct:>3}/{n_sia:>3})  "
              f"{ssa_rate*100:6.2f}% ({n_ssa_correct:>3}/{n_ssa:>3})  "
              f"{delta*100:+7.2f}pp   {p:<10.4g}")

    # Population-pooled
    n_sia_total = sum(1 for c in cells if c.get("question_type") == "sia_capability")
    n_sia_correct = sum(1 for c in cells if c.get("question_type") == "sia_capability" and c["is_correct"])
    n_ssa_total = sum(1 for c in cells if c.get("question_type") == "ssa_capability")
    n_ssa_correct = sum(1 for c in cells if c.get("question_type") == "ssa_capability" and c["is_correct"])
    print(f"\n  Population-pooled:")
    print(f"    SIA-cap accuracy: {n_sia_correct}/{n_sia_total} = {n_sia_correct/n_sia_total*100:.2f}%")
    print(f"    SSA-cap accuracy: {n_ssa_correct}/{n_ssa_total} = {n_ssa_correct/n_ssa_total*100:.2f}%")
    print(f"    Δ (SIA-SSA) = {(n_sia_correct/n_sia_total - n_ssa_correct/n_ssa_total)*100:+.2f}pp")
    _, p_pop = chi2_2x2(n_sia_correct, n_sia_total - n_sia_correct,
                          n_ssa_correct, n_ssa_total - n_ssa_correct)
    print(f"    χ² p (pooled) = {p_pop:.4g}")

    # By cluster
    print(f"\n  By cluster (SB-type vs DD-type) — does asymmetry differ by cluster?")
    print(f"  {'cluster':<10} {'SIA-cap acc':<16} {'SSA-cap acc':<16} {'Δ':<10} {'p':<10}")
    print("  " + "-" * 70)
    for cl in ("SB-type", "DD-type"):
        sia_lst = [c for c in cells if c["cluster"] == cl and c.get("question_type") == "sia_capability"]
        ssa_lst = [c for c in cells if c["cluster"] == cl and c.get("question_type") == "ssa_capability"]
        n_sia_c = sum(1 for c in sia_lst if c["is_correct"])
        n_ssa_c = sum(1 for c in ssa_lst if c["is_correct"])
        sia_rate = n_sia_c / len(sia_lst)
        ssa_rate = n_ssa_c / len(ssa_lst)
        _, p = chi2_2x2(n_sia_c, len(sia_lst) - n_sia_c, n_ssa_c, len(ssa_lst) - n_ssa_c)
        print(f"  {cl:<10} {sia_rate*100:6.2f}% ({n_sia_c}/{len(sia_lst)})  "
              f"{ssa_rate*100:6.2f}% ({n_ssa_c}/{len(ssa_lst)})  "
              f"{(sia_rate - ssa_rate)*100:+6.2f}pp   {p:<10.4g}")

    # Paired McNemar at instance level
    print(f"\n  McNemar's-like paired test per (model, mode):")
    print(f"  For each problem-instance, take modal SIA-cap correct and modal SSA-cap correct;")
    print(f"  pairs are discordant if one correct, other not.")
    print(f"  {'model':<32} {'mode':<5} {'concordant':<12} {'SIA✓ SSA✗':<12} {'SIA✗ SSA✓':<12} "
          f"{'McNemar p':<10}")
    print("  " + "-" * 100)
    for mm in sorted(by_mm_qt.keys()):
        # Group by problem_instance
        instances: dict = defaultdict(dict)
        for qt in ("sia_capability", "ssa_capability"):
            for c in by_mm_qt[mm].get(qt, []):
                inst = c["instance"]
                if inst is None:
                    continue
                instances[inst].setdefault(qt, []).append(c["is_correct"])
        concordant = 0
        b_sia_only = 0  # SIA correct, SSA wrong
        c_ssa_only = 0  # SSA correct, SIA wrong
        both_wrong = 0
        for inst, dd_ in instances.items():
            if "sia_capability" not in dd_ or "ssa_capability" not in dd_:
                continue
            sia_modal = sum(dd_["sia_capability"]) > len(dd_["sia_capability"]) / 2
            ssa_modal = sum(dd_["ssa_capability"]) > len(dd_["ssa_capability"]) / 2
            if sia_modal and ssa_modal:
                concordant += 1
            elif sia_modal and not ssa_modal:
                b_sia_only += 1
            elif not sia_modal and ssa_modal:
                c_ssa_only += 1
            else:
                both_wrong += 1
        mc_p = mcnemar_p(b_sia_only, c_ssa_only)
        print(f"  {mm[0]:<32} {mm[1]:<5} "
              f"{concordant:<12} {b_sia_only:<12} {c_ssa_only:<12} {mc_p:<10.4g}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
