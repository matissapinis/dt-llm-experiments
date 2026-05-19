#!/usr/bin/env python3
"""RQ1 — Population-level halfer/thirder preference on attitude questions.

Pre-registered primary hypothesis (two-sided):
  For each (problem, q-type), the population proportion choosing the "thirder"
  (SIA-aligned) option differs from 0.5 chance.

Operational definition of "thirder choice":
  matches sia_preference per the problem's row_order mapping
  (= SIA-aligned letter, by symmetry with our earlier SSA-aligned analyses).

Data subset: attitude cells (normative + personal) with parsed extracted_choice
and known SIA-letter mapping. Pooled across all 12 (model, mode) configurations.

Tests:
  - Per (problem, q-type) binomial test vs 0.5 (two-sided). 8 tests.
  - Bonferroni within RQ1: α = 0.05 / 8 = 0.00625
  - Plus per-cluster pooled and population-level pooled summaries.

Note: per the pluralism deep-dive, the "halfer/thirder" framing aggregated
across all problems averages out cluster-specific patterns. We report
per-problem first, then cluster-pooled, then global.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

D = Path("experiment_results/main_run_20260516")
N_PRIMARY_TESTS = 8  # 4 problems × 2 attitude q-types
ALPHA_BONF = 0.05 / N_PRIMARY_TESTS  # = 0.00625


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


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p_hat = k / n
    denom = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denom
    half_width = z * math.sqrt(p_hat * (1 - p_hat) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def binomial_two_sided_p(k: int, n: int, p0: float = 0.5) -> float:
    """Two-sided exact binomial test."""
    if n == 0:
        return 1.0
    p_hat = k / n
    # Use exact binomial for moderate n; normal approximation for large n.
    if n <= 200:
        # Exact two-sided
        smaller_tail = min(k, n - k)
        cdf_smaller = sum(math.comb(n, i) for i in range(0, smaller_tail + 1)) * (0.5**n)
        return min(1.0, 2 * cdf_smaller)
    # Normal approximation for large n
    se = math.sqrt(p0 * (1 - p0) / n)
    if se == 0:
        return 1.0 if p_hat == p0 else 0.0
    z = abs((p_hat - p0) / se)
    return math.erfc(z / math.sqrt(2))


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
        sia_letter = get_sia_aligned_letter(d.get("preferred_actions"), d.get("row_order"))
        d["is_thirder"] = (ch == sia_letter) if sia_letter else None
        if d["is_thirder"] is not None:
            cells.append(d)
    return cells


def main() -> int:
    cells = load_attitude_cells()
    print(f"Loaded {len(cells)} attitude cells with parsed choice and known SIA-letter mapping")
    print(f"  Operational definition: 'thirder choice' = picks SIA-aligned letter")
    print(f"  Bonferroni α (8 primary tests): {ALPHA_BONF:.5f}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("PRIMARY: per (problem_class, q-type) thirder rate vs 0.5 null")
    print(f"{'=' * 100}")
    by_pq: dict = defaultdict(lambda: {"thirder": 0, "total": 0})
    for c in cells:
        key = (c["problem_class"], c["question_type"])
        by_pq[key]["total"] += 1
        if c["is_thirder"]:
            by_pq[key]["thirder"] += 1

    results = []
    for (pc, qt), v in sorted(by_pq.items()):
        k, n = v["thirder"], v["total"]
        rate = k / n
        lo, hi = wilson_ci(k, n)
        p = binomial_two_sided_p(k, n, 0.5)
        results.append({"pc": pc, "qt": qt, "k": k, "n": n, "rate": rate,
                        "ci": (lo, hi), "p": p, "sig": p < ALPHA_BONF})

    print(f"\n  {'problem':<8} {'q-type':<22} {'thirder/total':<16} {'rate':<8} "
          f"{'95% CI':<22} {'p':<12} {'Bonf-sig':<8} {'lean':<10}")
    print("  " + "-" * 120)
    for r in results:
        sig = "**" if r["sig"] else ""
        lean = "thirder" if r["rate"] > 0.5 else ("halfer" if r["rate"] < 0.5 else "neutral")
        magnitude_pp = abs(r["rate"] - 0.5) * 100
        lean_str = f"{lean} ({magnitude_pp:+.1f}pp)"
        ci_str = f"[{r['ci'][0]:.3f}, {r['ci'][1]:.3f}]"
        print(f"  {r['pc']:<8} {r['qt']:<22} {r['k']}/{r['n']:<10} {r['rate']:.4f}   "
              f"{ci_str:<22} {r['p']:<12.4g} {sig:<8} {lean_str:<10}")

    n_sig = sum(1 for r in results if r["sig"])
    n_thirder = sum(1 for r in results if r["sig"] and r["rate"] > 0.5)
    n_halfer = sum(1 for r in results if r["sig"] and r["rate"] < 0.5)
    print(f"\n  Bonferroni-significant: {n_sig}/8 — of which thirder: {n_thirder}, halfer: {n_halfer}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("SECONDARY: per-cluster pooled thirder rate")
    print(f"{'=' * 100}")
    by_cluster: dict = defaultdict(lambda: {"thirder": 0, "total": 0})
    for c in cells:
        by_cluster[(c["cluster"], c["question_type"])]["total"] += 1
        if c["is_thirder"]:
            by_cluster[(c["cluster"], c["question_type"])]["thirder"] += 1

    print(f"\n  {'cluster':<12} {'q-type':<22} {'thirder/total':<16} {'rate':<8} "
          f"{'95% CI':<22} {'p':<12}")
    print("  " + "-" * 100)
    for (cl, qt), v in sorted(by_cluster.items()):
        k, n = v["thirder"], v["total"]
        rate = k / n
        lo, hi = wilson_ci(k, n)
        p = binomial_two_sided_p(k, n, 0.5)
        ci_str = f"[{lo:.3f}, {hi:.3f}]"
        print(f"  {cl:<12} {qt:<22} {k}/{n:<10} {rate:.4f}   {ci_str:<22} {p:<12.4g}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("SECONDARY: population-level pooled across all problems")
    print(f"{'=' * 100}")
    by_q: dict = defaultdict(lambda: {"thirder": 0, "total": 0})
    for c in cells:
        by_q[c["question_type"]]["total"] += 1
        if c["is_thirder"]:
            by_q[c["question_type"]]["thirder"] += 1

    print(f"\n  {'q-type':<22} {'thirder/total':<18} {'rate':<8} {'95% CI':<22} {'p':<12}")
    print("  " + "-" * 90)
    for qt, v in sorted(by_q.items()):
        k, n = v["thirder"], v["total"]
        rate = k / n
        lo, hi = wilson_ci(k, n)
        p = binomial_two_sided_p(k, n, 0.5)
        ci_str = f"[{lo:.3f}, {hi:.3f}]"
        print(f"  {qt:<22} {k}/{n:<12} {rate:.4f}   {ci_str:<22} {p:<12.4g}")

    # Fully pooled
    total_t = sum(v["thirder"] for v in by_q.values())
    total_n = sum(v["total"] for v in by_q.values())
    rate = total_t / total_n
    lo, hi = wilson_ci(total_t, total_n)
    p = binomial_two_sided_p(total_t, total_n, 0.5)
    print(f"\n  Both attitudes pooled: {total_t}/{total_n} ({rate:.4f}), "
          f"95% CI [{lo:.3f}, {hi:.3f}], p = {p:.4g}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("CONTEXTUAL: per (model, mode, cluster) thirder rate — the heterogeneity behind the aggregates")
    print(f"{'=' * 100}")
    by_mmc: dict = defaultdict(lambda: {"thirder": 0, "total": 0})
    for c in cells:
        by_mmc[(c["model_short"], c["mode"], c["cluster"])]["total"] += 1
        if c["is_thirder"]:
            by_mmc[(c["model_short"], c["mode"], c["cluster"])]["thirder"] += 1

    # Pivoted: per (model, mode), show SB-type and DD-type thirder rates
    by_mm: dict = defaultdict(dict)
    for (m, mode, cl), v in by_mmc.items():
        by_mm[(m, mode)][cl] = v
    print(f"\n  {'model':<32} {'mode':<5} {'SB-type thirder':<18} {'DD-type thirder':<18} "
          f"{'Δ (SB-DD)':<10}")
    print("  " + "-" * 90)
    for (m, mode), clusters in sorted(by_mm.items()):
        sb = clusters.get("SB-type", {"thirder": 0, "total": 0})
        dd = clusters.get("DD-type", {"thirder": 0, "total": 0})
        sb_rate = sb["thirder"] / sb["total"] if sb["total"] > 0 else 0
        dd_rate = dd["thirder"] / dd["total"] if dd["total"] > 0 else 0
        delta = (sb_rate - dd_rate) * 100
        print(f"  {m:<32} {mode:<5} {sb['thirder']}/{sb['total']} ({sb_rate:.3f})   "
              f"{dd['thirder']}/{dd['total']} ({dd_rate:.3f})   {delta:+.1f}pp")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("PER (model, mode, cluster) thirder rate — significance vs 0.5 null")
    print(f"24 tests = 12 model-modes × 2 clusters, Bonferroni α = {0.05/24:.5f}")
    print(f"{'=' * 100}")
    alpha_24 = 0.05 / 24
    print(f"\n  {'model':<32} {'mode':<5} {'cluster':<10} {'thirder/total':<16} {'rate':<8} "
          f"{'p (vs 0.5)':<12} {'Bonf-sig':<10} {'lean':<18}")
    print("  " + "-" * 130)
    # Sort by (model, mode, cluster) for reading
    sorted_keys = sorted(by_mmc.keys())
    for key in sorted_keys:
        m, mode, cl = key
        v = by_mmc[key]
        k, n = v["thirder"], v["total"]
        if n == 0:
            continue
        rate = k / n
        p = binomial_two_sided_p(k, n, 0.5)
        sig = "**" if p < alpha_24 else ("*" if p < 0.05 else "")
        if rate > 0.5:
            lean = f"thirder (+{(rate-0.5)*100:.1f}pp)"
        elif rate < 0.5:
            lean = f"halfer (+{(0.5-rate)*100:.1f}pp)"
        else:
            lean = "neutral"
        print(f"  {m:<32} {mode:<5} {cl:<10} {k}/{n:<10} {rate:.4f}   "
              f"{p:<12.4g} {sig:<10} {lean:<18}")

    # Summary counts
    n_sig_thirder_sb = sum(1 for k in sorted_keys if k[2] == "SB-type"
                            and (by_mmc[k]["thirder"]/by_mmc[k]["total"]) > 0.5
                            and binomial_two_sided_p(by_mmc[k]["thirder"], by_mmc[k]["total"]) < alpha_24)
    n_sig_halfer_sb = sum(1 for k in sorted_keys if k[2] == "SB-type"
                           and (by_mmc[k]["thirder"]/by_mmc[k]["total"]) < 0.5
                           and binomial_two_sided_p(by_mmc[k]["thirder"], by_mmc[k]["total"]) < alpha_24)
    n_sig_thirder_dd = sum(1 for k in sorted_keys if k[2] == "DD-type"
                            and (by_mmc[k]["thirder"]/by_mmc[k]["total"]) > 0.5
                            and binomial_two_sided_p(by_mmc[k]["thirder"], by_mmc[k]["total"]) < alpha_24)
    n_sig_halfer_dd = sum(1 for k in sorted_keys if k[2] == "DD-type"
                           and (by_mmc[k]["thirder"]/by_mmc[k]["total"]) < 0.5
                           and binomial_two_sided_p(by_mmc[k]["thirder"], by_mmc[k]["total"]) < alpha_24)
    n_nonsig_sb = 12 - n_sig_thirder_sb - n_sig_halfer_sb
    n_nonsig_dd = 12 - n_sig_thirder_dd - n_sig_halfer_dd
    print(f"\n  Summary by cluster (Bonferroni at α = {alpha_24:.5f}):")
    print(f"    SB-cluster: {n_sig_thirder_sb}/12 sig-thirder, {n_sig_halfer_sb}/12 sig-halfer, "
          f"{n_nonsig_sb}/12 not significant")
    print(f"    DD-cluster: {n_sig_thirder_dd}/12 sig-thirder, {n_sig_halfer_dd}/12 sig-halfer, "
          f"{n_nonsig_dd}/12 not significant")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
