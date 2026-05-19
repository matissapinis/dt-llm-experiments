#!/usr/bin/env python3
"""RQ3 deep-dive: GPT-5.5 attitude shift toward SSA when reasoning is enabled.

From RQ3 main analysis:
  - GPT-5.5 normative_attitude: McNemar b=62, c=14, effect +0.632, p = 7×10⁻⁸
  - GPT-5.5 personal_attitude:  McNemar b=53, c=27, effect +0.325, p = 0.005
  - Both Bonferroni- or nominal-significant; both in direction "ON shifts toward SSA-aligned"

Question: is the shift uniform across problem types, or concentrated somewhere?

Breakdowns:
  1. By problem class (SB / INC / DD / PADD)
  2. By cluster (SB-type / DD-type)
  3. By parameterization (SB-type params / DD-type params)
  4. By theme (cluster-classic / aiinstance)
  5. By (cluster, parameterization) — full 2x2

For each breakdown, compute on-rate and off-rate of "choosing SSA-aligned answer",
plus chi-square test for shift significance.
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

D = Path("experiment_results/main_run_20260516")
TARGET_MODEL = "gpt-5.5-20260423"


def parse_mode(filename: str) -> str:
    if filename.endswith("_on.json"):
        return "on"
    if filename.endswith("_off.json"):
        return "off"
    return "?"


def parse_sample(filename: str) -> int:
    m = re.search(r"_sample(\d+)_", filename)
    return int(m.group(1)) if m else -1


def parse_problem_class(template_name: str) -> str:
    m = re.match(r"\d+_standard_(sb|inc|dd|padd)_", template_name or "")
    return m.group(1) if m else "?"


def parse_cluster(pc: str) -> str:
    return "SB-type" if pc in ("sb", "inc") else ("DD-type" if pc in ("dd", "padd") else "?")


def parse_theme(template_name: str) -> str:
    # Templates look like: 20260516_standard_<class>_<theme>[_scaled]_<row>
    m = re.match(r"\d+_standard_(?:sb|inc|dd|padd)_([a-z]+)(?:_scaled)?_\d+$", template_name or "")
    return m.group(1) if m else "?"


def parse_parameterization(template_name: str) -> str:
    """Returns 'DD-type' if _scaled in template, else 'SB-type'."""
    if "_scaled_" in (template_name or "") or (template_name or "").endswith("_scaled"):
        return "DD-type"
    return "SB-type"


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


def load_gpt55_attitudes():
    cells = []
    for f in sorted(D.glob("*.json")):
        d = json.load(open(f))
        model = (d.get("model_id_openrouter") or "").split("/")[-1]
        if model != TARGET_MODEL:
            continue
        qt = d.get("question_type") or ""
        if not qt.endswith("_attitude"):
            continue
        ch = d.get("extracted_choice")
        if ch not in ("A", "B"):
            continue
        d["mode"] = parse_mode(f.name)
        d["problem_class"] = parse_problem_class(d.get("template_name", ""))
        d["cluster"] = parse_cluster(d["problem_class"])
        d["theme"] = parse_theme(d.get("template_name", ""))
        d["parameterization"] = parse_parameterization(d.get("template_name", ""))
        ssa_letter = get_ssa_aligned_letter(d.get("preferred_actions"), d.get("row_order"))
        d["is_ssa_aligned"] = (ch == ssa_letter) if ssa_letter else None
        cells.append(d)
    return cells


def shift_analysis(cells: list[dict], group_label: str, group_key_fn) -> None:
    """Per-group shift in SSA-aligned rate (on vs off), with chi-square test."""
    groups: dict = defaultdict(lambda: {"on": [0, 0], "off": [0, 0]})  # [ssa, total]
    for c in cells:
        if c["is_ssa_aligned"] is None:
            continue
        key = group_key_fn(c)
        groups[key][c["mode"]][1] += 1
        if c["is_ssa_aligned"]:
            groups[key][c["mode"]][0] += 1
    print(f"\n  Breakdown by {group_label}:")
    print(f"  {'group':<35} {'off: SSA/total':<18} {'on: SSA/total':<18} "
          f"{'Δ (on-off)':<11} {'χ²':<7} {'p':<10} {'sig':<4}")
    print("  " + "-" * 110)
    for key, v in sorted(groups.items()):
        on_ssa, on_n = v["on"]
        off_ssa, off_n = v["off"]
        on_rate = on_ssa / on_n if on_n > 0 else 0
        off_rate = off_ssa / off_n if off_n > 0 else 0
        chi2, p = chi2_2x2(on_ssa, on_n - on_ssa, off_ssa, off_n - off_ssa)
        sig = "**" if p < 0.001 else ("*" if p < 0.05 else "")
        print(f"  {str(key):<35} {off_ssa}/{off_n} ({off_rate:.3f})    "
              f"{on_ssa}/{on_n} ({on_rate:.3f})    "
              f"{on_rate - off_rate:+.3f}     {chi2:<7.2f} {p:<10.4g} {sig:<4}")


def main() -> int:
    cells = load_gpt55_attitudes()
    print(f"Loaded {len(cells)} GPT-5.5 attitude cells with parsed choice and known SSA-letter mapping")
    n_on = sum(1 for c in cells if c["mode"] == "on")
    n_off = sum(1 for c in cells if c["mode"] == "off")
    print(f"  ON cells: {n_on}, OFF cells: {n_off}")

    # Overall shift
    print(f"\n{'=' * 100}")
    print(f"SECTION 0: Overall shift in P(SSA-aligned) when reasoning is enabled")
    print(f"{'=' * 100}")
    on_ssa = sum(1 for c in cells if c["mode"] == "on" and c["is_ssa_aligned"])
    off_ssa = sum(1 for c in cells if c["mode"] == "off" and c["is_ssa_aligned"])
    on_rate = on_ssa / n_on
    off_rate = off_ssa / n_off
    chi2, p = chi2_2x2(on_ssa, n_on - on_ssa, off_ssa, n_off - off_ssa)
    print(f"  OFF: {off_ssa}/{n_off} SSA-aligned ({off_rate:.4f})")
    print(f"  ON:  {on_ssa}/{n_on} SSA-aligned ({on_rate:.4f})")
    print(f"  Δ (on-off): {on_rate - off_rate:+.4f}")
    print(f"  χ² = {chi2:.2f}, p = {p:.4g}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print(f"SECTION 1: By question type (normative vs personal attitude)")
    print(f"{'=' * 100}")
    shift_analysis(cells, "question_type", lambda c: c["question_type"])

    print(f"\n{'=' * 100}")
    print(f"SECTION 2: By problem class")
    print(f"{'=' * 100}")
    shift_analysis(cells, "problem class", lambda c: c["problem_class"])

    print(f"\n{'=' * 100}")
    print(f"SECTION 3: By cluster")
    print(f"{'=' * 100}")
    shift_analysis(cells, "cluster", lambda c: c["cluster"])

    print(f"\n{'=' * 100}")
    print(f"SECTION 4: By parameterization")
    print(f"{'=' * 100}")
    shift_analysis(cells, "parameterization", lambda c: c["parameterization"])

    print(f"\n{'=' * 100}")
    print(f"SECTION 5: By theme")
    print(f"{'=' * 100}")
    shift_analysis(cells, "theme", lambda c: c["theme"])

    print(f"\n{'=' * 100}")
    print(f"SECTION 6: By (cluster × parameterization) — 2x2 grid")
    print(f"{'=' * 100}")
    shift_analysis(cells, "(cluster, parameterization)",
                   lambda c: f"{c['cluster']} / {c['parameterization']}-params")

    print(f"\n{'=' * 100}")
    print(f"SECTION 7: By (problem_class × question_type) — fine grid")
    print(f"{'=' * 100}")
    shift_analysis(cells, "(problem_class, question_type)",
                   lambda c: f"{c['problem_class']:>5} / {c['question_type'][:20]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
