#!/usr/bin/env python3
"""Additional sanity checks on Main run before locking dataset for analysis.

Eight checks:
  1. Row-order bias — for each (model, q-type), are row=12 A-rate and row=21
     A-rate approximately complementary (sum to ~1)? If yes, low position bias.
  2. Main run v1 vs Main run consistency on shared NON-SSA-capability cells
     (sia_cap, personal_att, normative_att on canonical SB/Inc) — same system
     prompts, so answers should be statistically similar.
  3. SSA prompt is the disambiguation clause everywhere in Main run — verify by checking the
     system_prompt field contains the SSA reference-class disambiguation clause.
  4. Reasoning toggle works — reasoning_off cells should have
     reasoning_tokens=0; reasoning_on cells should have reasoning_tokens>0.
  5. Within-cell sample consistency — for each (model, mode, problem, q-type, row)
     cell, look at distribution across the 9 samples.
  6. Param-scale predictions — scaled DD/PADD attitude should lean strongly
     toward SSA (high credence = doom-soon); scaled SB/Inc attitude should
     lean strongly toward SIA (low credence on Heads).
  7. Elapsed-time / cost outliers — any cells near 1200s timeout? Cost outliers?
  8. Null-cell category re-check — verify the 122 nulls match earlier
     categorization.

Usage:
  python scripts/sanity_checks_main_run.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

MAIN_RUN_DIR = Path("experiment_results/main_run_20260516")
SSA_DISAMBIGUATION_CLAUSE = "with the reference class contained within each hypothesis separately"


def load_cells(out_dir: Path) -> list[dict]:
    return [json.load(open(f)) for f in sorted(out_dir.glob("*.json"))]


def parse_problem_metadata(template_name: str) -> dict:
    parts = template_name.replace("20260516_standard_", "").replace("20260510_standard_", "")
    tokens = parts.rsplit("_", 1)
    row = tokens[1]
    rest = tokens[0]
    if rest.endswith("_scaled"):
        param = "scaled"
        rest = rest[:-len("_scaled")]
    else:
        param = "canonical"
    sub = rest.split("_", 1)
    return {"problem_class": sub[0], "theme": sub[1], "parameterization": param, "row": row}


def section_1_row_order_bias(cells: list[dict]) -> None:
    print("=" * 80)
    print("1. ROW-ORDER BIAS")
    print("=" * 80)
    print("For each (model, q-type), row=12 A-rate + row=21 A-rate should ~= 1")
    print("if models are content-consistent and position-unbiased.")
    print(f"{'model':<42} {'q_type':<22} {'r12 A%':>8} {'r21 A%':>8} {'sum':>6} {'|sum-1|':>8}\n")
    by = defaultdict(lambda: defaultdict(lambda: {"A": 0, "AB": 0}))
    for c in cells:
        meta = parse_problem_metadata(c["template_name"])
        if c.get("extracted_choice") not in ("A", "B"):
            continue
        key = (c["model"], c["question_type"])
        by[key][meta["row"]]["AB"] += 1
        if c["extracted_choice"] == "A":
            by[key][meta["row"]]["A"] += 1
    biggest_deviation = 0
    for key, rows in sorted(by.items()):
        r12 = rows.get("12", {"A": 0, "AB": 0})
        r21 = rows.get("21", {"A": 0, "AB": 0})
        if r12["AB"] == 0 or r21["AB"] == 0:
            continue
        r12_pct = r12["A"] / r12["AB"]
        r21_pct = r21["A"] / r21["AB"]
        s = r12_pct + r21_pct
        dev = abs(s - 1)
        biggest_deviation = max(biggest_deviation, dev)
        flag = "  ← biased?" if dev > 0.15 else ""
        print(f"{key[0]:<42} {key[1]:<22} {r12_pct:>7.1%} {r21_pct:>7.1%} {s:>6.2f} {dev:>7.1%}{flag}")
    print(f"\nLargest |sum-1| deviation: {biggest_deviation:.1%}")
    print(f"  (deviations > 15% would suggest meaningful position bias)\n")



def section_3_ssa_disambiguation_prompt_check(cells: list[dict]) -> None:
    print("=" * 80)
    print("3. SSA reference-class disambiguation PROMPT VERIFICATION")
    print("=" * 80)
    ssa_cells = [c for c in cells if c.get("question_type") == "ssa_capability"]
    n_with_disambiguation = sum(1 for c in ssa_cells if SSA_DISAMBIGUATION_CLAUSE in (c.get("system_prompt") or ""))
    print(f"ssa_capability cells: {len(ssa_cells)}")
    print(f"cells with SSA disambiguation marker in system_prompt: {n_with_disambiguation}")
    print(f"cells without SSA disambiguation marker: {len(ssa_cells) - n_with_disambiguation}")
    # Also verify NON-ssa-capability cells DON'T have SSA disambiguation marker (they shouldn't)
    non_ssa = [c for c in cells if c.get("question_type") != "ssa_capability"]
    n_wrong = sum(1 for c in non_ssa if SSA_DISAMBIGUATION_CLAUSE in (c.get("system_prompt") or ""))
    print(f"\nnon-SSA-capability cells with SSA disambiguation marker (should be 0): {n_wrong}")
    print()


def section_4_reasoning_toggle(cells: list[dict]) -> None:
    print("=" * 80)
    print("4. REASONING TOGGLE VERIFICATION")
    print("=" * 80)
    # reasoning_off cells should have reasoning_tokens=0
    # reasoning_on cells should have reasoning_tokens>0 (for hybrids), or always>0 for reasoning-only
    by = defaultdict(lambda: {"rt_zero": 0, "rt_positive": 0, "total": 0})
    for c in cells:
        u = c.get("usage_statistics") or {}
        rt = u.get("reasoning_tokens", 0) or 0
        key = (c["model"], c["reasoning_mode"])
        by[key]["total"] += 1
        if rt > 0:
            by[key]["rt_positive"] += 1
        else:
            by[key]["rt_zero"] += 1
    print(f"{'model':<42} {'mode':>4} {'rt=0':>6} {'rt>0':>6} {'%rt>0':>7} {'expected':>14}")
    for (m, mode), v in sorted(by.items()):
        pct = v["rt_positive"] / v["total"]
        expected = "rt=0" if mode == "off" else "rt>0"
        flag = ""
        if mode == "off" and pct > 0.05:
            flag = "  ← off cells have reasoning tokens!"
        if mode == "on" and pct < 0.95:
            flag = "  ← on cells lack reasoning tokens"
        print(f"{m:<42} {mode:>4} {v['rt_zero']:>6} {v['rt_positive']:>6} {pct:>6.1%} {expected:>14}{flag}")
    print()


def section_5_within_cell_consistency(cells: list[dict]) -> None:
    print("=" * 80)
    print("5. WITHIN-CELL SAMPLE CONSISTENCY")
    print("=" * 80)
    print("For each (model, mode, problem, q_type, row), how variable are the 9 samples?")
    print("Mostly 9/9 same = high commitment; mostly 5/9 = high variance.")
    by = defaultdict(lambda: defaultdict(int))  # key → choice → count
    for c in cells:
        if c.get("extracted_choice") not in ("A", "B"):
            continue
        key = (c["model"], c["reasoning_mode"], c["template_name"], c["question_type"])
        by[key][c["extracted_choice"]] += 1
    # Count distribution of majority sizes
    majority_distribution = defaultdict(int)  # majority_count → n_cells
    for key, counts in by.items():
        total = counts["A"] + counts["B"]
        if total == 0:
            continue
        majority = max(counts["A"], counts["B"])
        majority_distribution[majority] += 1
    print(f"  Distribution of majority counts across all (model, mode, problem, q, row) cells:")
    print(f"  (e.g., 'majority=9' means cell with 9/9 same choice; high commitment)")
    for m in sorted(majority_distribution.keys(), reverse=True):
        n = majority_distribution[m]
        print(f"    majority={m}: {n} cells")
    # Also report the rate of "tied" cells (5/4 or 4/5 with total=9)
    tied_cells = [k for k, v in by.items() if abs(v["A"] - v["B"]) <= 1 and (v["A"] + v["B"]) == 9]
    print(f"\n  Cells with near-tied distribution (4-5 split): {len(tied_cells)} (out of {len(by)} total cells)")
    print()


def section_6_param_scale_predictions(cells: list[dict]) -> None:
    print("=" * 80)
    print("6. PARAMETER-SCALE PREDICTIONS for attitude questions")
    print("=" * 80)
    print("Predictions:")
    print("  - Scaled DD/PADD personal/normative → models should converge on B (=0.1)")
    print("    because SIA recommends 0.1 (low) and SSA recommends 0.991 (high)")
    print("    in row=12. Most models followed SIA reasoning earlier on canonical DD.")
    print("    Wait — the predictions depend on which theory dominates in practice.")
    print("    For each scaled problem class, what is the dominant attitude direction?\n")
    by = defaultdict(lambda: {"A": 0, "B": 0})
    for c in cells:
        meta = parse_problem_metadata(c["template_name"])
        if meta["parameterization"] != "scaled":
            continue
        if c.get("question_type") not in ("personal_attitude", "normative_attitude"):
            continue
        if c.get("extracted_choice") not in ("A", "B"):
            continue
        # Convert to SSA-aligned vs SIA-aligned using ssa_aligned/sia_aligned fields
        key = (meta["problem_class"], meta["theme"], c["question_type"])
        if c["extracted_choice"] == "A":
            by[key]["A"] += 1
        else:
            by[key]["B"] += 1
    # Also compute SSA-aligned and SIA-aligned %s
    print(f"{'prob':<5} {'theme':<14} {'q_type':<22} {'A':>4} {'B':>4} {'SSA-aligned':>14} {'SIA-aligned':>14}")
    print("-" * 90)
    by_aligned = defaultdict(lambda: {"ssa": 0, "sia": 0, "total": 0})
    for c in cells:
        meta = parse_problem_metadata(c["template_name"])
        if meta["parameterization"] != "scaled":
            continue
        if c.get("question_type") not in ("personal_attitude", "normative_attitude"):
            continue
        if c.get("extracted_choice") not in ("A", "B"):
            continue
        key = (meta["problem_class"], meta["theme"], c["question_type"])
        by_aligned[key]["total"] += 1
        if c.get("ssa_aligned"):
            by_aligned[key]["ssa"] += 1
        if c.get("sia_aligned"):
            by_aligned[key]["sia"] += 1
    for key in sorted(by_aligned.keys()):
        ab = by[key]
        v = by_aligned[key]
        ssa_pct = v["ssa"] / v["total"] if v["total"] else 0
        sia_pct = v["sia"] / v["total"] if v["total"] else 0
        print(f"{key[0]:<5} {key[1]:<14} {key[2]:<22} {ab['A']:>4} {ab['B']:>4} {ssa_pct:>13.1%} {sia_pct:>13.1%}")
    print()


def section_7_outliers(cells: list[dict]) -> None:
    print("=" * 80)
    print("7. ELAPSED-TIME / COST OUTLIERS")
    print("=" * 80)
    timeouts_near = [c for c in cells if (c.get("elapsed_seconds") or 0) > 1100]
    timeouts_full = [c for c in cells if (c.get("elapsed_seconds") or 0) > 1200]
    print(f"  cells with elapsed > 1100s (close to 1200s timeout): {len(timeouts_near)}")
    print(f"  cells with elapsed > 1200s (full timeout):           {len(timeouts_full)}")
    if timeouts_near:
        for c in timeouts_near[:5]:
            print(f"    {c.get('elapsed_seconds'):.0f}s  {c['model']:<35}  {c.get('template_name', '').replace('20260516_standard_', '')[:40]}  {c.get('question_type')}")
    costs = sorted([(c.get("usage_statistics", {}) or {}).get("cost", 0) or 0 for c in cells], reverse=True)
    print(f"\n  Top 5 most expensive cells: ${costs[0]:.4f}, ${costs[1]:.4f}, ${costs[2]:.4f}, ${costs[3]:.4f}, ${costs[4]:.4f}")
    high_cost = [c for c in cells if ((c.get("usage_statistics") or {}).get("cost", 0) or 0) > 0.50]
    print(f"  cells with cost > $0.50: {len(high_cost)}")
    for c in high_cost[:5]:
        u = c.get("usage_statistics") or {}
        print(f"    ${u.get('cost'):.4f}  {c['model']}  rt={u.get('reasoning_tokens')}  ct={u.get('completion_tokens')}")
    print()


def section_8_null_cell_recheck(cells: list[dict]) -> None:
    print("=" * 80)
    print("8. NULL-CELL CATEGORY RE-CHECK")
    print("=" * 80)
    nulls = [c for c in cells if c.get("extracted_choice") is None]
    print(f"  total null cells: {len(nulls)} (expected: 122 after Gemini re-fire)")
    by_model = defaultdict(int)
    for c in nulls:
        by_model[(c["model"], c["reasoning_mode"])] += 1
    print(f"\n  Distribution by (model, mode):")
    for k, n in sorted(by_model.items(), key=lambda x: -x[1]):
        print(f"    {k[0]:<42} {k[1]:>3}  {n}")
    print()


def main() -> int:
    cells = load_cells(MAIN_RUN_DIR)
    print(f"loaded {len(cells)} Main run cells\n")
    section_1_row_order_bias(cells)
    section_3_ssa_disambiguation_prompt_check(cells)
    section_4_reasoning_toggle(cells)
    section_5_within_cell_consistency(cells)
    section_6_param_scale_predictions(cells)
    section_7_outliers(cells)
    section_8_null_cell_recheck(cells)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
