#!/usr/bin/env python3
"""Preliminary exploratory pass on Main run results.

Parallels scripts/quicklook_stage1.py but adapted for Main run's expanded
scope (32 problem dirs × 4 q-types × 9 samples × 12 cells = 13,824 cells).

Sections:
  A. Completeness audit
  B. Per-(model, mode) capability accuracy (overall and split by canonical/scaled)
  C. Choice distribution per (problem-family, theme, parameterization, q-type)
  D. SSA reference-class disambiguation validation: Main run canonical SB/Inc SSA-capability vs Main run v1
  E. Scaled parameterizations — did the predicted dramatic SSA/SIA contrasts materialize?
  F. Off-menu / unparseable cells
  G. Theme effect: classic/civilization vs AI-instance comparison
  H. Sample of reasoning traces from scaled SB-classic personal_attitude (for implausibility check)

Usage:
  python scripts/quicklook_main_run.py
"""
from __future__ import annotations

import json
import re
import random
from collections import defaultdict
from pathlib import Path

MAIN_RUN_DIR = Path("experiment_results/main_run_20260516")


def load_cells(out_dir: Path) -> list[dict]:
    cells = []
    for f in sorted(out_dir.glob("*.json")):
        j = json.load(open(f))
        j["_file"] = f.name
        cells.append(j)
    return cells


def parse_problem_metadata(template_name: str) -> dict:
    """Extract problem_class, theme, parameterization, row from dirname."""
    # e.g., "20260516_standard_sb_classic_12" or "20260516_standard_dd_civilization_scaled_21"
    parts = template_name.replace("20260516_standard_", "").replace("20260510_standard_", "")
    # parts is like "sb_classic_12" or "dd_civilization_scaled_21"
    tokens = parts.rsplit("_", 1)  # split off row
    row = tokens[1]
    rest = tokens[0]
    # rest could be "sb_classic", "dd_civilization", "sb_classic_scaled", etc.
    if rest.endswith("_scaled"):
        param = "scaled"
        rest = rest[:-len("_scaled")]
    else:
        param = "canonical"
    # rest is now "sb_classic", "inc_classic", "dd_civilization", "padd_civilization",
    # or aiinstance variants
    sub = rest.split("_", 1)
    problem_class = sub[0]   # sb, inc, dd, padd
    theme = sub[1]           # classic, civilization, aiinstance
    return {"problem_class": problem_class, "theme": theme, "parameterization": param, "row": row}


def section_a_completeness(cells: list[dict]) -> None:
    print("=" * 80)
    print("A. COMPLETENESS AUDIT")
    print("=" * 80)
    print(f"total cells on disk: {len(cells)}")
    expected = 32 * 4 * 9 * 12  # 32 problem dirs × 4 q-types × 9 samples × 12 cells
    print(f"expected:            {expected}")
    print(f"missing:             {expected - len(cells)}")
    null_choices = [c for c in cells if c.get("extracted_choice") is None]
    print(f"\ncells with null extracted_choice: {len(null_choices)}")
    finish_none = [c for c in cells if c.get("finish_reason") is None]
    print(f"cells with finish_reason=None:    {len(finish_none)}")
    empty_response = [c for c in cells if not (c.get("response") or "")]
    print(f"cells with empty response:        {len(empty_response)}")
    timed_out = [c for c in cells if (c.get("elapsed_seconds") or 0) > 1100]
    print(f"cells > 1100s elapsed (~timeout): {len(timed_out)}")
    # Scenario-level completeness
    by_scenario = defaultdict(list)
    for c in cells:
        key = (c.get("template_name"), c.get("question_type"), c.get("run_number"))
        by_scenario[key].append(c)
    incomplete = [(k, len(v)) for k, v in by_scenario.items() if len(v) != 12]
    print(f"\nscenarios with != 12 cells: {len(incomplete)}")
    for k, n in incomplete[:5]:
        print(f"  {k}: {n} cells")
    print()


def section_b_capability(cells: list[dict]) -> None:
    print("=" * 80)
    print("B. CAPABILITY ACCURACY")
    print("=" * 80)
    # Overall per (model, mode)
    by = defaultdict(lambda: {"correct": 0, "total": 0})
    for c in cells:
        if c.get("question_type") not in ("ssa_capability", "sia_capability"):
            continue
        key = (c["model"], c["reasoning_mode"])
        if c.get("correct_capability_answer") is True:
            by[key]["correct"] += 1
        if c.get("correct_capability_answer") in (True, False):
            by[key]["total"] += 1
    print(f"\n[B.1] Overall capability accuracy per (model, mode), aggregated across all 32 (problem×theme×param):")
    print(f"{'model':<42} {'mode':>4} {'correct':>8} {'total':>6} {'acc':>6}")
    print("-" * 72)
    for (m, mode), v in sorted(by.items()):
        acc = v["correct"] / v["total"] if v["total"] else 0
        print(f"{m:<42} {mode:>4} {v['correct']:>8} {v['total']:>6} {acc:>6.1%}")

    # Split by canonical vs scaled
    print(f"\n[B.2] Capability accuracy per (model, mode, parameterization):")
    by_p = defaultdict(lambda: {"correct": 0, "total": 0})
    for c in cells:
        if c.get("question_type") not in ("ssa_capability", "sia_capability"):
            continue
        meta = parse_problem_metadata(c["template_name"])
        key = (c["model"], c["reasoning_mode"], meta["parameterization"])
        if c.get("correct_capability_answer") is True:
            by_p[key]["correct"] += 1
        if c.get("correct_capability_answer") in (True, False):
            by_p[key]["total"] += 1
    print(f"{'model':<42} {'mode':>4} {'param':<10} {'correct':>8} {'total':>6} {'acc':>6}")
    print("-" * 82)
    for (m, mode, p), v in sorted(by_p.items()):
        acc = v["correct"] / v["total"] if v["total"] else 0
        print(f"{m:<42} {mode:>4} {p:<10} {v['correct']:>8} {v['total']:>6} {acc:>6.1%}")
    print()


def section_c_distributions(cells: list[dict]) -> None:
    print("=" * 80)
    print("C. CHOICE DISTRIBUTION per (problem_class, theme, parameterization, q-type)")
    print("=" * 80)
    print("Counts aggregated across models × modes × rows × samples.")
    print("'ssa_letter' = which letter SSA recommends per row (varies by row).\n")
    # For each cell, classify choice as 'ssa-aligned' or 'sia-aligned' using
    # ssa_preference and the choice mapping for the cell's row.
    by = defaultdict(lambda: {"A": 0, "B": 0, "null": 0})
    for c in cells:
        meta = parse_problem_metadata(c["template_name"])
        key = (meta["problem_class"], meta["theme"], meta["parameterization"], c.get("question_type"))
        choice = c.get("extracted_choice")
        if choice == "A":
            by[key]["A"] += 1
        elif choice == "B":
            by[key]["B"] += 1
        else:
            by[key]["null"] += 1
    print(f"{'prob':<5} {'theme':<14} {'param':<10} {'q_type':<22} {'A':>4} {'B':>4} {'null':>4} {'A%':>5} {'B%':>5}")
    print("-" * 92)
    for key, v in sorted(by.items()):
        total = sum(v.values())
        a_pct = v["A"] / total if total else 0
        b_pct = v["B"] / total if total else 0
        prob, theme, param, q = key
        print(f"{prob:<5} {theme:<14} {param:<10} {q:<22} {v['A']:>4} {v['B']:>4} {v['null']:>4} {a_pct:>5.1%} {b_pct:>5.1%}")
    print()



def section_e_scaled_validation(cells: list[dict]) -> None:
    print("=" * 80)
    print("E. SCALED PARAMETERIZATIONS — did the predicted dramatic contrasts materialize?")
    print("=" * 80)
    print("Predictions:")
    print("  - Scaled SB/Inc (Parameterization C): SSA→0.9 (A), SIA→0.0089 (B)")
    print("  - Scaled DD/PADD (DDA): SSA→0.991 (A in row=12), SIA→0.1 (B in row=12)\n")
    # Choice distribution per (problem-class, theme, q-type) restricted to scaled
    by = defaultdict(lambda: {"A": 0, "B": 0, "null": 0})
    for c in cells:
        meta = parse_problem_metadata(c["template_name"])
        if meta["parameterization"] != "scaled":
            continue
        key = (meta["problem_class"], meta["theme"], c.get("question_type"))
        choice = c.get("extracted_choice")
        if choice == "A": by[key]["A"] += 1
        elif choice == "B": by[key]["B"] += 1
        else: by[key]["null"] += 1
    print(f"{'prob':<5} {'theme':<14} {'q_type':<22} {'A':>4} {'B':>4} {'null':>4} {'A%':>5}")
    print("-" * 75)
    for key, v in sorted(by.items()):
        total = sum(v.values())
        a_pct = v["A"] / total if total else 0
        print(f"{key[0]:<5} {key[1]:<14} {key[2]:<22} {v['A']:>4} {v['B']:>4} {v['null']:>4} {a_pct:>5.1%}")
    print()


def section_f_off_menu(cells: list[dict], max_show: int = 10) -> None:
    print("=" * 80)
    print("F. OFF-MENU / UNPARSEABLE CELLS")
    print("=" * 80)
    nulls = [c for c in cells if c.get("extracted_choice") is None]
    print(f"total cells with null extracted_choice: {len(nulls)}\n")
    if not nulls:
        return
    # Group by why null
    no_response = [c for c in nulls if not (c.get("response") or "")]
    has_response = [c for c in nulls if c.get("response")]
    print(f"  empty response (no parseable text):    {len(no_response)}")
    print(f"  non-empty response but no FINAL ANSWER: {len(has_response)} (likely off-menu refusals)")
    # Show first few off-menu refusals
    if has_response:
        print(f"\nFirst {max_show} off-menu refusal previews:")
        for c in has_response[:max_show]:
            meta = parse_problem_metadata(c["template_name"])
            resp = (c.get("response") or "").strip()
            tail = resp[-300:] if len(resp) > 300 else resp
            print(f"\n--- {c['model']} ({c['reasoning_mode']}) {meta['problem_class']}-{meta['theme']}-{meta['parameterization']} {c['question_type']} sample {c['run_number']} ---")
            print(f"  response tail ({len(resp)} chars total):")
            for line in tail.splitlines()[-8:]:
                print(f"    {line}")
    print()


def section_g_theme_effect(cells: list[dict]) -> None:
    print("=" * 80)
    print("G. THEME EFFECT: canonical (classic/civilization) vs AI-instance theme")
    print("=" * 80)
    # Per (q-type, parameterization) compute classic-theme rate vs AI-instance-theme rate of choice A
    by = defaultdict(lambda: {"A": 0, "n": 0})
    for c in cells:
        meta = parse_problem_metadata(c["template_name"])
        # Group canonical-canonical theme together (classic, civilization)
        theme_group = "aiinstance" if meta["theme"] == "aiinstance" else "canonical_theme"
        key = (meta["problem_class"], theme_group, meta["parameterization"], c.get("question_type"))
        if c.get("extracted_choice") == "A":
            by[key]["A"] += 1
        if c.get("extracted_choice") in ("A", "B"):
            by[key]["n"] += 1
    # Show paired contrast per (problem, parameterization, q-type)
    print(f"{'prob':<5} {'param':<10} {'q_type':<22} {'canon_A%':>10} {'ai_A%':>8} {'diff':>7}")
    print("-" * 75)
    keys_seen = set()
    for c in cells:
        meta = parse_problem_metadata(c["template_name"])
        keys_seen.add((meta["problem_class"], meta["parameterization"], c.get("question_type")))
    for prob, param, q in sorted(keys_seen):
        canon = by.get((prob, "canonical_theme", param, q), {"A": 0, "n": 0})
        ai = by.get((prob, "aiinstance", param, q), {"A": 0, "n": 0})
        if canon["n"] == 0 or ai["n"] == 0:
            continue
        canon_pct = canon["A"] / canon["n"]
        ai_pct = ai["A"] / ai["n"]
        diff = canon_pct - ai_pct
        print(f"{prob:<5} {param:<10} {q:<22} {canon_pct:>10.1%} {ai_pct:>8.1%} {diff:>+7.1%}")
    print()


def section_h_implausibility_sample(cells: list[dict], n_samples: int = 4) -> None:
    print("=" * 80)
    print("H. IMPLAUSIBILITY CHECK: sample reads on scaled SB-classic personal_attitude")
    print("=" * 80)
    print("Pre-run spot check (24 cells) showed 0% implausibility flagging.")
    print(f"Random sample of {n_samples} from the full Main run run to confirm pattern.\n")
    sub = [
        c for c in cells
        if "sb_classic_scaled" in c.get("template_name", "")
        and c.get("question_type") == "personal_attitude"
    ]
    rng = random.Random(42)
    sample = rng.sample(sub, min(n_samples, len(sub)))
    implausibility_terms = ["implausible", "absurd", "unrealistic", "impossible",
                             "biologically", "physically impossible", "lifespan",
                             "thought experiment", "abstract", "feasible"]
    for c in sample:
        resp = (c.get("response") or "").strip()
        hits = [t for t in implausibility_terms if t.lower() in resp.lower()]
        print(f"\n--- {c['model']} ({c['reasoning_mode']}) sample {c['run_number']} → choice={c.get('extracted_choice')} ---")
        if hits:
            print(f"  IMPLAUSIBILITY FLAGS: {hits}")
        print(f"  response ({len(resp)} chars), first 600 chars:")
        for line in resp[:600].splitlines()[:15]:
            print(f"    {line}")
    print()


def main() -> int:
    print(f"loading Main run cells from {MAIN_RUN_DIR} ...")
    cells = load_cells(MAIN_RUN_DIR)
    print(f"loaded {len(cells)} cells\n")
    section_a_completeness(cells)
    section_b_capability(cells)
    section_c_distributions(cells)
    section_e_scaled_validation(cells)
    section_f_off_menu(cells)
    section_g_theme_effect(cells)
    section_h_implausibility_sample(cells)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
