#!/usr/bin/env python3
"""Re-derive correctness/alignment fields for all Main run JSONs in place.

Background: the Main run run inadvertently computed `correct_capability_answer`,
`ssa_aligned`, and `sia_aligned` using `exp.row_order` from the experiment
instance — but under the parallelized driver, that field was stale (stuck at
the last loaded problem's row_order, which was "21"). So row=12 cells got
inverted correctness verdicts. The raw response data (extracted_choice,
preferred_actions, row_order, response text, reasoning_trace) is correct;
only the three derived fields are wrong.

This script re-derives those three fields using each cell's own metadata
row_order, and rewrites the JSONs in place (with a backup of the original).

Idempotent: running twice produces the same result.

Usage:
  python scripts/rederive_correctness_main_run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from framework import NewcombExperiment  # type: ignore  # noqa: E402

MAIN_RUN_DIR = Path("experiment_results/main_run_20260516")


def main() -> int:
    if not MAIN_RUN_DIR.exists():
        print(f"ERROR: {MAIN_RUN_DIR} not found")
        return 1
    # Instantiate just to access the method bindings; no model setup needed.
    exp = NewcombExperiment(base_output_dir="/tmp/_rederive", temperature=0.8)

    files = sorted(MAIN_RUN_DIR.glob("*.json"))
    print(f"loaded {len(files)} JSON files")
    changed = unchanged = skipped = 0
    flipped_count = 0
    for fp in files:
        with open(fp) as f:
            j = json.load(f)
        choice = j.get("extracted_choice")
        row_order = j.get("row_order")
        q_type = j.get("question_type")
        preferred = j.get("preferred_actions") or {}
        if choice is None or row_order is None or not preferred:
            skipped += 1
            continue
        # Set problem_structure on the exp instance so get_theory_pair works.
        # (We only need ssa_preference / sia_preference keys present; the values
        # are read separately via the preferred_actions dict.)
        exp.problem_structure = preferred
        # Re-derive
        new_alignment = exp.determine_alignment(choice, preferred, row_order=row_order)
        new_correctness = exp.check_correctness(choice, q_type, preferred, row_order=row_order)
        # Compare with existing values
        dirty = False
        for k, v in new_alignment.items():
            if j.get(k) != v:
                dirty = True
                j[k] = v
        if new_correctness is not None:
            if j.get("correct_capability_answer") != new_correctness:
                if j.get("correct_capability_answer") is not None:
                    flipped_count += 1
                j["correct_capability_answer"] = new_correctness
                dirty = True
        elif "correct_capability_answer" in j:
            # attitude question shouldn't have this field, but leave it if framework set it
            pass
        if dirty:
            with open(fp, "w") as f:
                json.dump(j, f, indent=2)
            changed += 1
        else:
            unchanged += 1
    print(f"\ndone: {changed} files modified, {unchanged} unchanged, {skipped} skipped (null/missing)")
    print(f"      {flipped_count} correct_capability_answer flips")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
