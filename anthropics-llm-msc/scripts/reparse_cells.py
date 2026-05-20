#!/usr/bin/env python3
"""Re-extract extracted_choice for all Main run JSONs using the fixed parser.

The fixed parser uses LAST-match (rather than first-match) of the FINAL ANSWER
pattern, and rejects any letter that isn't A or B. This recovers cells where:
  - Models echoed the prompt's "FINAL ANSWER: X" example earlier in their
    reasoning, causing the old parser to extract X as the choice.
  - Models wrote N/A as off-menu refusal (old parser extracted N).
  - Other parser-confusion cases.

For cells whose extracted_choice changes, also re-derives correct_capability_answer,
ssa_aligned, sia_aligned using the cell's own row_order.

Usage:
  python scripts/reparse_main_run.py
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from framework import NewcombExperiment  # type: ignore  # noqa: E402

MAIN_RUN_DIR = Path("experiment_results/main_run_20260516")


def main() -> int:
    exp = NewcombExperiment(base_output_dir="/tmp/_reparse", temperature=0.8)
    files = sorted(MAIN_RUN_DIR.glob("*.json"))
    changes = Counter()  # (old, new) → count
    n_total = 0
    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        n_total += 1
        resp = d.get("response") or ""
        old_choice = d.get("extracted_choice")
        new_choice = exp.extract_final_answer(resp) if resp else None
        if old_choice == new_choice:
            continue
        changes[(old_choice, new_choice)] += 1
        # Update
        d["extracted_choice"] = new_choice
        # Re-derive correctness / alignment
        preferred = d.get("preferred_actions") or {}
        row_order = d.get("row_order")
        q_type = d.get("question_type")
        if new_choice and preferred and row_order:
            exp.problem_structure = preferred
            alignment = exp.determine_alignment(new_choice, preferred, row_order=row_order)
            d.update(alignment)
            correctness = exp.check_correctness(new_choice, q_type, preferred, row_order=row_order)
            if correctness is not None:
                d["correct_capability_answer"] = correctness
        else:
            # If new_choice is None, clear derived fields
            for k in ("correct_capability_answer", "ssa_aligned", "sia_aligned",
                      "cdt_aligned", "edt_aligned"):
                d.pop(k, None)
        with open(fp, "w") as f:
            json.dump(d, f, indent=2)

    print(f"Re-parsed {n_total} JSONs")
    print(f"\nChange summary (old → new):")
    for (old, new), n in sorted(changes.items(), key=lambda x: -x[1]):
        print(f"  {repr(old):>15}  →  {repr(new):>15}  : {n}")
    print(f"\nTotal cells changed: {sum(changes.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
