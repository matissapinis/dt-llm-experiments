#!/usr/bin/env python3
"""QT7d — Qwen-on FN audit (exhaustive): the 8 formally-incorrect capability cells.

Qwen-on has V1 capability accuracy 99.3% — 8 wrong cells out of 1144 capability
cells. This script loads all 8 and prints them anonymized for blind soundness
coding. With only 8 cells, exhaustive audit is feasible.

Question: are Qwen-on's 8 wrong answers genuinely unsound (model got it wrong),
or are they sound reasoning that was mislabeled by the formal grader (defensible
alternative interpretation)?
"""
from __future__ import annotations

import json
import re
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


def get_sia_aligned_letter(preferred_actions: dict, row_order: str) -> str | None:
    if not preferred_actions:
        return None
    sia_pref = preferred_actions.get("sia_preference")
    if not sia_pref:
        return None
    is_A_in_row12 = sia_pref in ("half", "high")
    return ("A" if is_A_in_row12 else "B") if row_order == "12" else ("B" if is_A_in_row12 else "A")


def main() -> int:
    wrong_cells = []
    for f in sorted(D.glob("*.json")):
        if not f.name.endswith("_on.json"):
            continue
        d = json.load(open(f))
        ch = d.get("extracted_choice")
        if ch not in ("A", "B"):
            continue
        qt = d.get("question_type")
        if qt not in ("sia_capability", "ssa_capability"):
            continue
        model = (d.get("model_id_openrouter") or "").split("/")[-1]
        if "qwen" not in model.lower():
            continue
        sia_letter = get_sia_aligned_letter(d.get("preferred_actions"), d.get("row_order"))
        if sia_letter is None:
            continue
        is_thirder = (ch == sia_letter)
        is_correct = is_thirder if qt == "sia_capability" else (not is_thirder)
        if is_correct:
            continue
        d["mode"] = parse_mode(f.name)
        d["model_short"] = model
        d["problem_class"] = parse_problem_class(d.get("template_name", ""))
        d["_filename"] = f.name
        rt = d.get("reasoning_trace") or ""
        resp = d.get("response") or ""
        d["_trace_or_resp"] = rt if rt.strip() else resp
        d["expected_letter"] = sia_letter if qt == "sia_capability" else ("A" if sia_letter == "B" else "B")
        wrong_cells.append(d)

    print(f"Total Qwen-on incorrect capability cells: {len(wrong_cells)}")
    for i, c in enumerate(wrong_cells, 1):
        print(f"\n{'#' * 100}")
        print(f"CASE #{i:02d}")
        print(f"{'#' * 100}")
        print(f"[meta — for cell identification]")
        print(f"  filename: {c['_filename']}")
        print(f"  q_type:   {c.get('question_type')}")
        print(f"  pc:       {c['problem_class']}")
        print(f"\nPROMPT:")
        print("-" * 100)
        print(c.get("user_prompt", ""))
        print()
        print(f"REASONING / RESPONSE:")
        print("-" * 100)
        print(c["_trace_or_resp"])
        print()
        print(f"EXTRACTED CHOICE: {c.get('extracted_choice')}  "
              f"|  V1 expected: {c['expected_letter']}  "
              f"|  formally INCORRECT")

    print(f"\n\n{'=' * 100}")
    print("LABEL KEY (always INCORRECT in this set):")
    print(f"{'=' * 100}")
    for i, c in enumerate(wrong_cells, 1):
        print(f"  Case #{i:02d}: q={c.get('question_type'):<16} pc={c['problem_class']:<5} "
              f"chose={c.get('extracted_choice')} expected={c['expected_letter']} "
              f"file={c['_filename'][:60]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
