#!/usr/bin/env python3
"""QT7b — Claude-specific FP audit on the 100% V1 capability cells.

Sample 30 formally-correct Claude capability cells (15 off + 15 on),
stratified across (problem_class, q_type) to ensure coverage. Print
anonymized for blind soundness coding, then reveal labels for FP estimation.

Question: is Claude's 100% V1 capability accuracy backed by sound reasoning,
or do some of those "correct" answers rely on lucky pattern-matching?
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

D = Path("experiment_results/main_run_20260516")
SEED = 20260518 + 900
N_PER_MODE = 15  # 15 off + 15 on = 30 total


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


def load_claude_correct_capability_cells():
    cells = []
    for f in sorted(D.glob("*.json")):
        d = json.load(open(f))
        ch = d.get("extracted_choice")
        if ch not in ("A", "B"):
            continue
        qt = d.get("question_type")
        if qt not in ("sia_capability", "ssa_capability"):
            continue
        model = (d.get("model_id_openrouter") or "").split("/")[-1]
        if "claude" not in model.lower():
            continue
        sia_letter = get_sia_aligned_letter(d.get("preferred_actions"), d.get("row_order"))
        if sia_letter is None:
            continue
        is_thirder = (ch == sia_letter)
        is_correct = is_thirder if qt == "sia_capability" else (not is_thirder)
        if not is_correct:
            continue
        d["mode"] = parse_mode(f.name)
        d["model_short"] = model
        d["problem_class"] = parse_problem_class(d.get("template_name", ""))
        rt = d.get("reasoning_trace") or ""
        resp = d.get("response") or ""
        d["_trace_or_resp"] = rt if rt.strip() else resp
        d["_filename"] = f.name
        cells.append(d)
    return cells


def stratified_sample(cells, n, seed):
    """Sample n cells stratified by (mode, problem_class, q_type)."""
    rng = random.Random(seed)
    by_stratum = defaultdict(list)
    for c in cells:
        by_stratum[(c["mode"], c["problem_class"], c.get("question_type"))].append(c)
    keys = sorted(by_stratum.keys())
    rng.shuffle(keys)
    picked = []
    cursor = 0
    while len(picked) < n and any(by_stratum[k] for k in keys):
        k = keys[cursor % len(keys)]
        if by_stratum[k]:
            idx = rng.randrange(len(by_stratum[k]))
            picked.append(by_stratum[k].pop(idx))
        cursor += 1
        if cursor > len(keys) * 200:
            break
    return picked[:n]


def main() -> int:
    cells = load_claude_correct_capability_cells()
    print(f"Total Claude correct capability cells: {len(cells)}")
    print(f"  By mode: off={sum(1 for c in cells if c['mode']=='off')}, "
          f"on={sum(1 for c in cells if c['mode']=='on')}")

    off_cells = [c for c in cells if c["mode"] == "off"]
    on_cells = [c for c in cells if c["mode"] == "on"]
    sample_off = stratified_sample(off_cells, N_PER_MODE, seed=SEED)
    sample_on = stratified_sample(on_cells, N_PER_MODE, seed=SEED + 1)
    combined = sample_off + sample_on
    rng = random.Random(SEED + 2)
    rng.shuffle(combined)

    label_key = {}
    for i, c in enumerate(combined, 1):
        label_key[i] = {
            "filename": c["_filename"],
            "mode": c["mode"],
            "q_type": c.get("question_type"),
            "problem_class": c["problem_class"],
            "choice": c.get("extracted_choice"),
        }
        print(f"\n{'#' * 100}")
        print(f"CASE #{i:02d}")
        print(f"{'#' * 100}")
        print(f"[meta — for cell identification only]")
        print(f"  filename: {c['_filename']}")
        print(f"  q_type:   {c.get('question_type')}")
        print(f"\nPROMPT:")
        print("-" * 100)
        print(c.get("user_prompt", ""))
        print()
        print(f"REASONING / RESPONSE:")
        print("-" * 100)
        print(c["_trace_or_resp"])
        print()
        print(f"EXTRACTED CHOICE: {c.get('extracted_choice')}")

    print(f"\n\n{'=' * 100}")
    print("LABEL KEY (revealed after blind coding)")
    print(f"{'=' * 100}")
    for i, lk in sorted(label_key.items()):
        print(f"  Case #{i:02d}: mode={lk['mode']:<3} | q={lk['q_type']:<16} | "
              f"pc={lk['problem_class']:<5} | chose={lk['choice']} | file={lk['filename'][:60]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
