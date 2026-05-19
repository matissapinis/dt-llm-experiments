#!/usr/bin/env python3
"""QT7c — FP audit on three additional near-100% V1 capability models.

Sample 25 formally-correct capability cells from each of:
  - gpt-5.5-20260423 on (V1 capability = 100.0%)
  - gemini-3.1-pro-preview-20260219 on (V1 capability = 100.0%)
  - qwen3.6-max-preview-20260420 on (V1 capability = 99.3%)

Stratified across (problem_class, q_type). Print anonymized for blind
soundness coding. Parallel to qt7b_claude_fp_audit.py.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

D = Path("experiment_results/main_run_20260516")
SEED = 20260518 + 1500
N_PER_MODEL = 25

TARGET_MODELS = [
    ("gpt-5.5-20260423", "on"),
    ("gemini-3.1-pro-preview-20260219", "on"),
    ("qwen3.6-max-preview-20260420", "on"),
]


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


def load_target_correct_cells():
    cells_by_model = defaultdict(list)
    for f in sorted(D.glob("*.json")):
        d = json.load(open(f))
        ch = d.get("extracted_choice")
        if ch not in ("A", "B"):
            continue
        qt = d.get("question_type")
        if qt not in ("sia_capability", "ssa_capability"):
            continue
        model = (d.get("model_id_openrouter") or "").split("/")[-1]
        mode = parse_mode(f.name)
        if (model, mode) not in TARGET_MODELS:
            continue
        sia_letter = get_sia_aligned_letter(d.get("preferred_actions"), d.get("row_order"))
        if sia_letter is None:
            continue
        is_thirder = (ch == sia_letter)
        is_correct = is_thirder if qt == "sia_capability" else (not is_thirder)
        if not is_correct:
            continue
        d["mode"] = mode
        d["model_short"] = model
        d["problem_class"] = parse_problem_class(d.get("template_name", ""))
        rt = d.get("reasoning_trace") or ""
        resp = d.get("response") or ""
        d["_trace_or_resp"] = rt if rt.strip() else resp
        d["_filename"] = f.name
        cells_by_model[(model, mode)].append(d)
    return cells_by_model


def stratified_sample(cells, n, seed):
    rng = random.Random(seed)
    by_stratum = defaultdict(list)
    for c in cells:
        by_stratum[(c["problem_class"], c.get("question_type"))].append(c)
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
    cells_by_model = load_target_correct_cells()
    for (m, mode), lst in cells_by_model.items():
        print(f"  {m}/{mode}: {len(lst)} correct capability cells in pool")
    print()

    # Sample 25 per target model
    all_picks = []
    for i, (m, mode) in enumerate(TARGET_MODELS):
        picks = stratified_sample(cells_by_model[(m, mode)], N_PER_MODEL, SEED + i * 100)
        for c in picks:
            all_picks.append(c)

    # Shuffle order for blind reading
    rng = random.Random(SEED + 9999)
    rng.shuffle(all_picks)

    label_key = {}
    for i, c in enumerate(all_picks, 1):
        label_key[i] = {
            "filename": c["_filename"],
            "model": c["model_short"],
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
        print(f"  Case #{i:02d}: model={lk['model']:<32} | mode={lk['mode']:<3} | "
              f"q={lk['q_type']:<16} | pc={lk['problem_class']:<5} | chose={lk['choice']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
