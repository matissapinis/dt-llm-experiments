#!/usr/bin/env python3
"""QT6 — False-positive / false-negative sampler for capability cells.

Stratified random sample of 10 "correct" + 10 "incorrect" capability cells
(ssa_capability + sia_capability), printed in anonymized form for blind
soundness coding.

For capability q-types:
  - sia_capability → "correct" = chose SIA-aligned letter (is_thirder=True)
  - ssa_capability → "correct" = chose SSA-aligned letter (is_thirder=False)

The script prints each cell's prompt + reasoning trace + extracted choice,
WITHOUT revealing whether it was correct per the formalism. Soundness
judgments are made blind; then the script reveals labels and tabulates
FP rate (correct + unsound reasoning) and FN rate (incorrect + sound reasoning).

Stratification: sample roughly evenly across (model, mode) to surface a
representative slice rather than over-weighting the high-volume models.
"""
from __future__ import annotations

import json
import random
import re
from collections import defaultdict
from pathlib import Path

D = Path("experiment_results/main_run_20260516")
SEED = 20260518
N_CORRECT = 10
N_INCORRECT = 10

# Cases already coded in the original 20-case pass; exclude on rerun so the
# expansion pass surfaces *new* cells (rather than re-drawing the same ones).
ALREADY_CODED_FILES = {
    "20260516_standard_padd_aiinstance_12_row12_ssa_capability_sample5_openai_gpt-5.5_off.json",
    "20260516_standard_sb_aiinstance_12_row12_ssa_capability_sample4_google_gemini-3.1-pro-preview_on.json",
    "20260516_standard_dd_aiinstance_21_row21_sia_capability_sample1_openai_gpt-5.5_on.json",
    "20260516_standard_dd_civilization_21_row21_sia_capability_sample2_x-ai_grok-4.3_on.json",
    "20260516_standard_sb_classic_scaled_21_row21_ssa_capability_sample5_qwen_qwen3.6-max-preview_off.json",
    "20260516_standard_padd_civilization_scaled_12_row12_ssa_capability_sample9_x-ai_grok-4.3_off.json",
    "20260516_standard_inc_aiinstance_21_row21_ssa_capability_sample9_openai_gpt-5.5_off.json",
    "20260516_standard_padd_aiinstance_scaled_12_row12_sia_capability_sample8_x-ai_grok-4.3_off.json",
    "20260516_standard_dd_aiinstance_12_row12_ssa_capability_sample2_anthropic_claude-opus-4.7_on.json",
    "20260516_standard_padd_civilization_scaled_12_row12_ssa_capability_sample2_deepseek_deepseek-v4-pro_off.json",
    "20260516_standard_dd_aiinstance_scaled_21_row21_sia_capability_sample2_deepseek_deepseek-v4-pro_on.json",
    "20260516_standard_inc_aiinstance_scaled_21_row21_ssa_capability_sample6_deepseek_deepseek-v4-pro_off.json",
    "20260516_standard_sb_aiinstance_scaled_21_row21_ssa_capability_sample7_anthropic_claude-opus-4.7_off.json",
    "20260516_standard_inc_aiinstance_scaled_12_row12_ssa_capability_sample6_deepseek_deepseek-v4-pro_off.json",
    "20260516_standard_inc_classic_12_row12_ssa_capability_sample7_x-ai_grok-4.3_on.json",
    "20260516_standard_dd_civilization_scaled_12_row12_sia_capability_sample3_qwen_qwen3.6-max-preview_on.json",
    "20260516_standard_inc_aiinstance_scaled_12_row12_ssa_capability_sample9_google_gemini-3-flash-preview_off.json",
    "20260516_standard_dd_civilization_scaled_21_row21_ssa_capability_sample7_qwen_qwen3.6-max-preview_on.json",
    "20260516_standard_dd_aiinstance_12_row12_sia_capability_sample9_qwen_qwen3.6-max-preview_off.json",
    "20260516_standard_dd_civilization_scaled_12_row12_sia_capability_sample4_deepseek_deepseek-v4-pro_on.json",
    # Second batch (cases #21-50 in cumulative numbering)
    "20260516_standard_inc_classic_12_row12_ssa_capability_sample2_qwen_qwen3.6-max-preview_off.json",
    "20260516_standard_sb_classic_21_row21_sia_capability_sample2_google_gemini-3-flash-preview_off.json",
    "20260516_standard_inc_aiinstance_scaled_12_row12_ssa_capability_sample9_qwen_qwen3.6-max-preview_on.json",
    "20260516_standard_inc_classic_12_row12_ssa_capability_sample5_google_gemini-3-flash-preview_off.json",
    "20260516_standard_sb_aiinstance_12_row12_ssa_capability_sample8_google_gemini-3.1-pro-preview_on.json",
    "20260516_standard_dd_civilization_scaled_21_row21_ssa_capability_sample6_openai_gpt-5.5_off.json",
    "20260516_standard_dd_aiinstance_scaled_12_row12_ssa_capability_sample1_openai_gpt-5.5_off.json",
    "20260516_standard_sb_classic_scaled_21_row21_sia_capability_sample7_deepseek_deepseek-v4-pro_off.json",
    "20260516_standard_inc_aiinstance_scaled_12_row12_ssa_capability_sample2_deepseek_deepseek-v4-pro_on.json",
    "20260516_standard_sb_classic_scaled_21_row21_sia_capability_sample8_qwen_qwen3.6-max-preview_off.json",
    "20260516_standard_inc_classic_scaled_12_row12_ssa_capability_sample6_qwen_qwen3.6-max-preview_on.json",
    "20260516_standard_padd_civilization_12_row12_sia_capability_sample8_x-ai_grok-4.3_off.json",
    "20260516_standard_sb_aiinstance_scaled_12_row12_sia_capability_sample9_anthropic_claude-opus-4.7_off.json",
    "20260516_standard_padd_civilization_scaled_21_row21_sia_capability_sample3_openai_gpt-5.5_off.json",
    "20260516_standard_sb_classic_scaled_12_row12_ssa_capability_sample4_openai_gpt-5.5_off.json",
    "20260516_standard_sb_aiinstance_scaled_12_row12_sia_capability_sample3_openai_gpt-5.5_on.json",
    "20260516_standard_inc_classic_scaled_21_row21_sia_capability_sample6_qwen_qwen3.6-max-preview_on.json",
    "20260516_standard_sb_classic_21_row21_ssa_capability_sample5_google_gemini-3-flash-preview_off.json",
    "20260516_standard_inc_classic_scaled_21_row21_sia_capability_sample7_anthropic_claude-opus-4.7_on.json",
    "20260516_standard_padd_civilization_scaled_12_row12_ssa_capability_sample2_x-ai_grok-4.3_on.json",
    "20260516_standard_sb_classic_21_row21_ssa_capability_sample8_qwen_qwen3.6-max-preview_off.json",
    "20260516_standard_sb_aiinstance_21_row21_ssa_capability_sample8_deepseek_deepseek-v4-pro_off.json",
    "20260516_standard_inc_aiinstance_scaled_12_row12_ssa_capability_sample9_x-ai_grok-4.3_off.json",
    "20260516_standard_sb_aiinstance_21_row21_ssa_capability_sample3_deepseek_deepseek-v4-pro_off.json",
    "20260516_standard_sb_aiinstance_scaled_12_row12_ssa_capability_sample9_anthropic_claude-opus-4.7_on.json",
    "20260516_standard_dd_civilization_scaled_12_row12_ssa_capability_sample5_x-ai_grok-4.3_off.json",
    "20260516_standard_sb_classic_12_row12_sia_capability_sample3_deepseek_deepseek-v4-pro_on.json",
    "20260516_standard_inc_classic_scaled_12_row12_sia_capability_sample4_qwen_qwen3.6-max-preview_off.json",
    "20260516_standard_sb_aiinstance_scaled_12_row12_ssa_capability_sample9_deepseek_deepseek-v4-pro_on.json",
    "20260516_standard_dd_aiinstance_scaled_12_row12_sia_capability_sample8_x-ai_grok-4.3_on.json",
}


def parse_mode(filename: str) -> str:
    if filename.endswith("_on.json"):
        return "on"
    if filename.endswith("_off.json"):
        return "off"
    return "?"


def get_sia_aligned_letter(preferred_actions: dict, row_order: str) -> str | None:
    if not preferred_actions:
        return None
    sia_pref = preferred_actions.get("sia_preference")
    if not sia_pref:
        return None
    is_A_in_row12 = sia_pref in ("half", "high")
    return ("A" if is_A_in_row12 else "B") if row_order == "12" else ("B" if is_A_in_row12 else "A")


def load_capability_cells():
    cells = []
    for f in sorted(D.glob("*.json")):
        d = json.load(open(f))
        ch = d.get("extracted_choice")
        if ch not in ("A", "B"):
            continue
        qt = d.get("question_type")
        if qt not in ("sia_capability", "ssa_capability"):
            continue
        model = d.get("model_id_openrouter") or ""
        d["mode"] = parse_mode(f.name)
        d["model_short"] = model.split("/")[-1]
        d["_filename"] = f.name
        sia_letter = get_sia_aligned_letter(d.get("preferred_actions"), d.get("row_order"))
        if sia_letter is None:
            continue
        d["sia_letter"] = sia_letter
        is_thirder = (ch == sia_letter)
        # Correctness per q-type
        if qt == "sia_capability":
            d["is_correct"] = is_thirder
        else:  # ssa_capability
            d["is_correct"] = not is_thirder
        # Need a non-empty reasoning trace OR a non-trivial response
        rt = d.get("reasoning_trace") or ""
        resp = d.get("response") or ""
        d["_trace_or_resp"] = rt if rt.strip() else resp
        if not d["_trace_or_resp"].strip():
            continue
        if f.name in ALREADY_CODED_FILES:
            continue
        cells.append(d)
    return cells


def stratified_sample(cells, n, predicate, seed):
    """Sample n cells matching predicate, stratifying over (model, mode)."""
    rng = random.Random(seed)
    pool = [c for c in cells if predicate(c)]
    if not pool:
        return []
    by_mm = defaultdict(list)
    for c in pool:
        by_mm[(c["model_short"], c["mode"])].append(c)
    mms = sorted(by_mm.keys())
    rng.shuffle(mms)
    picks = []
    # Round-robin from each (mm) until n reached
    cursor = 0
    while len(picks) < n and any(by_mm[mm] for mm in mms):
        mm = mms[cursor % len(mms)]
        if by_mm[mm]:
            pick_idx = rng.randrange(len(by_mm[mm]))
            picks.append(by_mm[mm].pop(pick_idx))
        cursor += 1
        if cursor > len(mms) * 100:
            break
    return picks[:n]


def render_anonymized(idx, cell):
    """Print a cell for blind reading: prompt + trace + extracted choice. NO labels."""
    print(f"\n{'#' * 100}")
    print(f"CASE #{idx:02d}")
    print(f"{'#' * 100}")
    print(f"[meta — not part of the read, just identifies the cell]")
    print(f"  filename:   {cell['_filename']}")
    print(f"  q-type:     {cell.get('question_type')}")
    print(f"  row order:  {cell.get('row_order')}")
    print()
    print("PROMPT:")
    print("-" * 100)
    up = cell.get("user_prompt", "")
    print(up)
    print()
    print("REASONING / RESPONSE:")
    print("-" * 100)
    print(cell["_trace_or_resp"])
    print()
    print(f"EXTRACTED CHOICE: {cell.get('extracted_choice')}")


def main() -> int:
    cells = load_capability_cells()
    print(f"Loaded {len(cells)} capability cells with non-empty trace/response and known SIA letter")
    n_correct_pool = sum(1 for c in cells if c["is_correct"])
    n_incorrect_pool = sum(1 for c in cells if not c["is_correct"])
    print(f"  correct pool: {n_correct_pool}, incorrect pool: {n_incorrect_pool}")

    correct_picks = stratified_sample(cells, N_CORRECT, lambda c: c["is_correct"], seed=SEED)
    incorrect_picks = stratified_sample(cells, N_INCORRECT, lambda c: not c["is_correct"], seed=SEED + 1)

    # Combine + shuffle into a randomized presentation order
    combined = correct_picks + incorrect_picks
    rng = random.Random(SEED + 2)
    rng.shuffle(combined)

    # Save label-key for later reveal
    label_key = {i + 1: {"_filename": c["_filename"], "is_correct": c["is_correct"],
                          "model_short": c["model_short"], "mode": c["mode"],
                          "q_type": c.get("question_type"),
                          "extracted_choice": c.get("extracted_choice"),
                          "sia_letter": c["sia_letter"]}
                 for i, c in enumerate(combined)}

    # Print each case anonymized
    for i, c in enumerate(combined, start=1):
        render_anonymized(i, c)

    # Print the label key at the end (for revelation after coding)
    print(f"\n\n{'=' * 100}")
    print("LABEL KEY (revealed after blind coding)")
    print(f"{'=' * 100}")
    for idx, lk in sorted(label_key.items()):
        verdict = "CORRECT" if lk["is_correct"] else "INCORRECT"
        print(f"  Case #{idx:02d}: {verdict:<9} | {lk['model_short']:<32} | {lk['mode']:<3} | "
              f"{lk['q_type']:<16} | chose={lk['extracted_choice']} | SIA-letter={lk['sia_letter']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
