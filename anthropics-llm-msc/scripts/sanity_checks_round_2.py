#!/usr/bin/env python3
"""Round-2 additional sanity checks on Main run (post-DeepSeek-decontamination).

Round 1 surfaced two real issues (row_order bug, DeepSeek-off contamination).
This round hits checks that could expose additional problems:

  1. RE-VERIFY SSA reference-class disambiguation validation — does decontamination change the picture?
  2. RE-VERIFY capability accuracy — esp. DeepSeek (changed substantially)
  3. OPUS-ON rt=0 deep dive — symmetric to DeepSeek issue (115 Opus-on cells
     had rt=0 despite reasoning_on). Are these legitimate or another leak?
  4. Per-(model, mode) cell count — each should have exactly 1,152 cells
     (32 dirs × 4 q-types × 9 samples = 1,152). Detect any missing cells.
  5. DataFrame uniqueness — no two cells should share the same identifying
     tuple (template, q-type, sample, model, mode).
  6. Cross-q-type within-model consistency — for the same (model, problem, row),
     does ssa_capability and personal_attitude pattern correlate as expected?
  7. Extracted-choice values — should be only "A", "B", or None. Any others?
  8. Scaled-prompt content verification — verify the scaled templates have
     the expected "200 billion" / "0.9" etc. content.
  9. Decontamination didn't break derived fields — spot-check a decontaminated
     DeepSeek-off cell to verify correct_capability_answer is right.

Usage:
  python scripts/sanity_checks_round2.py
"""
from __future__ import annotations

import json
from collections import defaultdict, Counter
from pathlib import Path

MAIN_RUN_DIR = Path("experiment_results/main_run_20260516")


def load_cells(out_dir: Path) -> list[dict]:
    return [json.load(open(f)) for f in sorted(out_dir.glob("*.json"))]


def parse_meta(t: str) -> dict:
    parts = t.replace("20260516_standard_", "").replace("20260510_standard_", "")
    rest, row = parts.rsplit("_", 1)
    param = "canonical"
    if rest.endswith("_scaled"):
        param = "scaled"
        rest = rest[:-len("_scaled")]
    pc, theme = rest.split("_", 1)
    return {"problem_class": pc, "theme": theme, "parameterization": param, "row": row}



def s2_capability_revisited(cells: list[dict]) -> None:
    print("=" * 80)
    print("2. CAPABILITY ACCURACY — re-verified, focus on DeepSeek (decontaminated)")
    print("=" * 80)
    by = defaultdict(lambda: {"correct": 0, "total": 0})
    for c in cells:
        if c.get("question_type") not in ("ssa_capability", "sia_capability"):
            continue
        k = (c["model"], c["reasoning_mode"])
        if c.get("correct_capability_answer") is True: by[k]["correct"] += 1
        if c.get("correct_capability_answer") in (True, False): by[k]["total"] += 1
    print(f"{'model':<42} {'mode':>4} {'correct':>8} {'total':>6} {'acc':>6}")
    for (m, mode), v in sorted(by.items()):
        acc = v["correct"] / v["total"] if v["total"] else 0
        mark = " ← changed?" if "deepseek" in m else ""
        print(f"{m:<42} {mode:>4} {v['correct']:>8} {v['total']:>6} {acc:>6.1%}{mark}")
    print()


def s3_opus_on_rt0_deepdive(cells: list[dict]) -> None:
    print("=" * 80)
    print("3. OPUS-ON rt=0 DEEP DIVE — 115 cells had rt=0 despite reasoning_on")
    print("=" * 80)
    sub = []
    for c in cells:
        if c["model"] != "anthropic/claude-opus-4.7": continue
        if c["reasoning_mode"] != "on": continue
        rt = (c.get("usage_statistics") or {}).get("reasoning_tokens", 0) or 0
        sub.append((rt, c))
    rt0 = [c for rt, c in sub if rt == 0]
    rt_pos = [c for rt, c in sub if rt > 0]
    print(f"  Opus-on cells: {len(sub)} total, {len(rt0)} with rt=0, {len(rt_pos)} with rt>0")
    # Sample rt0 cells — what do they look like?
    print(f"\n  Sample of 3 Opus-on rt=0 cells:")
    for c in rt0[:3]:
        u = c.get("usage_statistics") or {}
        resp = (c.get("response") or "")
        rtrace = (c.get("reasoning_trace") or "")
        print(f"    --- {c.get('template_name', '').replace('20260516_standard_', '')[:40]} {c['question_type']} sample {c['run_number']} ---")
        print(f"      completion_tokens: {u.get('completion_tokens')}  cost: ${u.get('cost', 0):.4f}")
        print(f"      response_len: {len(resp)}, reasoning_trace_len: {len(rtrace)}")
        print(f"      response (first 200 chars): {resp[:200]}")
    # Compare token-cost ratios: should be lower for rt=0 cells (no reasoning compute)
    if rt0 and rt_pos:
        avg_cost_rt0 = sum((c.get("usage_statistics") or {}).get("cost", 0) or 0 for c in rt0) / len(rt0)
        avg_cost_rtp = sum((c.get("usage_statistics") or {}).get("cost", 0) or 0 for c in rt_pos) / len(rt_pos)
        print(f"\n  Avg cost rt=0 Opus-on cells: ${avg_cost_rt0:.4f}")
        print(f"  Avg cost rt>0 Opus-on cells: ${avg_cost_rtp:.4f}")
        print(f"  Ratio: {avg_cost_rt0/avg_cost_rtp:.2f}x ({'rt=0 is cheaper, as expected' if avg_cost_rt0 < avg_cost_rtp else 'rt=0 cheap effect unclear'})")
    print()


def s4_cell_count_per_model_mode(cells: list[dict]) -> None:
    print("=" * 80)
    print("4. CELL COUNT per (model, mode) — should be exactly 1152 each")
    print("=" * 80)
    by = Counter((c["model"], c["reasoning_mode"]) for c in cells)
    expected = 1152
    print(f"{'model':<42} {'mode':>4} {'count':>5} {'expected':>9} {'diff':>5}")
    for (m, mode), n in sorted(by.items()):
        diff = n - expected
        flag = "" if diff == 0 else "  ← MISMATCH"
        print(f"{m:<42} {mode:>4} {n:>5} {expected:>9} {diff:>+5}{flag}")
    print()


def s5_uniqueness_check(cells: list[dict]) -> None:
    print("=" * 80)
    print("5. UNIQUENESS — no two cells should share the same identifying tuple")
    print("=" * 80)
    tuples = [(c.get("template_name"), c.get("question_type"), c.get("run_number"), c["model"], c["reasoning_mode"]) for c in cells]
    cnt = Counter(tuples)
    dupes = [(k, v) for k, v in cnt.items() if v > 1]
    if dupes:
        print(f"  FOUND {len(dupes)} duplicates!")
        for k, v in dupes[:10]:
            print(f"    {k}: {v}x")
    else:
        print(f"  ✓ all {len(tuples)} cells have unique (template, q-type, sample, model, mode) tuples")
    print()


def s6_cross_qtype_within_model(cells: list[dict]) -> None:
    print("=" * 80)
    print("6. CROSS-Q-TYPE CONSISTENCY WITHIN MODEL")
    print("=" * 80)
    print("For each model: does the model's SSA-aligned% in personal_attitude track")
    print("how often the model picks SSA in ssa_capability? Coherence within model.\n")
    by = defaultdict(lambda: defaultdict(lambda: {"ssa_aligned": 0, "n": 0}))
    for c in cells:
        if c.get("extracted_choice") not in ("A", "B"): continue
        k = c["model"]
        q = c["question_type"]
        by[k][q]["n"] += 1
        if c.get("ssa_aligned"): by[k][q]["ssa_aligned"] += 1
    print(f"{'model':<42} {'ssa_cap%':>9} {'sia_cap%':>9} {'norm_att%':>11} {'pers_att%':>11}")
    print("-" * 90)
    for m in sorted(by.keys()):
        rates = {q: by[m][q]["ssa_aligned"] / by[m][q]["n"] if by[m][q]["n"] else 0
                 for q in ("ssa_capability", "sia_capability", "normative_attitude", "personal_attitude")}
        print(f"{m:<42} {rates['ssa_capability']:>8.1%} {rates['sia_capability']:>8.1%} "
              f"{rates['normative_attitude']:>10.1%} {rates['personal_attitude']:>10.1%}")
    print("\n  (high ssa_cap% = model identifies SSA correctly; high pers_att% = picks SSA when given choice)")
    print()


def s7_extracted_choice_values(cells: list[dict]) -> None:
    print("=" * 80)
    print("7. EXTRACTED_CHOICE VALUE DISTRIBUTION")
    print("=" * 80)
    cnt = Counter(c.get("extracted_choice") for c in cells)
    print(f"  Value distribution:")
    for v, n in sorted(cnt.items(), key=lambda x: (x[0] is None, str(x[0]))):
        marker = "" if v in ("A", "B", None) else "  ← UNEXPECTED"
        print(f"    {repr(v):<15}: {n:>6}{marker}")
    print()


def s8_scaled_prompt_content(cells: list[dict]) -> None:
    print("=" * 80)
    print("8. SCALED PROMPT CONTENT VERIFICATION")
    print("=" * 80)
    # For each scaled problem, verify the user_prompt contains expected text
    expected_markers = {
        "sb_classic_scaled": ["200 billion", "200 trillion", "0.9", "0.0089"],
        "sb_aiinstance_scaled": ["200 billion", "200 trillion", "0.9", "AI instance"],
        "inc_classic_scaled": ["200 billion", "200 trillion", "windowless"],
        "inc_aiinstance_scaled": ["200 billion", "200 trillion", "AI instances"],
        "dd_civilization_scaled": ["200 billion", "200 trillion", "0.1", "0.991", "100 billion"],
        "dd_aiinstance_scaled": ["200 billion", "200 trillion", "cohort", "100 billion"],
        "padd_civilization_scaled": ["200 billion", "200 trillion", "reverse birth rank", "0.1", "0.991"],
        "padd_aiinstance_scaled": ["200 billion", "200 trillion", "reverse sequence position"],
    }
    misses = []
    for c in cells:
        tmpl = c.get("template_name", "").replace("20260516_standard_", "")
        # strip row suffix
        tmpl_base = tmpl.rsplit("_", 1)[0]
        if tmpl_base not in expected_markers: continue
        prompt = c.get("user_prompt") or ""
        for marker in expected_markers[tmpl_base]:
            if marker not in prompt:
                misses.append((c.get("template_name"), marker))
                break
    if misses:
        print(f"  FOUND {len(misses)} cells with missing markers (showing first 5):")
        for tmpl, marker in misses[:5]:
            print(f"    {tmpl}: missing {repr(marker)}")
    else:
        print(f"  ✓ all scaled-problem cells contain expected markers in user_prompt")
    print()


def s9_decontamination_spotcheck(cells: list[dict]) -> None:
    print("=" * 80)
    print("9. DECONTAMINATION SPOT-CHECK — derived fields correct on re-fired cells?")
    print("=" * 80)
    # Find DeepSeek-off cells that have a _refire_history field (those were decontaminated)
    refired = [c for c in cells
               if c["model"] == "deepseek/deepseek-v4-pro" and c["reasoning_mode"] == "off"
               and c.get("_refire_history")]
    print(f"  decontaminated DeepSeek-off cells (have _refire_history): {len(refired)}")
    if not refired:
        print("  (none found — check that decontamination script ran successfully)")
        return
    # Spot-check 3 cells: verify correct_capability_answer matches what we'd expect
    print(f"\n  Spot-checking 3 cells — verifying correct_capability_answer matches expected:")
    sample = refired[:3]
    for c in sample:
        u = c.get("usage_statistics") or {}
        rt = u.get("reasoning_tokens", 0) or 0
        choice = c.get("extracted_choice")
        preferred = c.get("preferred_actions") or {}
        row = c.get("row_order")
        q_type = c.get("question_type")
        # For capability questions, manually compute expected
        if q_type in ("ssa_capability", "sia_capability"):
            theory = q_type[:-len("_capability")]
            pref = preferred.get(f"{theory}_preference")
            # Mapping: row=12: half/high→A, third/low→B; row=21: swapped
            if row == "12":
                pref_letter = {"half": "A", "third": "B", "high": "A", "low": "B"}.get(pref)
            else:
                pref_letter = {"half": "B", "third": "A", "high": "B", "low": "A"}.get(pref)
            expected_correct = choice == pref_letter
            actual = c.get("correct_capability_answer")
            match = "✓" if actual == expected_correct else "✗ MISMATCH"
            print(f"    {c.get('template_name', '')[-40:]} {q_type} sample {c['run_number']}")
            print(f"      choice={choice} pref={pref}({pref_letter}) → expected correct={expected_correct}, actual={actual} {match}")
            print(f"      rt={rt} (should be 0 after decontamination)")
        else:
            print(f"    {c.get('template_name', '')[-40:]} {q_type} sample {c['run_number']} (attitude, no correctness)")
            print(f"      rt={rt} (should be 0 after decontamination)")
    print()


def main() -> int:
    cells = load_cells(MAIN_RUN_DIR)
    print(f"loaded {len(cells)} Main run cells\n")
    s2_capability_revisited(cells)
    s3_opus_on_rt0_deepdive(cells)
    s4_cell_count_per_model_mode(cells)
    s5_uniqueness_check(cells)
    s6_cross_qtype_within_model(cells)
    s7_extracted_choice_values(cells)
    s8_scaled_prompt_content(cells)
    s9_decontamination_spotcheck(cells)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
