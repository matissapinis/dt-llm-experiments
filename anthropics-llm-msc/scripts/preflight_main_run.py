#!/usr/bin/env python3
"""Pre-flight check for Main run — verifies all 32 problem dirs load,
prompts format, ground truths compute, and the choice mappings via the
new 'high'/'low' labels resolve to the expected SSA-correct letter.
No API calls.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from framework import MODEL_CONFIG, NewcombExperiment  # type: ignore  # noqa: E402
from run_main_experiment import (  # type: ignore  # noqa: E402
    ACTIVE_MODELS, PROBLEM_BASES, ROW_ORDERS,
    QUESTION_TYPES, N_SAMPLES, enumerate_cells, make_problem_name,
)


def main() -> int:
    failures: list[str] = []

    cells = enumerate_cells()
    if len(cells) != 12:
        failures.append(f"cell enumeration: got {len(cells)}, expected 12")
    else:
        print(f"[OK] enumerate_cells() → {len(cells)} cells")

    excluded = {"moonshotai/kimi-k2.6", "z-ai/glm-5.1"}
    if not all(e not in ACTIVE_MODELS for e in excluded):
        failures.append("ACTIVE_MODELS unexpectedly includes Kimi or GLM")
    else:
        print(f"[OK] ACTIVE_MODELS correctly excludes {sorted(excluded)}")

    if ACTIVE_MODELS - set(MODEL_CONFIG.keys()):
        failures.append(f"unknown active models: {ACTIVE_MODELS - set(MODEL_CONFIG.keys())}")
    else:
        print(f"[OK] all {len(ACTIVE_MODELS)} active models are in MODEL_CONFIG")

    print(f"\n[checking 32 problem dirs (16 canonical + 16 scaled, both row orders) ...]")
    exp = NewcombExperiment(base_output_dir="/tmp/preflight_main_run", temperature=0.8)
    for problem_class, theme, param_label in PROBLEM_BASES:
        for row in ROW_ORDERS:
            problem = make_problem_name(problem_class, theme, param_label, row)
            try:
                exp.load_problem(problem)
                params = exp.generate_parameters(exp.param_config, exp.problem_structure)
                _eu, preferred = exp.compute_problem_groundtruth(params)
                user_prompt = exp.prompt_templates[problem].format(**params)
                problem_dir = exp.find_problem_dir(problem)
                system_prompts = {
                    q: (problem_dir / "system_prompts" / f"{q}.txt").read_text().strip()
                    for q in QUESTION_TYPES
                }
                if not user_prompt or any(not v for v in system_prompts.values()):
                    failures.append(f"{problem}: empty prompt produced")
                    continue
                # Verify the SSA capability prompt is SSA reference-class disambiguation (contains the diagnostic substring)
                if "with the reference class contained within each hypothesis separately" not in system_prompts["ssa_capability"]:
                    failures.append(f"{problem}: ssa_capability prompt is NOT SSA reference-class disambiguation")
                    continue
                # Verify choice mapping resolves correctly
                ssa_pref = exp.problem_structure.get("ssa_preference")
                sia_pref = exp.problem_structure.get("sia_preference")
                if ssa_pref is None or sia_pref is None:
                    failures.append(f"{problem}: missing ssa_preference / sia_preference")
                    continue
                cm = exp.get_choice_mapping(exp.row_order)
                if ssa_pref not in cm or sia_pref not in cm:
                    failures.append(f"{problem}: pref label not in choice mapping (cm={cm})")
                    continue
                ssa_letter = cm[ssa_pref]
                sia_letter = cm[sia_pref]
                print(f"[OK] {problem:<60} row_order={exp.row_order} ssa→{ssa_letter} sia→{sia_letter} preferred={preferred}")
            except Exception as e:
                failures.append(f"{problem}: {type(e).__name__}: {e}")
                continue

    planned = len(PROBLEM_BASES) * len(ROW_ORDERS) * len(QUESTION_TYPES) * N_SAMPLES * len(cells)
    print(f"\n[OK] calls planned for Main run: {planned}")

    print()
    if failures:
        print(f"FAIL: {len(failures)} issue(s):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("GREEN: Main run driver pre-flight passed. Safe to fire.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
