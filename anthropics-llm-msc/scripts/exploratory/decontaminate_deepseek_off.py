#!/usr/bin/env python3
"""Iteratively re-fire contaminated DeepSeek-off cells until clean.

A 'contaminated' cell is one where we sent reasoning_config={"enabled": False}
but the response came back with reasoning_tokens > 0 — i.e., OR or DeepSeek
didn't honor the disable flag. Empirical testing shows this is intermittent
(re-firing the same prompt typically lands a clean response). This script
iterates: identify contaminated cells, re-fire each in place, check results.
Caps at MAX_ITERS rounds — any cell still contaminated after that is a
diagnostic finding (suggests deterministic prompt-specific leak).

Re-derives correct_capability_answer / ssa_aligned / sia_aligned on the new
response using the cell's own row_order from metadata (matches the post-bug
recovery flow).

Usage:
  python scripts/decontaminate_deepseek_off.py
"""
from __future__ import annotations

import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from framework import MODEL_CONFIG, NewcombExperiment  # type: ignore  # noqa: E402

MAIN_RUN_DIR = Path("experiment_results/main_run_20260516")
TARGET_MODEL = "deepseek/deepseek-v4-pro"
MAX_ITERS = 5
PARALLEL_WORKERS = 8
PROVIDER_SEMAPHORE = threading.Semaphore(4)


def is_contaminated(cell: dict) -> bool:
    u = cell.get("usage_statistics") or {}
    rt = u.get("reasoning_tokens", 0) or 0
    return rt > 0


def find_contaminated() -> list[Path]:
    out = []
    for f in MAIN_RUN_DIR.glob("*deepseek-v4-pro_off.json"):
        d = json.load(open(f))
        if is_contaminated(d):
            out.append(f)
    return out


def refire_cell(exp: NewcombExperiment, cell_file: Path) -> tuple[Path, bool, str]:
    """Re-fire one cell. Returns (file, success, status_message).

    success means the new response is clean (rt=0). If False, either API
    error or still contaminated.
    """
    try:
        d = json.load(open(cell_file))
    except Exception as e:
        return (cell_file, False, f"json load error: {e}")

    messages = [
        {"role": "system", "content": d["system_prompt"]},
        {"role": "user", "content": d["user_prompt"]},
    ]
    cfg = MODEL_CONFIG[TARGET_MODEL]

    try:
        with PROVIDER_SEMAPHORE:
            response = exp._openrouter_call(
                TARGET_MODEL, messages,
                reasoning_config={"enabled": False},
                max_tokens=cfg["max_tokens"],
            )
    except Exception as e:
        return (cell_file, False, f"API error: {type(e).__name__}: {e}")

    if not getattr(response, "choices", None):
        return (cell_file, False, "malformed response (no choices)")

    choice_obj = response.choices[0]
    msg = choice_obj.message
    new_response_text = msg.content or ""
    new_reasoning_trace = getattr(msg, "reasoning", None)
    new_extracted_choice = exp.extract_final_answer(new_response_text) if new_response_text else None
    new_usage = exp._serialize_usage(response.usage) if response.usage is not None else {}

    new_rt = (new_usage.get("reasoning_tokens", 0) or 0)
    is_clean = new_rt == 0

    # Update the cell with new response data, keeping all metadata
    d["response"] = new_response_text
    d["reasoning_trace"] = new_reasoning_trace
    d["finish_reason"] = choice_obj.finish_reason
    d["extracted_choice"] = new_extracted_choice
    d["usage_statistics"] = new_usage
    d["model_id_openrouter"] = getattr(response, "model", d.get("model_id_openrouter"))
    d["openrouter_response_id"] = getattr(response, "id", d.get("openrouter_response_id"))
    # Keep cumulative re-fire history note
    history = d.get("_refire_history") or []
    history.append({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reason": "decontamination",
        "new_rt": new_rt,
        "is_clean": is_clean,
    })
    d["_refire_history"] = history

    # Re-derive correctness if a choice was extracted
    if new_extracted_choice:
        exp.problem_structure = d.get("preferred_actions") or {}
        alignment = exp.determine_alignment(
            new_extracted_choice, d["preferred_actions"], row_order=d["row_order"],
        )
        d.update(alignment)
        correctness = exp.check_correctness(
            new_extracted_choice, d["question_type"], d["preferred_actions"],
            row_order=d["row_order"],
        )
        if correctness is not None:
            d["correct_capability_answer"] = correctness

    # Write back
    with open(cell_file, "w") as f:
        json.dump(d, f, indent=2)

    return (cell_file, is_clean, "clean" if is_clean else f"still contaminated (rt={new_rt})")


def main() -> int:
    exp = NewcombExperiment(base_output_dir="/tmp/_decontaminate", temperature=0.8)

    persistent_failures: dict[str, int] = {}  # cell filename → #attempts
    for iter_num in range(1, MAX_ITERS + 1):
        contaminated = find_contaminated()
        print(f"\n=== Iter {iter_num}/{MAX_ITERS}: {len(contaminated)} contaminated cells ===")
        if not contaminated:
            print(f"All DeepSeek-off cells now have reasoning_tokens=0.")
            break

        # Re-fire in parallel with provider semaphore
        results = []
        with ThreadPoolExecutor(max_workers=PARALLEL_WORKERS) as ex:
            futures = {ex.submit(refire_cell, exp, f): f for f in contaminated}
            for fut in as_completed(futures):
                results.append(fut.result())

        n_clean_now = sum(1 for _, clean, _ in results if clean)
        n_still = len(results) - n_clean_now
        print(f"  After iter {iter_num}: {n_clean_now} cleaned, {n_still} still contaminated")

        # Track persistent failures
        for f, clean, _ in results:
            if not clean:
                persistent_failures[f.name] = persistent_failures.get(f.name, 0) + 1

    # Final report
    final_contaminated = find_contaminated()
    print(f"\n{'=' * 70}")
    print(f"FINAL STATE: {len(final_contaminated)} cells still contaminated after {iter_num} iterations")
    print(f"{'=' * 70}")
    if final_contaminated:
        print(f"\nPersistent-contamination cells (likely deterministic prompt-specific leaks):")
        for f in final_contaminated[:20]:
            attempts = persistent_failures.get(f.name, 1)
            d = json.load(open(f))
            rt = (d.get("usage_statistics") or {}).get("reasoning_tokens", 0)
            tmpl = d.get("template_name", "").replace("20260516_standard_", "")
            print(f"  {f.name[:60]:<60} attempts={attempts}  current_rt={rt}  template={tmpl}")
        if len(final_contaminated) > 20:
            print(f"  ... and {len(final_contaminated) - 20} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
