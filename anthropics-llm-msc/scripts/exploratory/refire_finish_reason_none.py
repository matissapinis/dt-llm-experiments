#!/usr/bin/env python3
"""Re-fire exactly 3 cells with finish_reason=None and coherent-mid-derivation cutoff.

These 3 cells have a signature consistent only with infrastructure stream-truncation:
coherent English text cut off mid-equation. (Distinct from the 30 'mixed-language
garbage' no_final_answer cells, which have an ambiguous model-vs-infrastructure
cause and so are not re-fired.)

Criterion (applied to all 13,824 cells, identifies exactly these 3):
  - finish_reason is None
  - response is non-empty and ends mid-derivation (no FINAL ANSWER)
  - extracted_choice is None
  - response content is coherent English (not garbage tokens)

Single attempt per cell. Records _refire_history. Re-derives correctness/alignment
if the new response produces a valid A/B answer.

Usage:
  python scripts/refire_finish_reason_none.py
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from framework import MODEL_CONFIG, NewcombExperiment  # type: ignore  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from two_stage_parse_main_run import two_stage_parse  # type: ignore  # noqa: E402

MAIN_RUN_DIR = Path("experiment_results/main_run_20260516")

TARGETS = [
    "20260516_standard_inc_aiinstance_scaled_21_row21_sia_capability_sample5_deepseek_deepseek-v4-pro_off.json",
    "20260516_standard_inc_aiinstance_scaled_21_row21_sia_capability_sample5_deepseek_deepseek-v4-pro_on.json",
    "20260516_standard_inc_classic_scaled_12_row12_normative_attitude_sample4_deepseek_deepseek-v4-pro_off.json",
]

PROVIDER_SEM = threading.Semaphore(2)  # gentle, 3 cells total


def refire_one(exp: NewcombExperiment, cell_file: Path) -> str:
    with open(cell_file) as f:
        d = json.load(f)

    # cell's model_id_openrouter has a date suffix (e.g. "deepseek/deepseek-v4-pro-20260423");
    # MODEL_CONFIG is keyed by the request-time id without suffix. Strip to find the config key.
    raw_model = d["model_id_openrouter"]
    model = next((k for k in MODEL_CONFIG if raw_model.startswith(k)), None)
    if model is None:
        return f"no MODEL_CONFIG match for {raw_model}"
    mode_on = cell_file.name.endswith("_on.json")
    cfg = MODEL_CONFIG[model]
    messages = [
        {"role": "system", "content": d["system_prompt"]},
        {"role": "user", "content": d["user_prompt"]},
    ]
    if mode_on:
        reasoning_config = cfg.get("reasoning_config_on")
    else:
        reasoning_config = {"enabled": False}

    try:
        with PROVIDER_SEM:
            response = exp._openrouter_call(
                model, messages,
                reasoning_config=reasoning_config,
                max_tokens=cfg["max_tokens"],
            )
    except Exception as e:
        return f"API error: {type(e).__name__}: {e}"

    if not getattr(response, "choices", None):
        return "malformed response (no choices)"

    choice_obj = response.choices[0]
    msg = choice_obj.message
    new_response_text = msg.content or ""
    new_reasoning_trace = getattr(msg, "reasoning", None)
    new_finish_reason = choice_obj.finish_reason
    new_usage = exp._serialize_usage(response.usage) if response.usage is not None else {}

    # Re-parse via two-stage parser
    new_choice, new_quality = two_stage_parse(new_response_text)

    # Update cell
    d["response"] = new_response_text
    d["reasoning_trace"] = new_reasoning_trace
    d["finish_reason"] = new_finish_reason
    d["extracted_choice"] = new_choice
    d["parse_quality"] = new_quality
    d["usage_statistics"] = new_usage
    d["model_id_openrouter"] = getattr(response, "model", model)
    d["openrouter_response_id"] = getattr(response, "id", None)

    history = d.get("_refire_history") or []
    history.append({
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "reason": "finish_reason_none_infrastructure_cutoff",
        "new_finish_reason": new_finish_reason,
        "new_parse_quality": new_quality,
        "new_extracted_choice": new_choice,
        "new_response_len": len(new_response_text),
    })
    d["_refire_history"] = history

    # Re-derive correctness / alignment if valid choice
    if new_choice in ("A", "B"):
        pref = d.get("preferred_actions") or {}
        row = d.get("row_order")
        qt = d.get("question_type")
        if pref and row:
            exp.problem_structure = pref
            alignment = exp.determine_alignment(new_choice, pref, row_order=row)
            d.update(alignment)
            if qt and qt.endswith("_capability"):
                corr = exp.check_correctness(new_choice, qt, pref, row_order=row)
                if corr is not None:
                    d["correct_capability_answer"] = corr
    else:
        # No valid choice → strip any stale derived fields
        for k in ("correct_capability_answer", "ssa_aligned", "sia_aligned",
                  "cdt_aligned", "edt_aligned"):
            d.pop(k, None)

    with open(cell_file, "w") as f:
        json.dump(d, f, indent=2)

    return (f"finish_reason={new_finish_reason}  parse_quality={new_quality}  "
            f"choice={new_choice}  resp_len={len(new_response_text)}")


def main() -> int:
    exp = NewcombExperiment(base_output_dir="/tmp/_refire_fr_none", temperature=0.8)

    for fname in TARGETS:
        fp = MAIN_RUN_DIR / fname
        if not fp.exists():
            print(f"  ✗ NOT FOUND: {fname}")
            continue
        # Pre-check: matches the criterion?
        d = json.load(open(fp))
        pre_fr = d.get("finish_reason")
        pre_ch = d.get("extracted_choice")
        pre_pq = d.get("parse_quality")
        pre_len = len(d.get("response") or "")
        print(f"  → re-firing {fname}")
        print(f"    pre:  finish={pre_fr}  pq={pre_pq}  ch={pre_ch}  resp_len={pre_len}")
        result = refire_one(exp, fp)
        print(f"    post: {result}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
