#!/usr/bin/env python3
"""Multi-model availability + reasoning-mode probe for the MSc lineup.

For each candidate model, runs two calls through the migrated framework
(reasoning OFF + reasoning MAX), then classifies the model's reasoning-mode
support based on (a) whether each call succeeded and (b) whether the
reasoning_tokens count is zero vs nonzero.

Uses the same MSc problem and question type as the original probe
(20260510_standard_sb_firewall_12 / personal_attitude) so token counts are
roughly comparable across models.

Usage:
  python src/check_models_msc.py
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent))
from framework import NewcombExperiment  # type: ignore  # noqa: E402


MODELS = [
    "anthropic/claude-opus-4.7",
    # Gemini 3.1 Pro Preview is reasoning-only on OpenRouter
    # ("Reasoning is mandatory for this endpoint and cannot be disabled");
    # Gemini 3 Flash Preview is the within-family hybrid used as its
    # non-reasoning counterpart.
    "google/gemini-3.1-pro-preview",
    "google/gemini-3-flash-preview",
    "openai/gpt-5.5",
    "x-ai/grok-4.3",
    "z-ai/glm-5.1",
    "qwen/qwen3.6-max-preview",
    "moonshotai/kimi-k2.6",
    "deepseek/deepseek-v4-pro",
]
PROBLEM = "20260516_standard_sb_classic_12"
QUESTION_TYPE = "personal_attitude"
TEMPERATURE = 0.8
MAX_TOKENS = 128000
REASONING_MAX_TOKENS = 60000


def run_one(model: str, reasoning_config: dict, label: str, out_dir: Path) -> dict:
    """Return a compact summary dict for one (model, reasoning_config) call."""
    summary: dict = {
        "model": model, "label": label, "reasoning_config": reasoning_config,
        "ok": False, "error": None,
        "extracted": None, "finish_reason": None,
        "completion_tokens": None, "reasoning_tokens": None, "cost": None,
    }
    try:
        exp = NewcombExperiment(
            base_output_dir=str(out_dir),
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            reasoning_config=reasoning_config,
        )
        # Skip catalog re-validation per call — we already confirmed availability.
        exp.models = [model]
        exp.load_problem(PROBLEM)
        results = exp.run_experiments_with_question_types(
            question_types=[QUESTION_TYPE],
            repeats_per_model=1,
            display_examples=False,
        )
        runs = results.get(model, [])
        if not runs:
            summary["error"] = "no_runs_returned"
            return summary
        r = runs[0]
        usage = r.get("usage_statistics") or {}
        summary.update(
            ok=True,
            extracted=r.get("extracted_choice"),
            finish_reason=r.get("finish_reason"),
            completion_tokens=usage.get("completion_tokens"),
            reasoning_tokens=usage.get("reasoning_tokens", 0),
            cost=usage.get("cost"),
        )
    except Exception as e:
        summary["error"] = f"{type(e).__name__}: {e}"
    return summary


def classify(off: dict, on: dict) -> str:
    if not off["ok"] and not on["ok"]:
        return "BROKEN (both configs failed)"
    if off["ok"] and not on["ok"]:
        return "NON-REASONING-ONLY (reasoning config rejected)"
    if not off["ok"] and on["ok"]:
        return "REASONING-ONLY (disable rejected)"
    # Both succeeded:
    off_rt = off.get("reasoning_tokens") or 0
    on_rt = on.get("reasoning_tokens") or 0
    if off_rt == 0 and on_rt > 0:
        return "HYBRID (toggle honored)"
    if off_rt == 0 and on_rt == 0:
        return "NON-REASONING-ONLY (max_tokens ignored; never thinks)"
    if off_rt > 0 and on_rt > 0:
        return "REASONING-ONLY (disable ignored; always thinks)"
    return "UNEXPECTED"


def main() -> int:
    launch = datetime.now().strftime("%Y%m%dT%H%M%S")
    root = Path("experiment_results") / f"check_models_msc_{launch}"
    print(f"output root: {root}")
    print(f"problem:     {PROBLEM}")
    print(f"question:    {QUESTION_TYPE}")
    print(f"models ({len(MODELS)}):")
    for m in MODELS:
        print(f"  {m}")

    rows: list[tuple[str, str, dict, dict]] = []
    for model in MODELS:
        print("\n" + "=" * 72)
        print(f"MODEL: {model}")
        print("=" * 72)
        safe = model.replace("/", "_").replace(":", "_")
        model_dir = root / safe
        off = run_one(model, {"enabled": False}, "01_reasoning_off", model_dir / "01_reasoning_off")
        on = run_one(model, {"max_tokens": REASONING_MAX_TOKENS}, "02_reasoning_max", model_dir / "02_reasoning_max")
        verdict = classify(off, on)
        rows.append((model, verdict, off, on))

    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    for model, verdict, off, on in rows:
        print(f"\n{model}")
        print(f"  verdict: {verdict}")
        for label, r in (("off", off), ("on ", on)):
            if r["ok"]:
                print(
                    f"  [{label}] extracted={r['extracted']!r} finish={r['finish_reason']!r} "
                    f"ct={r['completion_tokens']} rt={r['reasoning_tokens']} cost=${r['cost']:.6f}"
                )
            else:
                print(f"  [{label}] ERROR: {r['error']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
