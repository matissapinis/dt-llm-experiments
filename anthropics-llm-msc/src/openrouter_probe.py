#!/usr/bin/env python3
"""OpenRouter probe.

Confirms:
  - Response shape is what we expect (text, finish_reason, usage, reasoning).
  - Reasoning can be toggled OFF and set to a large EXPLICIT budget on a
    hybrid frontier model — using reasoning.max_tokens (not effort) so the
    model cannot adaptively shrink the thinking pass to near-zero tokens.

Mirrors BSc parameters where applicable:
  - temperature = 0.8 (BSc default, framework.py:17).
  - max_tokens set to the model's documented max_completion_tokens to avoid
    the BSc 8192 truncation issue (where 58.3% of reasoning-model outputs truncated).

Persists every run as a flat JSON file under experiment_results/probe_<launch>/,
mirroring the BSc framework's per-run save schema (timestamp, model, full
prompts, response, parameters, problem metadata, extracted_choice,
usage_statistics including reasoning_tokens) plus reasoning_config,
finish_reason, reasoning_trace, and OpenRouter cost details.

Requires:
  - openai>=1.0 installed (OpenAI SDK; OpenRouter is OpenAI-compatible).
  - OPENROUTER_API_KEY env var set.

Usage:
  python src/openrouter_probe.py
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from openai import OpenAI


TEMPERATURE = 0.8

# Frontier Claude successor to BSc 3.5/3.7 Sonnet. Hybrid reasoning model on
# OpenRouter, supports `reasoning.enabled`, `reasoning.effort`, `reasoning.max_tokens`.
MODEL = "anthropic/claude-opus-4.7"

# Max output budget = model's documented max_completion_tokens (queried from
# OpenRouter /api/v1/models). Set high to minimize risk of truncating the
# parseable final answer.
MAX_TOKENS = 128000

# Explicit reasoning budget for the "max effort" run. Using reasoning.max_tokens
# (not reasoning.effort) because effort="high" gets mapped to a permitted
# *cap* that adaptive thinking can choose to leave unused; an explicit large
# budget is the strongest "use lots of reasoning here" signal. Set well below
# MAX_TOKENS so the final answer still has token headroom.
REASONING_MAX_TOKENS = 60000

CONFIG_DIR = Path(__file__).parent.parent / "config" / "problems_msc_draft"
RESULTS_DIR = Path(__file__).parent.parent / "experiment_results"
PROBLEM = "20260510_standard_sb_firewall_12"
SYSTEM_PROMPT_NAME = "personal_attitude"

FINAL_ANSWER_RE = re.compile(r"FINAL\s+ANSWER:\s*([A-Z])", re.IGNORECASE)


def load_problem(problem_dir: Path, system_prompt_name: str) -> dict:
    """Load all prompt components and metadata for one MSc problem config."""
    system = (problem_dir / "system_prompts" / f"{system_prompt_name}.txt").read_text().strip()
    template = (problem_dir / "user_prompt_template.txt").read_text()
    params_meta = json.loads((problem_dir / "user_prompt_parameters.json").read_text())
    substitutions = {k: v["value"] for k, v in params_meta["parameters"].items() if "value" in v}
    user = template.format(**substitutions)
    return {
        "system_prompt": system,
        "user_prompt_template": template,
        "user_prompt": user,
        "parameters": substitutions,
        "params_meta": params_meta,
    }


def extract_final_answer(text: str | None) -> str | None:
    if not text:
        return None
    m = FINAL_ANSWER_RE.search(text)
    return m.group(1).upper() if m else None


def serialize_usage(usage) -> dict:
    """Convert OpenAI CompletionUsage to a plain dict, including OpenRouter
    extensions (cost, cost_details) and reasoning_tokens from
    completion_tokens_details."""
    d: dict = {}
    for attr in ("prompt_tokens", "completion_tokens", "total_tokens"):
        if hasattr(usage, attr):
            d[attr] = getattr(usage, attr)

    details = getattr(usage, "completion_tokens_details", None)
    if details is not None:
        details_d = {}
        for attr in ("reasoning_tokens", "accepted_prediction_tokens", "rejected_prediction_tokens", "audio_tokens", "image_tokens"):
            if hasattr(details, attr):
                val = getattr(details, attr)
                if val is not None:
                    details_d[attr] = val
        if details_d:
            d["completion_tokens_details"] = details_d
            # Promote reasoning_tokens to top level for ergonomic access:
            if "reasoning_tokens" in details_d:
                d["reasoning_tokens"] = details_d["reasoning_tokens"]

    if hasattr(usage, "cost"):
        d["cost"] = usage.cost
    if hasattr(usage, "cost_details"):
        cd = usage.cost_details
        d["cost_details"] = dict(cd) if not isinstance(cd, dict) else cd
    if hasattr(usage, "is_byok"):
        d["is_byok"] = usage.is_byok
    return d


def save_result(result: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    model_safe = result["model"].replace("/", "_").replace(":", "_")
    filename = (
        f"{result['launch_timestamp']}_{result['timestamp']}"
        f"_{result['run_label']}_{model_safe}.json"
    )
    filepath = out_dir / filename
    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
        f.flush()
    return filepath


def run(
    client: OpenAI,
    problem: dict,
    reasoning: dict,
    label: str,
    run_label: str,
    launch_timestamp: str,
    out_dir: Path,
) -> None:
    print("\n" + "=" * 72)
    print(label)
    print("=" * 72)
    print(f"reasoning config: {reasoning}")

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": problem["system_prompt"]},
                {"role": "user", "content": problem["user_prompt"]},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            extra_body={"reasoning": reasoning},
        )
    except Exception as e:
        print(f"\nERROR: {type(e).__name__}: {e}")
        return

    choice = resp.choices[0]
    msg = choice.message
    response_text = msg.content
    reasoning_text = getattr(msg, "reasoning", None)
    finish_reason = choice.finish_reason
    extracted_choice = extract_final_answer(response_text)
    usage_dict = serialize_usage(resp.usage) if resp.usage is not None else {}

    params_meta = problem["params_meta"]
    structure_meta = params_meta.get("structure", {})
    if not isinstance(structure_meta, dict):
        structure_meta = {}

    result = {
        "launch_timestamp": launch_timestamp,
        "timestamp": datetime.now().isoformat(),
        "run_label": run_label,
        "provider": "openrouter",
        "model": MODEL,
        "model_id_openrouter": getattr(resp, "model", None),
        "openrouter_response_id": getattr(resp, "id", None),
        "temperature": TEMPERATURE,
        "max_tokens": MAX_TOKENS,
        "reasoning_config": reasoning,

        "problem": PROBLEM,
        "problem_type": params_meta.get("type"),
        "problem_theme": params_meta.get("theme"),
        "problem_structure": structure_meta.get("type"),
        "row_order": params_meta.get("row_order"),
        "ssa_preference": structure_meta.get("ssa_preference"),
        "sia_preference": structure_meta.get("sia_preference"),
        "system_prompt_name": SYSTEM_PROMPT_NAME,
        "system_prompt": problem["system_prompt"],
        "user_prompt_template": problem["user_prompt_template"],
        "user_prompt": problem["user_prompt"],
        "parameters": problem["parameters"],

        "response": response_text,
        "reasoning_trace": reasoning_text,
        "extracted_choice": extracted_choice,
        "finish_reason": finish_reason,
        "usage_statistics": usage_dict,
    }

    filepath = save_result(result, out_dir)

    # Console summary:
    print(f"\nfinish_reason:    {finish_reason}")
    print(f"extracted_choice: {extracted_choice}")
    print(f"reasoning_tokens: {usage_dict.get('reasoning_tokens', 0)}")
    print(f"completion_tokens: {usage_dict.get('completion_tokens')}")
    print(f"cost:             ${usage_dict.get('cost', 0):.6f}")
    print(f"\nresponse text:\n{'-' * 72}\n{response_text}\n{'-' * 72}")
    if reasoning_text:
        preview = reasoning_text[:500] + ("..." if len(reasoning_text) > 500 else "")
        print(f"\nreasoning trace (first 500 chars):\n{preview}")
    else:
        print("\nreasoning trace: <none returned>")
    print(f"\nsaved: {filepath}")


def main() -> int:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        print(
            "ERROR: OPENROUTER_API_KEY not set.\n"
            "Get a key at https://openrouter.ai/settings/keys, then:\n"
            "  export OPENROUTER_API_KEY=sk-or-v1-..."
        )
        return 1

    problem_dir = CONFIG_DIR / PROBLEM
    if not problem_dir.exists():
        print(f"ERROR: problem dir not found: {problem_dir}")
        return 1

    problem = load_problem(problem_dir, SYSTEM_PROMPT_NAME)
    launch_timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    out_dir = RESULTS_DIR / f"probe_{launch_timestamp}"

    print(f"model:         {MODEL}")
    print(f"temperature:   {TEMPERATURE}")
    print(f"max_tokens:    {MAX_TOKENS}")
    print(f"problem:       {PROBLEM}")
    print(f"system prompt: {SYSTEM_PROMPT_NAME}")
    print(f"output dir:    {out_dir}")

    client = OpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=api_key,
    )

    run(
        client, problem, {"enabled": False},
        "RUN 1: reasoning DISABLED (single forward pass)",
        "01_reasoning_off", launch_timestamp, out_dir,
    )
    run(
        client, problem, {"max_tokens": REASONING_MAX_TOKENS},
        "RUN 2: reasoning MAX (explicit budget)",
        "02_reasoning_max", launch_timestamp, out_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
