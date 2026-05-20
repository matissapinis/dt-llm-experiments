#!/usr/bin/env python3
"""Spot-check whether models flag the scaled SB-classic scenario as implausible.

Scaled SB-classic asks the model to reason about Beauty being awakened 200
billion vs 200 trillion times — biologically/physically implausible for a
single human Beauty (though more natural in the AI-instance theme or in DD).

Fires 24 cells on `20260516_standard_sb_classic_scaled_12`:
  - 12 cells × ssa_capability (formal-theory question)
  - 12 cells × personal_attitude (model's own opinion question)

Cost ~$0.60, wall time ~5 min.

For each cell:
  - Print extracted choice (A=0.9 SSA-correct, B=0.0089 SIA-correct).
  - Search reasoning for implausibility flag terms.
  - Dump the full response so we can read what models say.

Usage:
  python scripts/spotcheck_scaled_sb_implausibility.py
"""
from __future__ import annotations

import json
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from framework import MODEL_CONFIG, NewcombExperiment  # type: ignore  # noqa: E402

PROBLEM = "20260516_standard_sb_classic_scaled_12"
QUESTION_TYPES = ["ssa_capability", "personal_attitude"]
ACTIVE_MODELS = {
    "anthropic/claude-opus-4.7",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3-flash-preview",
    "openai/gpt-5.5",
    "x-ai/grok-4.3",
    "qwen/qwen3.6-max-preview",
    "deepseek/deepseek-v4-pro",
}

IMPLAUSIBILITY_TERMS = [
    "implausible", "implausibility",
    "impossible", "physically impossible", "biologically impossible",
    "absurd", "absurdity",
    "unrealistic", "unrealistically",
    "hypothetical", "thought experiment",
    "biological", "biologically",
    "lifespan", "lifetime",
    "abstract", "abstraction",
    "fantastical", "fanciful",
    "feasible", "infeasible",
    "200 billion", "200 trillion",
    "memory limits", "physical limits",
    "ill-defined", "ill-posed",
    "abstracted away", "set aside",
    "willing suspension", "stipulation",
    "philosophical fiction",
    "human", "person",  # in case they comment on Beauty being a human
]

PROVIDER_SEMAPHORES = {p: threading.Semaphore(4) for p in ("anthropic", "openai", "google", "x-ai", "qwen", "deepseek")}


def enumerate_cells():
    cells = []
    for model, cfg in MODEL_CONFIG.items():
        if model not in ACTIVE_MODELS:
            continue
        if cfg["reasoning_off"] is not None:
            cells.append((model, "off", cfg["reasoning_off"], cfg["max_tokens"]))
        if cfg["reasoning_on"] is not None:
            cells.append((model, "on", cfg["reasoning_on"], cfg["max_tokens"]))
    return cells


def fire_cell(exp, system_prompt, user_prompt, model, mode, cfg):
    provider = model.split("/")[0]
    sem = PROVIDER_SEMAPHORES[provider]
    messages = [{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}]
    t0 = time.time()
    try:
        with sem:
            response = exp._openrouter_call(
                model, messages,
                reasoning_config=cfg[f"reasoning_{mode}"],
                max_tokens=cfg["max_tokens"],
            )
        elapsed = time.time() - t0
        if not getattr(response, "choices", None):
            return {"error": "no choices", "elapsed": elapsed}
        choice_obj = response.choices[0]
        msg = choice_obj.message
        text = msg.content or ""
        choice = exp.extract_final_answer(text) if text else None
        usage = exp._serialize_usage(response.usage) if response.usage else {}
        return {
            "response": text,
            "reasoning_trace": getattr(msg, "reasoning", None),
            "extracted_choice": choice,
            "finish_reason": choice_obj.finish_reason,
            "elapsed": elapsed,
            "cost": usage.get("cost") or 0,
        }
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}", "elapsed": time.time() - t0}


def find_implausibility_terms(text: str) -> list[str]:
    lower = (text or "").lower()
    hits = []
    for term in IMPLAUSIBILITY_TERMS:
        if term.lower() in lower:
            hits.append(term)
    return list(dict.fromkeys(hits))  # dedupe preserving order


def main() -> int:
    exp = NewcombExperiment(base_output_dir="/tmp/scaled_sb_implausibility_check", temperature=0.8)
    exp.load_problem(PROBLEM)
    problem_dir = exp.find_problem_dir(PROBLEM)
    params = exp.generate_parameters(exp.param_config, exp.problem_structure)
    user_prompt = exp.prompt_templates[PROBLEM].format(**params)

    print(f"problem: {PROBLEM}")
    print(f"params: {params}")
    print(f"\nuser prompt (substituted):\n{user_prompt}\n")

    cells = enumerate_cells()
    all_results = []  # (q_type, model, mode, result)

    for q_type in QUESTION_TYPES:
        system_prompt = (problem_dir / "system_prompts" / f"{q_type}.txt").read_text().strip()
        print(f"\n{'='*80}\nQ_TYPE: {q_type}\n{'='*80}")
        print(f"system prompt:\n{system_prompt}\n")
        with ThreadPoolExecutor(max_workers=16) as executor:
            futures = {}
            for model, mode, _, max_tok in cells:
                cfg = MODEL_CONFIG[model]
                future = executor.submit(fire_cell, exp, system_prompt, user_prompt, model, mode, cfg)
                futures[future] = (model, mode)
            for f in as_completed(futures):
                model, mode = futures[f]
                r = f.result()
                all_results.append((q_type, model, mode, r))
                if "error" in r:
                    print(f"  [{mode:>3}] {model:<40}  ERROR: {r['error']}")
                else:
                    text = r.get("response") or ""
                    trace = r.get("reasoning_trace") or ""
                    hits = find_implausibility_terms(text + " " + trace)
                    print(f"  [{mode:>3}] {model:<40}  choice={r['extracted_choice']}  cost=${r['cost']:.4f}  ({r['elapsed']:.1f}s)  flags={hits[:6]}")

    # Per-(q_type, model, mode) cell deep dump
    print(f"\n{'='*80}\nDEEP DUMP — full response per cell, with flags highlighted\n{'='*80}")
    for q_type, model, mode, r in all_results:
        if "error" in r:
            continue
        text = r.get("response") or ""
        trace = r.get("reasoning_trace") or ""
        hits = find_implausibility_terms(text + " " + trace)
        print(f"\n--- [{q_type}] {model} ({mode}) → choice={r['extracted_choice']} ---")
        if hits:
            print(f"FLAGS: {hits}")
        print(f"response ({len(text)} chars):")
        for line in text.splitlines()[:50]:
            print(f"  {line}")

    # Summary
    print(f"\n{'='*80}\nSUMMARY\n{'='*80}")
    total_cost = sum((r.get("cost") or 0) for _, _, _, r in all_results if "error" not in r)
    print(f"total cells: {len(all_results)}")
    print(f"total cost: ${total_cost:.4f}")

    print(f"\nFlag rate per q_type:")
    for q_type in QUESTION_TYPES:
        sub = [r for q, _, _, r in all_results if q == q_type and "error" not in r]
        n_flagged = sum(1 for r in sub if find_implausibility_terms((r.get("response") or "") + " " + (r.get("reasoning_trace") or "")))
        print(f"  {q_type:<22} {n_flagged}/{len(sub)} cells flagged terms")

    print(f"\nChoice distribution per q_type (A=SSA→0.9, B=SIA→0.0089):")
    for q_type in QUESTION_TYPES:
        sub = [r for q, _, _, r in all_results if q == q_type and "error" not in r]
        choices = [r["extracted_choice"] for r in sub]
        print(f"  {q_type:<22} A={choices.count('A')} B={choices.count('B')} other/null={sum(1 for c in choices if c not in ('A', 'B'))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
