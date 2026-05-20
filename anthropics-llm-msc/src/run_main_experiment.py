#!/usr/bin/env python3
"""Main run: bundled expansion of MSc anthropic-reasoning experiments.

Cartesian product:
  4 problem classes (sb, inc, dd, padd)
  × 2 themes (canonical: classic/civilization; AI-instance)
  × 2 parameterizations (canonical small-N; scaled 200B/200T)
  × 2 row orders (12, 21)
  × 4 question types
  × N samples (default 9)
  × 12 active cells
  = 32 problem dirs × 4 q-types × N × 12 = 13,824 calls at N=9

Three differences from Main run:
  1. V3-A disambiguated SSA capability prompts in all 32 dirs.
  2. AI-instance theme alongside classic/civilization theme.
  3. N=9 samples per scenario.

Execution: parallelized with per-provider semaphores + flat ThreadPoolExecutor.
  - All 13,824 cells submitted upfront to a single executor (max_workers=32).
  - Each cell's API call is wrapped in a per-provider Semaphore (4 concurrent
    per provider) to stay within OR / provider rate limits.
  - Cells fire across scenarios in parallel; the slowest cell of one scenario
    no longer blocks the fast cells of subsequent scenarios.
  - Per-cell behavior unchanged from Main run: same retry-with-backoff, same
    1200s wallclock timeout, same response-malformed guard, same skip-if-exists
    per-cell JSON.

Smoke mode (env var MAIN_RUN_SMOKE=1): reduces scope to 1 problem × 1 row
order × 1 q-type × 1 sample × 12 cells = 12 cells. Use to validate the
parallelization on a small scope before firing the full bundled run.

Background-friendly invocation:
  LAUNCH=$(date +%Y%m%dT%H%M%S)
  cd <repo>
  MAIN_RUN_OUT_DIR="experiment_results/msc_main_run_$LAUNCH" \\
    caffeinate -dis venv/bin/python -u src/run_msc_main_run.py 2>&1 \\
    | tee "experiment_results/msc_main_run_$LAUNCH.log"

Usage:
  python -u src/run_msc_main_run.py
"""
from __future__ import annotations

import concurrent.futures
import json
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).parent))
from framework import MODEL_CONFIG, NewcombExperiment  # type: ignore  # noqa: E402


# 32 problem dirs: 4 problem classes × 2 themes × 2 parameterizations × 2 row orders.
PROBLEM_BASES: List[Tuple[str, str, str]] = [
    ("sb",   "classic",       "canonical"),
    ("sb",   "classic",       "scaled"),
    ("sb",   "aiinstance",    "canonical"),
    ("sb",   "aiinstance",    "scaled"),
    ("inc",  "classic",       "canonical"),
    ("inc",  "classic",       "scaled"),
    ("inc",  "aiinstance",    "canonical"),
    ("inc",  "aiinstance",    "scaled"),
    ("dd",   "civilization",  "canonical"),
    ("dd",   "civilization",  "scaled"),
    ("dd",   "aiinstance",    "canonical"),
    ("dd",   "aiinstance",    "scaled"),
    ("padd", "civilization",  "canonical"),
    ("padd", "civilization",  "scaled"),
    ("padd", "aiinstance",    "canonical"),
    ("padd", "aiinstance",    "scaled"),
]
ROW_ORDERS: List[str] = ["12", "21"]
QUESTION_TYPES: List[str] = ["ssa_capability", "sia_capability", "normative_attitude", "personal_attitude"]
N_SAMPLES: int = 9
TEMPERATURE: float = 0.8

ACTIVE_MODELS: set[str] = {
    "anthropic/claude-opus-4.7",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3-flash-preview",
    "openai/gpt-5.5",
    "x-ai/grok-4.3",
    "qwen/qwen3.6-max-preview",
    "deepseek/deepseek-v4-pro",
}

# Per-provider concurrent-call cap (semaphores limit simultaneous API calls
# per provider to avoid tripping rate limits). 4 per provider × 7 providers =
# 28 effective parallel slots in steady state. Conservative; OR typically
# allows higher per-provider concurrency.
PROVIDER_CONCURRENCY: Dict[str, int] = {
    "anthropic":   4,
    "openai":      4,
    "google":      4,
    "x-ai":        4,
    "qwen":        4,
    "deepseek":    4,
    "moonshotai":  4,  # not in active set but defined in case it returns
    "z-ai":        4,
}
DEFAULT_PROVIDER_CONCURRENCY = 4
_provider_semaphores: Dict[str, threading.Semaphore] = {
    p: threading.Semaphore(c) for p, c in PROVIDER_CONCURRENCY.items()
}
_semaphore_lock = threading.Lock()

MAX_WORKERS: int = 32  # global executor pool; per-provider semaphores throttle below this

# Smoke mode (env var) reduces the scope to 12 cells for validating the
# parallelized driver before firing the full bundled run.
SMOKE_MODE: bool = os.environ.get("MAIN_RUN_SMOKE") == "1"


def get_provider_semaphore(provider: str) -> threading.Semaphore:
    """Return (lazily-created) semaphore for an unrecognized provider."""
    sem = _provider_semaphores.get(provider)
    if sem is not None:
        return sem
    with _semaphore_lock:
        sem = _provider_semaphores.get(provider)
        if sem is None:
            sem = threading.Semaphore(DEFAULT_PROVIDER_CONCURRENCY)
            _provider_semaphores[provider] = sem
        return sem


def make_problem_name(problem_class: str, theme: str, param_label: str, row: str) -> str:
    """Map (problem_class, theme, param_label, row) → 20260516 dir name."""
    if param_label == "canonical":
        return f"20260516_standard_{problem_class}_{theme}_{row}"
    elif param_label == "scaled":
        return f"20260516_standard_{problem_class}_{theme}_scaled_{row}"
    else:
        raise ValueError(f"unknown param_label: {param_label}")


def enumerate_cells() -> List[Tuple[str, str, Dict[str, Any], int]]:
    cells: List[Tuple[str, str, Dict[str, Any], int]] = []
    for model, cfg in MODEL_CONFIG.items():
        if model not in ACTIVE_MODELS:
            continue
        if cfg["reasoning_off"] is not None:
            cells.append((model, "off", cfg["reasoning_off"], cfg["max_tokens"]))
        if cfg["reasoning_on"] is not None:
            cells.append((model, "on", cfg["reasoning_on"], cfg["max_tokens"]))
    return cells


def run_cell(
    exp: NewcombExperiment,
    cell: Tuple[str, str, Dict[str, Any], int],
    messages: List[Dict[str, str]],
    metadata: Dict[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    model, mode, reasoning_cfg, max_tok = cell
    safe = model.replace("/", "_").replace(":", "_")
    problem_tag = metadata["template_name"]
    row = metadata["row_order"]
    q = metadata["question_type"]
    sample = metadata["run_number"]
    filename = f"{problem_tag}_row{row}_{q}_sample{sample}_{safe}_{mode}.json"
    filepath = out_dir / filename

    if filepath.exists():
        return {"status": "skipped", "model": model, "mode": mode, "filepath": str(filepath)}

    provider = model.split("/")[0]
    sem = get_provider_semaphore(provider)

    t0 = time.time()
    try:
        with sem:
            response = exp._openrouter_call(
                model, messages,
                reasoning_config=reasoning_cfg,
                max_tokens=max_tok,
            )
        elapsed = time.time() - t0
        if not getattr(response, "choices", None):
            raise ValueError(f"malformed response (no choices): {response!r}")
        choice_obj = response.choices[0]
        msg = choice_obj.message
        response_text = msg.content
        reasoning_text = getattr(msg, "reasoning", None)
        finish_reason = choice_obj.finish_reason
        extracted_choice = exp.extract_final_answer(response_text) if response_text else None
        usage_dict = exp._serialize_usage(response.usage) if response.usage is not None else {}
    except Exception as e:
        return {
            "status": "error", "model": model, "mode": mode,
            "error": f"{type(e).__name__}: {e}",
            "elapsed_seconds": time.time() - t0,
        }

    result: Dict[str, Any] = {
        **metadata,
        "model": model,
        "model_id_openrouter": getattr(response, "model", None),
        "openrouter_response_id": getattr(response, "id", None),
        "reasoning_mode": mode,
        "reasoning_config": reasoning_cfg,
        "max_tokens": max_tok,
        "response": response_text,
        "reasoning_trace": reasoning_text,
        "finish_reason": finish_reason,
        "extracted_choice": extracted_choice,
        "usage_statistics": usage_dict,
        "elapsed_seconds": elapsed,
    }

    if extracted_choice:
        # Pass row_order explicitly to avoid thread-unsafe `self.row_order` reads
        # under parallelization (other threads may have mutated it via load_problem).
        alignment = exp.determine_alignment(
            extracted_choice, metadata["preferred_actions"], row_order=metadata["row_order"],
        )
        result.update(alignment)
        correctness = exp.check_correctness(
            extracted_choice, metadata["question_type"], metadata["preferred_actions"],
            row_order=metadata["row_order"],
        )
        if correctness is not None:
            result["correct_capability_answer"] = correctness

    with open(filepath, "w") as f:
        json.dump(result, f, indent=2)
        f.flush()

    return {"status": "ok", **result, "filepath": str(filepath)}


def main() -> int:
    # Apply smoke mode to local copies if env var is set
    problem_bases = PROBLEM_BASES
    row_orders = ROW_ORDERS
    question_types = QUESTION_TYPES
    n_samples = N_SAMPLES
    if SMOKE_MODE:
        problem_bases = [PROBLEM_BASES[0]]   # one problem (sb classic canonical)
        row_orders = ["12"]
        question_types = ["ssa_capability"]
        n_samples = 1
        print("**** SMOKE MODE: scope reduced to 1 problem × 1 row × 1 q-type × 1 sample × 12 cells ****\n")

    override = os.environ.get("MAIN_RUN_OUT_DIR")
    if override:
        out_dir = Path(override)
        launch = out_dir.name.removeprefix("msc_main_run_") or datetime.now().strftime("%Y%m%dT%H%M%S")
    else:
        launch = datetime.now().strftime("%Y%m%dT%H%M%S")
        out_dir = Path("experiment_results") / f"msc_main_run_{launch}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"output:                   {out_dir}")
    print(f"problem bases:            {len(problem_bases)} (4 × 2 themes × 2 parameterizations)")
    print(f"row_orders:               {row_orders}")
    print(f"question_types:           {question_types}")
    print(f"samples:                  {n_samples}")
    print(f"max_workers:              {MAX_WORKERS}")
    print(f"per-provider concurrency: {PROVIDER_CONCURRENCY}")

    cells = enumerate_cells()
    print(f"cells (filtered):         {len(cells)}")

    total_scenarios = len(problem_bases) * len(row_orders) * len(question_types) * n_samples
    total_calls_planned = total_scenarios * len(cells)
    print(f"\nscenarios:                {total_scenarios}")
    print(f"calls planned:            {total_calls_planned}")

    # === Phase 1: pre-compute all scenario setup serially in main thread ===
    print(f"\n[phase 1] precomputing {total_scenarios} scenarios ...")
    setup_start = time.time()
    exp = NewcombExperiment(base_output_dir=str(out_dir), temperature=TEMPERATURE)
    all_scenarios: List[Tuple[List[Dict[str, str]], Dict[str, Any]]] = []
    for problem_class, theme, param_label in problem_bases:
        for row in row_orders:
            problem = make_problem_name(problem_class, theme, param_label, row)
            exp.load_problem(problem)
            problem_dir = exp.find_problem_dir(problem)
            system_prompts = {
                q: (problem_dir / "system_prompts" / f"{q}.txt").read_text().strip()
                for q in question_types
            }
            for sample_idx in range(n_samples):
                params = exp.generate_parameters(exp.param_config, exp.problem_structure)
                expected_utilities, preferred_actions = exp.compute_problem_groundtruth(params)
                user_prompt = exp.prompt_templates[problem].format(**params)
                for q_type in question_types:
                    messages = [
                        {"role": "system", "content": system_prompts[q_type]},
                        {"role": "user", "content": user_prompt},
                    ]
                    metadata = {
                        "launch_timestamp": launch,
                        "template_name": problem,
                        "question_type": q_type,
                        "run_number": sample_idx + 1,
                        "provider": "openrouter",
                        "temperature": TEMPERATURE,
                        "system_prompt": system_prompts[q_type],
                        "user_prompt": user_prompt,
                        "parameters": params,
                        "problem_type": exp.problem_type,
                        "problem_theme": exp.problem_theme,
                        "problem_structure": exp.problem_structure,
                        "problem_valence": exp.problem_valence,
                        "row_order": exp.row_order,
                        "expected_utilities": expected_utilities,
                        "preferred_actions": preferred_actions,
                    }
                    all_scenarios.append((messages, metadata))
    setup_elapsed = time.time() - setup_start
    print(f"[phase 1] precomputed {len(all_scenarios)} scenarios in {setup_elapsed:.1f}s")

    # === Phase 2: submit all cells to a single executor ===
    print(f"\n[phase 2] submitting {total_calls_planned} cells to executor (max_workers={MAX_WORKERS}) ...")
    summary: List[Dict[str, Any]] = []
    t_start = time.time()

    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        all_futures: List[concurrent.futures.Future] = []
        for messages, metadata in all_scenarios:
            for cell in cells:
                future = executor.submit(run_cell, exp, cell, messages, metadata, out_dir)
                all_futures.append(future)
        print(f"[phase 2] all {len(all_futures)} futures submitted; processing completions ...\n")

        # === Phase 3: process completions ===
        n_done = 0
        for f in concurrent.futures.as_completed(all_futures):
            r = f.result()
            summary.append(r)
            n_done += 1
            if r["status"] == "ok":
                u = r.get("usage_statistics", {}) or {}
                cost = u.get("cost") or 0
                rt = u.get("reasoning_tokens", 0)
                tmpl = r.get("template_name", "")
                tmpl_short = tmpl.replace("20260516_standard_", "")
                print(
                    f"  [{n_done:>5}/{total_calls_planned}] [{r['reasoning_mode']:>3}] "
                    f"{r['model']:<40} {tmpl_short:<48} "
                    f"q={r.get('question_type', '')[:18]:<18} smp={r.get('run_number', ''):>2} "
                    f"choice={str(r['extracted_choice']):<5} cost=${cost:.5f} {r['elapsed_seconds']:.1f}s"
                )
            elif r["status"] == "skipped":
                pass  # quiet
            else:
                print(f"  [{n_done:>5}/{total_calls_planned}] ERROR: {r['model']} {r['mode']}: {r['error']}")
            # Periodic progress summary
            if n_done % 100 == 0 or n_done == total_calls_planned:
                running_ok = sum(1 for x in summary if x["status"] == "ok")
                running_err = sum(1 for x in summary if x["status"] == "error")
                running_skip = sum(1 for x in summary if x["status"] == "skipped")
                running_cost = sum((x.get("usage_statistics") or {}).get("cost") or 0 for x in summary if x["status"] == "ok")
                elapsed_min = (time.time() - t_start) / 60
                print(
                    f"  >>> progress: ok={running_ok} err={running_err} skip={running_skip} "
                    f"cost=${running_cost:.4f} elapsed={elapsed_min:.1f}min"
                )

    t_end = time.time()
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    n_ok = sum(1 for r in summary if r["status"] == "ok")
    n_err = sum(1 for r in summary if r["status"] == "error")
    n_skip = sum(1 for r in summary if r["status"] == "skipped")
    total_cost = sum((r.get("usage_statistics") or {}).get("cost") or 0 for r in summary if r["status"] == "ok")
    print(f"calls ok:           {n_ok}")
    print(f"calls error:        {n_err}")
    print(f"calls skipped:      {n_skip}")
    print(f"total cost:         ${total_cost:.4f}")
    print(f"total wall time:    {(t_end - t_start) / 60:.1f}min")
    if n_err:
        print("\nERRORS:")
        for r in summary:
            if r["status"] == "error":
                print(f"  {r['model']:<40} {r['mode']:>3}  {r['error']}")
    print(f"\noutputs at: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
