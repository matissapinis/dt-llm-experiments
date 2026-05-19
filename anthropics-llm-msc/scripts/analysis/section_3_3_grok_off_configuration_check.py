#!/usr/bin/env python3
"""Sanity check: investigate Grok-off's poor consistency and capability.

Hypotheses to test:
  1. Wrong model name / version (check model_id_openrouter values)
  2. Reasoning not actually off (check reasoning_tokens for off cells)
  3. Truncated responses (check completion_tokens distribution)
  4. Short/shallow responses (check response length distribution)

Compare Grok-off to:
  - Grok-on (same model, different mode)
  - Other off-mode models (DeepSeek off, Qwen off, GPT-5.5 off, etc.)
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

D = Path("experiment_results/main_run_20260516")


def parse_mode(filename: str) -> str:
    if filename.endswith("_on.json"):
        return "on"
    if filename.endswith("_off.json"):
        return "off"
    return "?"


def load_cells():
    cells = []
    for f in sorted(D.glob("*.json")):
        d = json.load(open(f))
        d["mode"] = parse_mode(f.name)
        model = d.get("model_id_openrouter") or ""
        d["model_short"] = model.split("/")[-1]
        d["_filename"] = f.name
        cells.append(d)
    return cells


def main() -> int:
    cells = load_cells()
    print(f"Loaded {len(cells)} cells\n")

    grok_off = [c for c in cells if "grok" in c["model_short"].lower() and c["mode"] == "off"]
    grok_on = [c for c in cells if "grok" in c["model_short"].lower() and c["mode"] == "on"]
    print(f"Grok off cells: {len(grok_off)}")
    print(f"Grok on cells: {len(grok_on)}")

    # =================================================================
    print(f"\n{'=' * 80}")
    print("CHECK 1: model_id_openrouter values for Grok cells")
    print(f"{'=' * 80}")
    grok_off_models = Counter(c.get("model_id_openrouter") for c in grok_off)
    grok_on_models = Counter(c.get("model_id_openrouter") for c in grok_on)
    print(f"\n  Grok OFF model IDs:")
    for mid, n in grok_off_models.most_common():
        print(f"    {mid}: {n}")
    print(f"\n  Grok ON model IDs:")
    for mid, n in grok_on_models.most_common():
        print(f"    {mid}: {n}")

    # =================================================================
    print(f"\n{'=' * 80}")
    print("CHECK 2: reasoning_tokens for Grok off vs Grok on")
    print(f"{'=' * 80}")
    rt_off = [(c.get("usage_statistics") or {}).get("reasoning_tokens", 0) or 0 for c in grok_off]
    rt_on = [(c.get("usage_statistics") or {}).get("reasoning_tokens", 0) or 0 for c in grok_on]
    print(f"\n  Grok OFF reasoning_tokens distribution:")
    print(f"    min={min(rt_off)}, max={max(rt_off)}, median={statistics.median(rt_off):.0f}, mean={statistics.mean(rt_off):.1f}")
    print(f"    cells with rt > 0: {sum(1 for r in rt_off if r > 0)}/{len(rt_off)}")
    print(f"\n  Grok ON reasoning_tokens distribution:")
    print(f"    min={min(rt_on)}, max={max(rt_on)}, median={statistics.median(rt_on):.0f}, mean={statistics.mean(rt_on):.1f}")
    print(f"    cells with rt > 0: {sum(1 for r in rt_on if r > 0)}/{len(rt_on)}")

    # =================================================================
    print(f"\n{'=' * 80}")
    print("CHECK 3: completion_tokens for Grok off vs other off models")
    print(f"{'=' * 80}")
    off_models = {c["model_short"] for c in cells if c["mode"] == "off"}
    print(f"\n  {'model':<32} {'cells':<7} {'ct min':<8} {'ct p25':<8} {'ct median':<10} {'ct p75':<8} {'ct max':<10}")
    print("  " + "-" * 90)
    for model in sorted(off_models):
        ms = [c for c in cells if c["model_short"] == model and c["mode"] == "off"]
        cts = sorted((c.get("usage_statistics") or {}).get("completion_tokens", 0) or 0 for c in ms)
        if not cts:
            continue
        n = len(cts)
        marker = " ← LOW" if statistics.median(cts) < 100 else ""
        print(f"  {model:<32} {n:<7} {cts[0]:<8} {cts[n//4]:<8} {cts[n//2]:<10} {cts[3*n//4]:<8} {cts[-1]:<10}{marker}")

    # =================================================================
    print(f"\n{'=' * 80}")
    print("CHECK 4: response length (chars) for Grok off vs other off models")
    print(f"{'=' * 80}")
    print(f"\n  {'model':<32} {'cells':<7} {'rl min':<8} {'rl p25':<8} {'rl median':<10} {'rl p75':<8} {'rl max':<10}")
    print("  " + "-" * 90)
    for model in sorted(off_models):
        ms = [c for c in cells if c["model_short"] == model and c["mode"] == "off"]
        rls = sorted(len(c.get("response") or "") for c in ms)
        if not rls:
            continue
        n = len(rls)
        marker = " ← LOW" if statistics.median(rls) < 200 else ""
        print(f"  {model:<32} {n:<7} {rls[0]:<8} {rls[n//4]:<8} {rls[n//2]:<10} {rls[3*n//4]:<8} {rls[-1]:<10}{marker}")

    # =================================================================
    print(f"\n{'=' * 80}")
    print("CHECK 5: Sample 5 Grok-off responses (random) to see what they look like")
    print(f"{'=' * 80}")
    import random
    random.seed(7)
    sample = random.sample(grok_off, 5)
    for i, c in enumerate(sample, start=1):
        print(f"\n  --- Sample {i}: {c['_filename'][:80]}")
        ct = (c.get("usage_statistics") or {}).get("completion_tokens", 0) or 0
        rt = (c.get("usage_statistics") or {}).get("reasoning_tokens", 0) or 0
        resp = c.get("response") or ""
        print(f"    completion_tokens={ct}, reasoning_tokens={rt}, response_len={len(resp)}, "
              f"extracted_choice={c.get('extracted_choice')}")
        print(f"    FULL response:")
        print(f"    {resp}")

    # =================================================================
    print(f"\n{'=' * 80}")
    print("CHECK 6: finish_reason distribution for Grok off vs Grok on")
    print(f"{'=' * 80}")
    fr_off = Counter(c.get("finish_reason") for c in grok_off)
    fr_on = Counter(c.get("finish_reason") for c in grok_on)
    print(f"\n  Grok OFF finish_reason:")
    for fr, n in fr_off.most_common():
        print(f"    {fr!r}: {n}")
    print(f"\n  Grok ON finish_reason:")
    for fr, n in fr_on.most_common():
        print(f"    {fr!r}: {n}")

    # =================================================================
    print(f"\n{'=' * 80}")
    print("CHECK 7: Reasoning config in cells (any cell-level reasoning config record?)")
    print(f"{'=' * 80}")
    grok_off_sample = grok_off[0]
    print(f"\n  Sample Grok-off cell keys: {sorted(grok_off_sample.keys())}")
    # Check for any reasoning config field
    for key in grok_off_sample.keys():
        if "reasoning" in key.lower() or "config" in key.lower():
            print(f"    {key} = {grok_off_sample[key]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
