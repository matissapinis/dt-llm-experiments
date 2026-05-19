#!/usr/bin/env python3
"""Categorize the 125 null-extracted_choice cells in Main run by failure mode.

For each null cell, capture:
  - model + mode (which model is dropping the ball)
  - finish_reason (stop / length / content_filter / None)
  - completion_tokens (did it burn its token budget?)
  - reasoning_tokens (did it spend everything in reasoning trace?)
  - response shape (empty / short garbled / long no-FINAL-ANSWER)

Then bucket into:
  TECH_TRANSIENT — clearly xAI capacity / network 502/500/etc.; safe to re-fire.
  TECH_TRUNCATION — model hit max_tokens (finish=length); MAYBE re-fire with higher cap, but cap is already very high.
  REASONING_BURN — model burned all tokens on reasoning, no completion budget left.
  OFF_MENU_REFUSAL — model gave a coherent answer that doesn't fit the A/B menu.
  GARBLED — output got corrupted / degenerate tokens.
  OTHER

Usage:
  python scripts/diagnose_null_cells_main_run.py
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

MAIN_RUN_DIR = Path("experiment_results/main_run_20260516")


def classify_cell(cell: dict) -> dict:
    response = cell.get("response") or ""
    trace = cell.get("reasoning_trace") or ""
    finish = cell.get("finish_reason")
    usage = cell.get("usage_statistics") or {}
    completion_tokens = usage.get("completion_tokens", 0) or 0
    reasoning_tokens = usage.get("reasoning_tokens", 0) or 0
    response_len = len(response)
    trace_len = len(trace)
    has_final_answer_pattern = bool(re.search(r"FINAL\s*ANSWER\s*:", response, re.IGNORECASE))

    # Heuristic categorization
    if response_len == 0 and trace_len == 0:
        category = "TECH_TRANSIENT_OR_EMPTY"  # likely xAI 502, no useful output at all
    elif response_len == 0 and trace_len > 0:
        # Reasoning happened but final response is empty
        if reasoning_tokens > 50000:  # heavy reasoning, likely burned budget
            category = "REASONING_BURN"
        else:
            category = "RESPONSE_EMPTY_REASONING_OK"
    elif finish == "length":
        category = "TECH_TRUNCATION_MAX_TOKENS"
    elif "FINAL ANSWER" in response.upper() and not has_final_answer_pattern:
        # Has FINAL ANSWER text but doesn't match the regex strictly
        category = "OFF_MENU_OR_PARSER_MISS"
    elif has_final_answer_pattern:
        # Has FINAL ANSWER: but not parseable as A/B — likely "FINAL ANSWER: Neither" or similar
        category = "OFF_MENU_REFUSAL"
    elif len(response.splitlines()) < 5 and response_len < 200:
        # Short garbled-looking text
        category = "GARBLED_OR_TRUNCATED"
    else:
        # Long response without FINAL ANSWER format — could be off-menu or just non-compliant
        category = "NO_FINAL_ANSWER_FORMAT"
    return {
        "category": category,
        "model": cell.get("model"),
        "mode": cell.get("reasoning_mode"),
        "template": cell.get("template_name"),
        "q_type": cell.get("question_type"),
        "sample": cell.get("run_number"),
        "finish_reason": finish,
        "completion_tokens": completion_tokens,
        "reasoning_tokens": reasoning_tokens,
        "response_len": response_len,
        "trace_len": trace_len,
        "response": response,
        "reasoning_trace": trace[-500:] if trace else "",
    }


def main() -> int:
    cells = []
    for f in sorted(MAIN_RUN_DIR.glob("*.json")):
        j = json.load(open(f))
        if j.get("extracted_choice") is None:
            cells.append(j)
    print(f"loaded {len(cells)} null-choice cells\n")

    classified = [classify_cell(c) for c in cells]

    # Bucket counts
    by_cat = defaultdict(list)
    for c in classified:
        by_cat[c["category"]].append(c)
    print("=" * 80)
    print(f"OVERALL CATEGORIZATION ({len(classified)} cells)")
    print("=" * 80)
    for cat, items in sorted(by_cat.items(), key=lambda x: -len(x[1])):
        print(f"  {cat:<35} {len(items):>4}")
    print()

    # Breakdown by (model, mode) within each category
    print("=" * 80)
    print("BREAKDOWN BY (model, mode) PER CATEGORY")
    print("=" * 80)
    for cat in sorted(by_cat.keys()):
        items = by_cat[cat]
        print(f"\n[{cat}] {len(items)} cells:")
        by_model = defaultdict(int)
        for c in items:
            by_model[(c["model"], c["mode"])] += 1
        for (m, mode), n in sorted(by_model.items(), key=lambda x: -x[1]):
            print(f"  {m:<40} {mode:>3}  {n:>3}")

    # Representative examples for each category
    print("\n" + "=" * 80)
    print("REPRESENTATIVE EXAMPLES (up to 3 per category)")
    print("=" * 80)
    for cat in sorted(by_cat.keys()):
        items = by_cat[cat]
        print(f"\n\n### {cat} — {len(items)} cells total — showing up to 3:")
        for c in items[:3]:
            print(f"\n  --- {c['model']} ({c['mode']}) {c['template'].replace('20260516_standard_', '')} {c['q_type']} sample {c['sample']} ---")
            print(f"  finish_reason: {c['finish_reason']}  completion_tokens: {c['completion_tokens']}  reasoning_tokens: {c['reasoning_tokens']}")
            print(f"  response_len: {c['response_len']}  trace_len: {c['trace_len']}")
            if c["response"]:
                print(f"  response (first 400 chars):")
                print("    " + (c["response"][:400].replace("\n", "\n    ")))
                if len(c["response"]) > 400:
                    print(f"  ... [response total {c['response_len']} chars]")
            if c["reasoning_trace"]:
                print(f"  reasoning_trace tail (last 300 of {c['trace_len']}):")
                print("    " + c["reasoning_trace"][-300:].replace("\n", "\n    "))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
