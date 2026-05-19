#!/usr/bin/env python3
"""RQ3 — Reasoning effect (ON vs OFF) for the 5 hybrid models.

Pre-registered primary hypothesis (one-sided for capability, two-sided for attitudes):
  - Capability: reasoning ON ≥ reasoning OFF in capability accuracy.
  - Attitudes: reasoning shifts attitude distribution (no committed direction).

Data subset: paired ON/OFF within (model, template_name, question_type, sample)
for the 5 hybrid models: Claude Opus, GPT-5.5, Grok 4.3, DeepSeek V4 Pro, Qwen 3.6 Max.
(Gemini 3.1 Pro is reasoning-only; Gemini 3 Flash is non-reasoning leg — neither has on/off pairing.)

Tests:
  - Capability: McNemar's paired test on correctness, per (model, q-type). 10 tests.
  - Attitudes: McNemar's paired test on SSA-aligned-vs-not, per (model, q-type). 10 tests.
  - Total: 20 tests. Bonferroni within RQ3: α = 0.05 / 20 = 0.0025.

Sub-hypothesis: reasoning helps capability more than it shifts attitude.
Quantified by comparing |Δ capability accuracy| vs |Δ SSA-aligned rate| per model.
"""
from __future__ import annotations

import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path

D = Path("experiment_results/main_run_20260516")
HYBRID_MODELS = {
    "claude-4.7-opus-20260416",
    "gpt-5.5-20260423",
    "x-ai/grok-4.3",  # also matches grok-4.3-20260430 (we use short name)
    "grok-4.3-20260430",
    "deepseek-v4-pro-20260423",
    "qwen3.6-max-preview-20260420",
}
N_TESTS = 20  # 5 hybrid models × 4 question types
ALPHA_FAMILY = 0.05
ALPHA_BONF = ALPHA_FAMILY / N_TESTS  # = 0.0025


def parse_mode(filename: str) -> str:
    if filename.endswith("_on.json"):
        return "on"
    if filename.endswith("_off.json"):
        return "off"
    return "?"


def parse_sample(filename: str) -> int:
    m = re.search(r"_sample(\d+)_", filename)
    return int(m.group(1)) if m else -1


def load_dataset() -> list[dict]:
    cells = []
    for f in sorted(D.glob("*.json")):
        d = json.load(open(f))
        d["_filename"] = f.name
        d["mode"] = parse_mode(f.name)
        d["sample_num"] = parse_sample(f.name)
        model = d.get("model_id_openrouter") or ""
        d["model_short"] = model.split("/")[-1]
        cells.append(d)
    return cells


def chi2_2x2(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    n = a + b + c + d
    if n == 0:
        return (0.0, 1.0)
    row1, row2 = a + b, c + d
    col1, col2 = a + c, b + d
    e_a = row1 * col1 / n
    e_b = row1 * col2 / n
    e_c = row2 * col1 / n
    e_d = row2 * col2 / n
    chi2 = 0.0
    for obs, exp in zip([a, b, c, d], [e_a, e_b, e_c, e_d]):
        if exp > 0:
            chi2 += (obs - exp) ** 2 / exp
    p_value = math.erfc(math.sqrt(chi2 / 2))
    return (chi2, p_value)


def mcnemar_one_sided(b: int, c: int) -> tuple[float, float]:
    """One-sided McNemar's test: P(X ≥ b | n=b+c, p=0.5).

    H1: more on-correct/off-wrong pairs (b) than off-correct/on-wrong pairs (c).
    Returns (effect_estimate, p_value).
    """
    n = b + c
    if n == 0:
        return (0.0, 1.0)
    effect = (b - c) / n
    # Exact binomial: P(X >= b | n, 0.5)
    if b == n:
        return (effect, 0.5**n)
    if b == 0:
        return (effect, 1.0)
    # Sum from b to n
    p_val = sum(math.comb(n, i) for i in range(b, n + 1)) * (0.5**n)
    return (effect, p_val)


def mcnemar_two_sided(b: int, c: int) -> tuple[float, float]:
    """Two-sided McNemar's test: chi-square with continuity correction.

    Test statistic: (|b - c| - 1)² / (b + c).
    Returns (effect_estimate = (b-c)/n, p_value).
    """
    n = b + c
    if n == 0:
        return (0.0, 1.0)
    effect = (b - c) / n
    if n < 25:
        # Exact two-sided binomial test
        # p = 2 × min(P(X ≤ min(b,c)), P(X ≥ max(b,c))) given p=0.5
        smaller = min(b, c)
        # Cumulative P(X ≤ smaller | n, 0.5)
        cdf_smaller = sum(math.comb(n, i) for i in range(0, smaller + 1)) * (0.5**n)
        p_val = min(1.0, 2 * cdf_smaller)
        return (effect, p_val)
    # Asymptotic chi-square with continuity correction
    chi2 = (abs(b - c) - 1) ** 2 / n
    p_val = math.erfc(math.sqrt(chi2 / 2))
    return (effect, p_val)


def get_ssa_aligned_letter(preferred_actions: dict, row_order: str) -> str | None:
    """Return the letter (A or B) that maps to the SSA-aligned recommendation."""
    if not preferred_actions:
        return None
    ssa_pref = preferred_actions.get("ssa_preference")
    if not ssa_pref:
        return None
    # row=12 mapping: half/high → A, third/low → B
    # row=21 mapping: half/high → B, third/low → A
    is_A_in_row12 = ssa_pref in ("half", "high")
    if row_order == "12":
        return "A" if is_A_in_row12 else "B"
    else:  # row=21
        return "B" if is_A_in_row12 else "A"


def main() -> int:
    cells = load_dataset()
    print(f"Loaded {len(cells)} cells")

    # Filter to hybrid models
    hybrid_cells = [c for c in cells if c["model_short"] in HYBRID_MODELS]
    print(f"Hybrid-model cells: {len(hybrid_cells)}")

    # Build pair index: key = (model, template_name, question_type, sample_num) → {on: cell, off: cell}
    pairs: dict = defaultdict(dict)
    for c in hybrid_cells:
        key = (c["model_short"], c.get("template_name"), c.get("question_type"), c["sample_num"])
        pairs[key][c["mode"]] = c

    # Count fully-paired keys
    full_pairs = {k: v for k, v in pairs.items() if "on" in v and "off" in v}
    print(f"Fully-paired (model, template, q-type, sample) keys: {len(full_pairs)}")

    # =================================================================
    print(f"\n{'=' * 80}")
    print("SECTION 1: Capability accuracy — paired McNemar's test (one-sided: ON ≥ OFF)")
    print(f"Bonferroni α = {ALPHA_BONF:.4f} within RQ3 (20 tests)")
    print(f"{'=' * 80}")

    # Per (model, q-type) where q-type is capability
    cap_results = []
    by_mq: dict = defaultdict(lambda: {"n": 0, "b": 0, "c": 0, "concordant_correct": 0,
                                         "concordant_wrong": 0, "both_null": 0, "either_null": 0})
    for key, v in full_pairs.items():
        model, tmpl, qt, sample = key
        if not (qt or "").endswith("_capability"):
            continue
        on_c = v["on"].get("correct_capability_answer")
        off_c = v["off"].get("correct_capability_answer")
        mq_key = (model, qt)
        by_mq[mq_key]["n"] += 1
        if on_c is None or off_c is None:
            by_mq[mq_key]["either_null"] += 1
            if on_c is None and off_c is None:
                by_mq[mq_key]["both_null"] += 1
            continue
        # b = on correct, off wrong; c = on wrong, off correct
        if on_c and not off_c:
            by_mq[mq_key]["b"] += 1
        elif not on_c and off_c:
            by_mq[mq_key]["c"] += 1
        elif on_c and off_c:
            by_mq[mq_key]["concordant_correct"] += 1
        else:
            by_mq[mq_key]["concordant_wrong"] += 1

    print(f"\n  {'model':<32} {'q-type':<20} {'n_pairs':<8} {'b (on✓ off✗)':<14} "
          f"{'c (on✗ off✓)':<14} {'effect':<8} {'p (1-sided)':<14} {'Bonf-sig':<8}")
    print("  " + "-" * 130)
    for (model, qt), v in sorted(by_mq.items()):
        b, c = v["b"], v["c"]
        effect, p = mcnemar_one_sided(b, c)
        sig = "**" if p < ALPHA_BONF else ("*" if p < 0.05 else "")
        cap_results.append({"model": model, "q_type": qt, "b": b, "c": c, "effect": effect, "p": p, "sig": p < ALPHA_BONF})
        print(f"  {model:<32} {qt:<20} {v['n']:<8} {b:<14} {c:<14} "
              f"{effect:+.3f}   {p:<14.4g} {sig:<8}")

    # =================================================================
    print(f"\n\n{'=' * 80}")
    print("SECTION 2: Attitude shift — paired McNemar's test (two-sided)")
    print(f"Comparison: SSA-aligned letter choice on vs SSA-aligned letter choice off")
    print(f"{'=' * 80}")

    att_by_mq: dict = defaultdict(lambda: {"n": 0, "b": 0, "c": 0, "both_ssa": 0,
                                            "both_not_ssa": 0, "either_null": 0})
    for key, v in full_pairs.items():
        model, tmpl, qt, sample = key
        if qt not in ("normative_attitude", "personal_attitude"):
            continue
        on_choice = v["on"].get("extracted_choice")
        off_choice = v["off"].get("extracted_choice")
        if on_choice not in ("A", "B") or off_choice not in ("A", "B"):
            att_by_mq[(model, qt)]["either_null"] += 1
            continue
        # Determine SSA-aligned letter for each cell
        on_ssa_letter = get_ssa_aligned_letter(v["on"].get("preferred_actions"),
                                                 v["on"].get("row_order"))
        off_ssa_letter = get_ssa_aligned_letter(v["off"].get("preferred_actions"),
                                                  v["off"].get("row_order"))
        on_is_ssa = (on_choice == on_ssa_letter)
        off_is_ssa = (off_choice == off_ssa_letter)
        att_by_mq[(model, qt)]["n"] += 1
        if on_is_ssa and not off_is_ssa:
            att_by_mq[(model, qt)]["b"] += 1  # on shifts to SSA
        elif not on_is_ssa and off_is_ssa:
            att_by_mq[(model, qt)]["c"] += 1  # on shifts away from SSA
        elif on_is_ssa and off_is_ssa:
            att_by_mq[(model, qt)]["both_ssa"] += 1
        else:
            att_by_mq[(model, qt)]["both_not_ssa"] += 1

    print(f"\n  {'model':<32} {'q-type':<22} {'n_pairs':<8} {'b (on→SSA)':<13} "
          f"{'c (on→¬SSA)':<13} {'effect':<8} {'p (2-sided)':<14} {'Bonf-sig':<8}")
    print("  " + "-" * 130)
    att_results = []
    for (model, qt), v in sorted(att_by_mq.items()):
        b, c = v["b"], v["c"]
        effect, p = mcnemar_two_sided(b, c)
        sig = "**" if p < ALPHA_BONF else ("*" if p < 0.05 else "")
        att_results.append({"model": model, "q_type": qt, "b": b, "c": c, "effect": effect, "p": p, "sig": p < ALPHA_BONF})
        print(f"  {model:<32} {qt:<22} {v['n']:<8} {b:<13} {c:<13} "
              f"{effect:+.3f}   {p:<14.4g} {sig:<8}")

    # =================================================================
    print(f"\n\n{'=' * 80}")
    print("SECTION 3: Sub-hypothesis — reasoning helps capability more than shifts attitude")
    print(f"{'=' * 80}")
    print(f"\n  Comparing |Δ capability accuracy| vs |Δ SSA-aligned attitude rate| per model.")

    # Aggregate per model: mean |capability effect| vs mean |attitude effect|
    per_model_cap: dict = defaultdict(list)
    per_model_att: dict = defaultdict(list)
    for r in cap_results:
        per_model_cap[r["model"]].append(abs(r["effect"]))
    for r in att_results:
        per_model_att[r["model"]].append(abs(r["effect"]))

    print(f"\n  {'model':<32} {'|cap Δ| avg':<14} {'|att Δ| avg':<14} {'cap > att?':<10}")
    print("  " + "-" * 80)
    n_cap_larger = 0
    for model in sorted(set(list(per_model_cap.keys()) + list(per_model_att.keys()))):
        cap_avg = sum(per_model_cap[model]) / len(per_model_cap[model]) if per_model_cap[model] else 0
        att_avg = sum(per_model_att[model]) / len(per_model_att[model]) if per_model_att[model] else 0
        flag = "YES" if cap_avg > att_avg else "no"
        if cap_avg > att_avg:
            n_cap_larger += 1
        print(f"  {model:<32} {cap_avg:<14.4f} {att_avg:<14.4f} {flag:<10}")
    print(f"\n  Models where |cap Δ| > |att Δ|: {n_cap_larger}/5")

    # =================================================================
    print(f"\n\n{'=' * 80}")
    print("SECTION 3b: SUB-HYPOTHESIS FORMAL TEST")
    print("Does reasoning produce more flips (discordant pairs) on capability than on attitude?")
    print(f"{'=' * 80}")
    print(f"\n  Discordance rate = (b + c) / n_pairs = fraction of pairs where on ≠ off response")
    print(f"  Capability: 'response differs' = correctness flipped")
    print(f"  Attitude:   'response differs' = SSA-aligned-ness flipped")
    print(f"\n  Per-model 2x2 chi-square test:")
    print(f"  {'model':<32} {'cap disc':<14} {'att disc':<14} {'Δ (cap-att)':<13} "
          f"{'χ²':<7} {'p':<12} {'sig':<5}")
    print("  " + "-" * 110)

    # Aggregate per model
    per_model_pool: dict = defaultdict(lambda: {"cap_disc": 0, "cap_n": 0,
                                                  "att_disc": 0, "att_n": 0})
    for r in cap_results:
        per_model_pool[r["model"]]["cap_disc"] += (r["b"] + r["c"])
        per_model_pool[r["model"]]["cap_n"] += (r["b"] + r["c"])  # n_pairs of contributing pairs
    # need actual n_pairs from by_mq counts
    for (model, qt), v in by_mq.items():
        # n is total pairs that had both correctness defined
        n_contrib = v["b"] + v["c"] + v["concordant_correct"] + v["concordant_wrong"]
        per_model_pool[model]["cap_n"] = max(per_model_pool[model]["cap_n"], 0)
        per_model_pool[model]["cap_n"] += n_contrib
    # Wait I need to redo this — let me reset and recompute cleanly
    per_model_pool = defaultdict(lambda: {"cap_disc": 0, "cap_n": 0,
                                            "att_disc": 0, "att_n": 0})
    for (model, qt), v in by_mq.items():
        n_contrib = v["b"] + v["c"] + v["concordant_correct"] + v["concordant_wrong"]
        per_model_pool[model]["cap_disc"] += (v["b"] + v["c"])
        per_model_pool[model]["cap_n"] += n_contrib
    for (model, qt), v in att_by_mq.items():
        n_contrib = v["b"] + v["c"] + v["both_ssa"] + v["both_not_ssa"]
        per_model_pool[model]["att_disc"] += (v["b"] + v["c"])
        per_model_pool[model]["att_n"] += n_contrib

    pop_cap_disc = pop_cap_n = pop_att_disc = pop_att_n = 0
    for model in sorted(per_model_pool):
        v = per_model_pool[model]
        cap_rate = v["cap_disc"] / v["cap_n"] if v["cap_n"] > 0 else 0
        att_rate = v["att_disc"] / v["att_n"] if v["att_n"] > 0 else 0
        cap_conc = v["cap_n"] - v["cap_disc"]
        att_conc = v["att_n"] - v["att_disc"]
        chi2, p = chi2_2x2(v["cap_disc"], cap_conc, v["att_disc"], att_conc)
        sig = "**" if p < 0.01 else ("*" if p < 0.05 else "")
        print(f"  {model:<32} {v['cap_disc']}/{v['cap_n']} ({cap_rate:.3f})  "
              f"{v['att_disc']}/{v['att_n']} ({att_rate:.3f})  "
              f"{cap_rate - att_rate:+.3f}        {chi2:<7.2f} {p:<12.4g} {sig:<5}")
        pop_cap_disc += v["cap_disc"]
        pop_cap_n += v["cap_n"]
        pop_att_disc += v["att_disc"]
        pop_att_n += v["att_n"]

    # Population-level pooled
    print(f"\n  Population-level (pooled across all 5 hybrid models):")
    pop_cap_rate = pop_cap_disc / pop_cap_n if pop_cap_n > 0 else 0
    pop_att_rate = pop_att_disc / pop_att_n if pop_att_n > 0 else 0
    pop_cap_conc = pop_cap_n - pop_cap_disc
    pop_att_conc = pop_att_n - pop_att_disc
    chi2, p = chi2_2x2(pop_cap_disc, pop_cap_conc, pop_att_disc, pop_att_conc)
    print(f"    Capability discordance: {pop_cap_disc}/{pop_cap_n} = {pop_cap_rate:.4f}")
    print(f"    Attitude   discordance: {pop_att_disc}/{pop_att_n} = {pop_att_rate:.4f}")
    print(f"    Δ (cap − att): {pop_cap_rate - pop_att_rate:+.4f}")
    print(f"    χ² = {chi2:.2f}, p = {p:.4g}")
    if pop_cap_rate > pop_att_rate and p < 0.01:
        print(f"    → Sub-hypothesis supported at the population level (p < 0.01)")
    elif pop_cap_rate > pop_att_rate and p < 0.05:
        print(f"    → Sub-hypothesis nominally supported at the population level (p < 0.05)")
    else:
        print(f"    → Sub-hypothesis NOT supported at the population level")

    # =================================================================
    print(f"\n\n{'=' * 80}")
    print("SECTION 4: Summary headline numbers")
    print(f"{'=' * 80}")
    cap_sig = sum(1 for r in cap_results if r["sig"])
    att_sig = sum(1 for r in att_results if r["sig"])
    cap_nominal = sum(1 for r in cap_results if r["p"] < 0.05)
    att_nominal = sum(1 for r in att_results if r["p"] < 0.05)
    print(f"  Capability tests: {cap_sig}/{len(cap_results)} Bonferroni-sig, "
          f"{cap_nominal}/{len(cap_results)} nominal-sig (p<0.05)")
    print(f"  Attitude tests:   {att_sig}/{len(att_results)} Bonferroni-sig, "
          f"{att_nominal}/{len(att_results)} nominal-sig (p<0.05)")

    cap_pos = sum(1 for r in cap_results if r["effect"] > 0)
    cap_neg = sum(1 for r in cap_results if r["effect"] < 0)
    print(f"\n  Capability direction: {cap_pos} model-q-types where ON helps; "
          f"{cap_neg} where OFF helps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
