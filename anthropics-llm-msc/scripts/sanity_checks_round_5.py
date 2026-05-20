#!/usr/bin/env python3
"""Round 5 sanity checks — hunt for issue types previous rounds didn't cover.

Sections:
  1. finish_reason analysis — were responses truncated by length limit?
  2. Sample diversity — are 9 samples per (template, question, model, mode) genuinely diverse?
  3. Template × model parse_quality heatmap — pockets of systematic failure
  4. Preferred_actions consistency — same problem must give same preferred_actions
  5. DeepSeek-off no_final_answer deep-dive — why 31 cells? Template-specific?
  6. Response length outliers — unusually short / long responses by parse_quality
  7. Filename ↔ metadata consistency (model, row_order)
  8. Cross-model agreement on capability questions — high agreement = unambiguous truth
  9. Decontaminated cells — do their stats now look like non-contaminated cells?
 10. Reasoning_tokens distribution — anomalies

Usage:
  python scripts/sanity_checks_round5.py
"""
from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

D = Path("experiment_results/main_run_20260516")


def load_all() -> list[dict]:
    out = []
    for f in sorted(D.glob("*.json")):
        d = json.load(open(f))
        d["_filename"] = f.name
        out.append(d)
    return out


def section_header(title: str):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def main() -> int:
    cells = load_all()
    n = len(cells)
    print(f"Loaded {n} cells")

    # =================================================================
    section_header("1. finish_reason DISTRIBUTION — any truncation by length?")
    fr_counter: Counter = Counter()
    fr_by_model: dict = defaultdict(Counter)
    for c in cells:
        fr = c.get("finish_reason") or "<none>"
        fr_counter[fr] += 1
        model = (c.get("model_id_openrouter") or "<missing>").split("/")[-1]
        fr_by_model[model][fr] += 1
    print(f"  overall finish_reason distribution:")
    for fr, n_fr in fr_counter.most_common():
        print(f"    {fr:<20} {n_fr}")
    print(f"\n  by model:")
    all_frs = sorted(fr_counter.keys())
    print(f"    {'model':<42}  " + "  ".join(f"{fr[:10]:>10}" for fr in all_frs))
    for m, cnt in sorted(fr_by_model.items()):
        print(f"    {m:<42}  " + "  ".join(f"{cnt.get(fr, 0):>10}" for fr in all_frs))

    # =================================================================
    section_header("2. SAMPLE DIVERSITY — are 9 samples genuinely diverse, or repetitive?")
    # Group cells by (template, question_type, model_id, mode)
    groups: dict = defaultdict(list)
    for c in cells:
        key = (c.get("template_name"), c.get("question_type"),
               c.get("model_id_openrouter"),
               "on" if c["_filename"].endswith("_on.json") else "off")
        groups[key].append(c)
    # For each group of 9, count distinct (extracted_choice) and response-length spread
    choice_uniformity = Counter()  # how many groups have all same choice?
    weirdly_uniform_groups = []  # groups where ALL 9 responses are byte-identical
    for key, gcells in groups.items():
        if len(gcells) != 9:
            continue
        choices = [c.get("extracted_choice") for c in gcells]
        distinct_choices = len(set(choices))
        choice_uniformity[distinct_choices] += 1
        # check for byte-identical responses
        responses = [c.get("response") or "" for c in gcells]
        if len(set(responses)) == 1 and responses[0]:
            weirdly_uniform_groups.append((key, len(responses[0])))
    print(f"  total (template,question,model,mode) groups: {len(groups)}")
    print(f"  choice diversity within 9-sample groups:")
    for k in sorted(choice_uniformity):
        print(f"    {k} distinct choice(s): {choice_uniformity[k]} groups")
    print(f"\n  byte-identical-response groups (suspicious): {len(weirdly_uniform_groups)}")
    for key, rl in weirdly_uniform_groups[:5]:
        tmpl, qt, m, mode = key
        print(f"    {tmpl[:30]} {qt[:20]} {m.split('/')[-1][:30]} {mode} (resp_len={rl})")

    # =================================================================
    section_header("3. TEMPLATE × MODEL parse_quality — find pockets of systematic failure")
    # Find (template, model, mode) combos where >30% of samples have parse_quality != strict_clean
    failure_pockets = []
    for key, gcells in groups.items():
        tmpl, qt, m, mode = key
        if len(gcells) != 9:
            continue
        n_not_clean = sum(1 for c in gcells if c.get("parse_quality") != "strict_clean")
        if n_not_clean >= 5:
            qualities = Counter(c.get("parse_quality") for c in gcells)
            failure_pockets.append((tmpl, qt, m.split("/")[-1], mode, n_not_clean, dict(qualities)))
    # Sort by # not-clean descending
    failure_pockets.sort(key=lambda x: -x[4])
    print(f"  groups with ≥5/9 cells not strict_clean: {len(failure_pockets)}")
    for tmpl, qt, m, mode, nn, qs in failure_pockets[:20]:
        print(f"    {nn}/9  {tmpl[:35]} {qt[:18]} {m[:30]} {mode}  {qs}")

    # =================================================================
    section_header("4. PREFERRED_ACTIONS CONSISTENCY — same (template, q_type, row_order) must have same preferred_actions")
    by_problem = defaultdict(list)
    for c in cells:
        key = (c.get("template_name"), c.get("question_type"), c.get("row_order"))
        by_problem[key].append(c.get("preferred_actions"))
    mismatches = 0
    for k, pas in by_problem.items():
        as_tuples = [json.dumps(p, sort_keys=True) for p in pas]
        if len(set(as_tuples)) > 1:
            mismatches += 1
            print(f"  ✗ {k} has {len(set(as_tuples))} distinct preferred_actions values")
    if mismatches == 0:
        print(f"  ✓ all {len(by_problem)} (template, q_type, row_order) groups have consistent preferred_actions")

    # =================================================================
    section_header("5. DEEPSEEK-OFF no_final_answer DEEP-DIVE")
    ds_off_nfa = [c for c in cells
                  if c["_filename"].endswith("_off.json")
                  and "deepseek" in (c.get("model_id_openrouter") or "").lower()
                  and c.get("parse_quality") == "no_final_answer"]
    print(f"  DeepSeek-off no_final_answer cells: {len(ds_off_nfa)}")
    # Template breakdown
    tmpl_dist = Counter(c.get("template_name") for c in ds_off_nfa)
    qt_dist = Counter(c.get("question_type") for c in ds_off_nfa)
    print(f"  by template: {dict(tmpl_dist)}")
    print(f"  by question_type: {dict(qt_dist)}")
    # Are these decontaminated cells?
    n_decontam = sum(1 for c in ds_off_nfa if c.get("_refire_history"))
    print(f"  of which decontaminated (have _refire_history): {n_decontam}")
    # finish_reason and length distribution
    fr_d = Counter(c.get("finish_reason") for c in ds_off_nfa)
    print(f"  finish_reasons: {dict(fr_d)}")
    lengths = [len(c.get("response") or "") for c in ds_off_nfa]
    if lengths:
        print(f"  response length: min={min(lengths)} max={max(lengths)} median={statistics.median(lengths):.0f}")

    # =================================================================
    section_header("6. RESPONSE LENGTH OUTLIERS by parse_quality")
    len_by_quality: dict = defaultdict(list)
    for c in cells:
        len_by_quality[c.get("parse_quality")].append(len(c.get("response") or ""))
    print(f"  {'quality':<28} {'count':<8} {'min':<8} {'p25':<8} {'median':<10} {'p75':<8} {'max':<8}")
    for q in sorted(len_by_quality):
        ls = sorted(len_by_quality[q])
        if not ls:
            continue
        p25 = ls[len(ls) // 4]
        p50 = ls[len(ls) // 2]
        p75 = ls[3 * len(ls) // 4]
        print(f"  {q:<28} {len(ls):<8} {ls[0]:<8} {p25:<8} {p50:<10} {p75:<8} {ls[-1]:<8}")

    # =================================================================
    section_header("7. FILENAME ↔ METADATA CONSISTENCY")
    row_mismatches, model_mismatches = [], []
    for c in cells:
        fname = c["_filename"]
        # row_order from filename
        rm = re.search(r"_row(12|21)_", fname)
        if rm and c.get("row_order") != rm.group(1):
            row_mismatches.append((fname, rm.group(1), c.get("row_order")))
        # model from filename
        mm = re.search(r"sample\d+_([^_]+)_([^_]+)_(on|off)\.json", fname)
        if mm:
            org_model = mm.group(1) + "_" + mm.group(2)
            actual = (c.get("model_id_openrouter") or "").lower()
            # crude check: model fragment should appear in actual
            org_frag = mm.group(2).lower().replace("-preview", "").replace("-flash", "")[:6]
            if org_frag not in actual.lower().replace("-", "").replace(".", ""):
                # might be false positive due to format diff; tolerate
                pass
    print(f"  row_order mismatches: {len(row_mismatches)}")
    for fn, expected, actual in row_mismatches[:5]:
        print(f"    {fn[:60]}: filename says {expected}, metadata {actual}")
    print(f"  (model-name consistency: skipped — filename format is loose)")

    # =================================================================
    section_header("8. CROSS-MODEL AGREEMENT on capability questions")
    # For each (template, question_type, row_order, sample) of a capability question,
    # take the answers across all 12 (model,mode) cells. Compute agreement rate.
    cap_cells: dict = defaultdict(list)
    for c in cells:
        qt = c.get("question_type") or ""
        if not qt.endswith("_capability"):
            continue
        key = (c.get("template_name"), qt, c.get("row_order"))
        cap_cells[key].append(c.get("extracted_choice"))
    # For each capability question, what's the most common answer and what fraction agree?
    high_disagree = []
    total = 0
    perfect_agree = 0
    for k, choices in cap_cells.items():
        valid = [ch for ch in choices if ch in ("A", "B")]
        if not valid:
            continue
        total += 1
        cnt = Counter(valid)
        most_common, mc_n = cnt.most_common(1)[0]
        agreement = mc_n / len(valid)
        if agreement == 1.0:
            perfect_agree += 1
        elif agreement < 0.7:
            high_disagree.append((k, dict(cnt), agreement))
    print(f"  total capability questions: {total}")
    print(f"  with perfect cross-model agreement: {perfect_agree}  ({100*perfect_agree/total:.1f}%)")
    print(f"  with <70% agreement (high disagreement): {len(high_disagree)}")
    for k, dist, agr in high_disagree[:20]:
        tmpl, qt, row = k
        print(f"    {agr:.0%} agr  {tmpl[:32]} {qt[:18]} row={row}  {dist}")

    # =================================================================
    section_header("9. DECONTAMINATED CELLS — stats vs non-contaminated DeepSeek-off")
    ds_off_decontam = [c for c in cells
                       if c["_filename"].endswith("_off.json")
                       and "deepseek" in (c.get("model_id_openrouter") or "").lower()
                       and c.get("_refire_history")]
    ds_off_clean = [c for c in cells
                    if c["_filename"].endswith("_off.json")
                    and "deepseek" in (c.get("model_id_openrouter") or "").lower()
                    and not c.get("_refire_history")]
    print(f"  decontaminated DeepSeek-off: {len(ds_off_decontam)}")
    print(f"  clean (never re-fired): {len(ds_off_clean)}")

    def stats(cells_list, label):
        rts = [(c.get("usage_statistics") or {}).get("reasoning_tokens", 0) or 0 for c in cells_list]
        cts = [(c.get("usage_statistics") or {}).get("completion_tokens", 0) or 0 for c in cells_list]
        qd = Counter(c.get("parse_quality") for c in cells_list)
        rt_max = max(rts) if rts else 0
        ct_median = statistics.median(cts) if cts else 0
        print(f"  {label}: rt_max={rt_max} ct_median={ct_median:.0f}  parse_quality_dist={dict(qd)}")

    stats(ds_off_decontam, "decontaminated")
    stats(ds_off_clean, "clean         ")

    # =================================================================
    section_header("10. REASONING_TOKENS DISTRIBUTION — any anomalies?")
    rt_by_mode = defaultdict(list)
    for c in cells:
        rt = (c.get("usage_statistics") or {}).get("reasoning_tokens", 0) or 0
        mode = "on" if c["_filename"].endswith("_on.json") else "off"
        m = (c.get("model_id_openrouter") or "").split("/")[-1]
        rt_by_mode[(m, mode)].append(rt)
    print(f"  {'model':<42} {'mode':<5} {'min':<6} {'median':<10} {'p95':<8} {'max':<10} {'#rt>0':<10}")
    for (m, mode), rts in sorted(rt_by_mode.items()):
        rts_sorted = sorted(rts)
        med = statistics.median(rts_sorted)
        p95 = rts_sorted[int(0.95 * len(rts_sorted))]
        n_pos = sum(1 for r in rts_sorted if r > 0)
        print(f"  {m:<42} {mode:<5} {rts_sorted[0]:<6} {med:<10.0f} {p95:<8} {rts_sorted[-1]:<10} {n_pos:<10}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
