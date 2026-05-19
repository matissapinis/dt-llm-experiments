#!/usr/bin/env python3
"""RQ10 — Within-cell consistency across the 9 samples per cell.

A "cell" = one (model, mode, problem, theme, row_order, parameterization, q_type)
combination, with 9 sample responses. The within-cell entropy of the 9-sample
choice distribution measures self-consistency:
  - Entropy = 0: all 9 samples picked the same letter (perfectly self-consistent)
  - Entropy = 1: 4-5 split between A and B (maximally inconsistent for binary)

Pre-registered hypothesis (exploratory):
  Within-cell consistency varies across models, q-types, and reasoning modes.

Analyses:
  1. Per-(model, mode) mean within-cell entropy and consistency rankings.
  2. Per-q-type entropy (do attitudes have higher entropy than capability?).
  3. Reasoning effect: does ON mode increase or decrease within-cell consistency?
     (paired Wilcoxon-style test on (model, problem, theme, row, param, q_type)
     within hybrid models)
  4. Distribution of consistency: how many cells unanimous, how many split?
"""
from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from pathlib import Path

D = Path("experiment_results/main_run_20260516")


def parse_mode(filename: str) -> str:
    if filename.endswith("_on.json"):
        return "on"
    if filename.endswith("_off.json"):
        return "off"
    return "?"


def parse_sample(filename: str) -> int:
    m = re.search(r"_sample(\d+)_", filename)
    return int(m.group(1)) if m else -1


def parse_problem_class(template_name: str) -> str:
    m = re.match(r"\d+_standard_(sb|inc|dd|padd)_", template_name or "")
    return m.group(1) if m else "?"


def parse_cluster(pc: str) -> str:
    return "SB-type" if pc in ("sb", "inc") else ("DD-type" if pc in ("dd", "padd") else "?")


def binary_entropy(n_a: int, n_b: int) -> float:
    """Shannon entropy in bits for binary distribution (A, B)."""
    n = n_a + n_b
    if n == 0:
        return 0.0
    p = n_a / n
    if p == 0 or p == 1:
        return 0.0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)


def load_cells():
    cells = []
    for f in sorted(D.glob("*.json")):
        d = json.load(open(f))
        d["mode"] = parse_mode(f.name)
        d["sample_num"] = parse_sample(f.name)
        model = d.get("model_id_openrouter") or ""
        d["model_short"] = model.split("/")[-1]
        d["problem_class"] = parse_problem_class(d.get("template_name", ""))
        d["cluster"] = parse_cluster(d["problem_class"])
        cells.append(d)
    return cells


def main() -> int:
    cells = load_cells()
    print(f"Loaded {len(cells)} cells")

    # Group by cell-key: (model, mode, template_name, q_type)
    cell_groups: dict = defaultdict(list)
    for c in cells:
        key = (c["model_short"], c["mode"], c.get("template_name"), c.get("question_type"))
        cell_groups[key].append(c)

    # Compute entropy per cell-group (excluding null choices)
    cell_summaries = []
    for key, samples in cell_groups.items():
        n_a = sum(1 for s in samples if s.get("extracted_choice") == "A")
        n_b = sum(1 for s in samples if s.get("extracted_choice") == "B")
        n_null = sum(1 for s in samples if s.get("extracted_choice") not in ("A", "B"))
        entropy = binary_entropy(n_a, n_b)
        cell_summaries.append({
            "model": key[0], "mode": key[1], "template": key[2], "q_type": key[3],
            "n_a": n_a, "n_b": n_b, "n_null": n_null,
            "entropy": entropy,
            "cluster": parse_cluster(parse_problem_class(key[2] or "")),
            "is_unanimous": (entropy == 0 and (n_a + n_b) > 0),
            "is_highly_split": (entropy > 0.9),
        })

    n_cells = len(cell_summaries)
    print(f"Total cell-groups (model × mode × template × q_type × 9-sample): {n_cells}")
    cells_with_data = [c for c in cell_summaries if (c["n_a"] + c["n_b"]) > 0]
    n_unan = sum(1 for c in cells_with_data if c["is_unanimous"])
    n_split = sum(1 for c in cells_with_data if c["is_highly_split"])
    mean_entropy = sum(c["entropy"] for c in cells_with_data) / len(cells_with_data)
    print(f"  Cells with at least one A/B response: {len(cells_with_data)}")
    print(f"  Unanimous (entropy=0): {n_unan} ({100*n_unan/len(cells_with_data):.2f}%)")
    print(f"  Highly split (entropy>0.9): {n_split} ({100*n_split/len(cells_with_data):.2f}%)")
    print(f"  Mean within-cell entropy: {mean_entropy:.4f}")

    # =================================================================
    print(f"\n{'=' * 80}")
    print("SECTION 1: Per (model, mode) within-cell consistency ranking")
    print(f"{'=' * 80}")
    by_mm: dict = defaultdict(list)
    for c in cells_with_data:
        by_mm[(c["model"], c["mode"])].append(c["entropy"])
    print(f"\n  {'rank':<5} {'model':<32} {'mode':<5} {'cells':<7} {'mean entropy':<14} "
          f"{'% unanimous':<14} {'% highly split':<15}")
    print("  " + "-" * 105)
    mm_summary = []
    for (m, mode), ents in by_mm.items():
        n = len(ents)
        mean_e = sum(ents) / n
        unan_pct = 100 * sum(1 for e in ents if e == 0) / n
        split_pct = 100 * sum(1 for e in ents if e > 0.9) / n
        mm_summary.append((m, mode, n, mean_e, unan_pct, split_pct))
    # Sort by mean entropy ascending (most consistent first)
    mm_summary.sort(key=lambda x: x[3])
    for rank, (m, mode, n, mean_e, unan_pct, split_pct) in enumerate(mm_summary, start=1):
        print(f"  {rank:<5} {m:<32} {mode:<5} {n:<7} {mean_e:<14.4f} {unan_pct:<14.2f} {split_pct:<15.2f}")

    # =================================================================
    print(f"\n\n{'=' * 80}")
    print("SECTION 2: Per question-type consistency")
    print(f"{'=' * 80}")
    by_qt: dict = defaultdict(list)
    for c in cells_with_data:
        by_qt[c["q_type"]].append(c["entropy"])
    print(f"\n  {'q-type':<25} {'cells':<7} {'mean entropy':<14} {'% unanimous':<14} {'% highly split':<15}")
    print("  " + "-" * 80)
    for qt, ents in sorted(by_qt.items()):
        n = len(ents)
        mean_e = sum(ents) / n
        unan_pct = 100 * sum(1 for e in ents if e == 0) / n
        split_pct = 100 * sum(1 for e in ents if e > 0.9) / n
        print(f"  {qt:<25} {n:<7} {mean_e:<14.4f} {unan_pct:<14.2f} {split_pct:<15.2f}")

    # =================================================================
    print(f"\n\n{'=' * 80}")
    print("SECTION 3: Per cluster consistency")
    print(f"{'=' * 80}")
    by_cluster: dict = defaultdict(list)
    for c in cells_with_data:
        by_cluster[c["cluster"]].append(c["entropy"])
    print(f"\n  {'cluster':<12} {'cells':<7} {'mean entropy':<14} {'% unanimous':<14} {'% highly split':<15}")
    print("  " + "-" * 70)
    for cl, ents in sorted(by_cluster.items()):
        n = len(ents)
        mean_e = sum(ents) / n
        unan_pct = 100 * sum(1 for e in ents if e == 0) / n
        split_pct = 100 * sum(1 for e in ents if e > 0.9) / n
        print(f"  {cl:<12} {n:<7} {mean_e:<14.4f} {unan_pct:<14.2f} {split_pct:<15.2f}")

    # =================================================================
    print(f"\n\n{'=' * 80}")
    print("SECTION 4: Reasoning effect on within-cell consistency (hybrid models)")
    print(f"{'=' * 80}")
    # For each (model, template, q_type), compare on-entropy vs off-entropy
    HYBRID = {"claude-4.7-opus-20260416", "gpt-5.5-20260423", "grok-4.3-20260430",
              "deepseek-v4-pro-20260423", "qwen3.6-max-preview-20260420"}
    paired: dict = defaultdict(dict)
    for c in cells_with_data:
        if c["model"] not in HYBRID:
            continue
        key = (c["model"], c["template"], c["q_type"])
        paired[key][c["mode"]] = c["entropy"]
    print(f"\n  {'model':<32} {'n pairs':<9} {'mean off ent':<14} {'mean on ent':<14} "
          f"{'Δ (on-off)':<12} {'sign test p':<13}")
    print("  " + "-" * 95)
    for model in sorted(HYBRID):
        pairs = [(v["off"], v["on"]) for k, v in paired.items()
                 if k[0] == model and "off" in v and "on" in v]
        if not pairs:
            continue
        off_mean = sum(p[0] for p in pairs) / len(pairs)
        on_mean = sum(p[1] for p in pairs) / len(pairs)
        diffs = [p[1] - p[0] for p in pairs]
        n_pos = sum(1 for d in diffs if d > 0)
        n_neg = sum(1 for d in diffs if d < 0)
        n_zero = sum(1 for d in diffs if d == 0)
        n_nonzero = n_pos + n_neg
        # Sign test (two-sided): p = 2 * min(P(X<=min(pos,neg) | n_nonzero, 0.5), ...)
        if n_nonzero == 0:
            p_val = 1.0
        else:
            smaller = min(n_pos, n_neg)
            cdf = sum(math.comb(n_nonzero, i) for i in range(0, smaller + 1)) * (0.5**n_nonzero)
            p_val = min(1.0, 2 * cdf)
        direction = "ON less consistent" if on_mean > off_mean else "ON more consistent"
        print(f"  {model:<32} {len(pairs):<9} {off_mean:<14.4f} {on_mean:<14.4f} "
              f"{on_mean - off_mean:+.4f}     {p_val:<13.4g}  ({direction})")

    # =================================================================
    print(f"\n\n{'=' * 80}")
    print("SECTION 5: Highly split cells (entropy > 0.9) — where do they concentrate?")
    print(f"{'=' * 80}")
    split_cells = [c for c in cells_with_data if c["is_highly_split"]]
    print(f"\n  Total highly-split cells: {len(split_cells)}")
    # Breakdown by (model, mode)
    by_mm_split: dict = defaultdict(int)
    by_qt_split: dict = defaultdict(int)
    by_cluster_split: dict = defaultdict(int)
    for c in split_cells:
        by_mm_split[(c["model"], c["mode"])] += 1
        by_qt_split[c["q_type"]] += 1
        by_cluster_split[c["cluster"]] += 1
    print(f"\n  By (model, mode):")
    for (m, mode), n in sorted(by_mm_split.items(), key=lambda x: -x[1]):
        print(f"    {m:<32} {mode:<5} {n}")
    print(f"\n  By q-type:")
    for qt, n in sorted(by_qt_split.items(), key=lambda x: -x[1]):
        print(f"    {qt:<25} {n}")
    print(f"\n  By cluster:")
    for cl, n in sorted(by_cluster_split.items(), key=lambda x: -x[1]):
        print(f"    {cl:<12} {n}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
