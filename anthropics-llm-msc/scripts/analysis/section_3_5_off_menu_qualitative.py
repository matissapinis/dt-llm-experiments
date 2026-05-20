#!/usr/bin/env python3
"""QT4 — Off-menu qualitative analysis (beyond QT2's parse-quality typology).

Three sub-analyses:

  Q4a. Deep-dive on hard-off-menu refusals (parse_quality == off_menu_refusal).
       All 6 cells: full quote + categorization of the refusal reason.

  Q4b. Hedged-but-parsed capability cells: model committed to A or B but with
       explicit hedging language ("if forced", "neither truly", "approximately",
       "between A and B", "the closest", "best of bad options", etc.). These
       are parsed as A/B but the parsing masks reluctance.

  Q4c. Quantify the rate of hedged-commitment across (model, mode, q_type).
       Test: do certain models hedge more on capability questions, and does
       the hedging concentrate on the problem-cluster combinations where the
       formal answer is ambiguous?
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

D = Path("experiment_results/main_run_20260516")


def parse_mode(filename: str) -> str:
    if filename.endswith("_on.json"):
        return "on"
    if filename.endswith("_off.json"):
        return "off"
    return "?"


def parse_problem_class(template_name: str) -> str:
    m = re.match(r"\d+_standard_(sb|inc|dd|padd)_", template_name or "")
    return m.group(1) if m else "?"


# Hedging-language patterns: model parsed as A/B but with reluctance
HEDGING_PATTERNS = [
    (r"\bif\s+(forced|i\s+must|i\s+have)\s+to\s+(choose|pick|select)", "forced-choice"),
    (r"\bneither\s+(option|choice|answer|fully|truly|exactly)", "neither-fits"),
    (r"\b(closest|best)\s+(option|choice|match|approximation|fit)", "closest-match"),
    (r"\bbetween\s+(a\s+and\s+b|the\s+(two\s+)?options)", "between-options"),
    (r"\b(none\s+of\s+the|none\s+of\s+these)\s+(options|choices|answers)", "none-fit"),
    (r"\bbest\s+of\s+(bad|the\s+two|the\s+options)", "best-of-bad"),
    (r"\b(must\s+pick|forced\s+to\s+pick|have\s+to\s+pick)", "must-pick"),
    (r"\bthe\s+(true|actual|correct)\s+(value|answer|credence)\s+(is|would\s+be)\s+(not|neither)", "true-not-in-menu"),
    (r"\bnot\s+exactly\s+(in|among|one\s+of)", "not-exactly-listed"),
    (r"\bsince\s+(2/3|that\s+value)\s+is\s+not\s+(an\s+option|listed|available)", "explicit-menu-mismatch"),
    (r"\bcloser\s+(to\s+the\s+true|to\s+the\s+correct|to\s+the\s+actual)", "closer-to-truth"),
    (r"\b(can|cannot|won['’]?t|will\s+not)\s+(commit|decide)", "non-commitment"),
]


def has_hedging(text):
    text_l = text.lower()
    hits = []
    for pat, label in HEDGING_PATTERNS:
        if re.search(pat, text_l, flags=re.IGNORECASE):
            hits.append(label)
    return hits


def load_cells():
    cells = []
    for f in sorted(D.glob("*.json")):
        d = json.load(open(f))
        model = (d.get("model_id_openrouter") or "").split("/")[-1]
        d["mode"] = parse_mode(f.name)
        d["model_short"] = model
        d["problem_class"] = parse_problem_class(d.get("template_name", ""))
        d["_filename"] = f.name
        rt = d.get("reasoning_trace") or ""
        resp = d.get("response") or ""
        d["_trace"] = rt if rt.strip() else resp
        cells.append(d)
    return cells


def main() -> int:
    cells = load_cells()
    print(f"Loaded {len(cells)} cells\n")

    # =================================================================
    # Q4a: Hard-off-menu refusals (parse_quality == off_menu_refusal)
    # =================================================================
    print(f"\n{'=' * 100}")
    print("Q4a: HARD-OFF-MENU REFUSALS (parse_quality == off_menu_refusal)")
    print("     Full quote and categorization for each of the 6 cells.")
    print(f"{'=' * 100}")

    refusals = [c for c in cells if c.get("parse_quality") == "off_menu_refusal"]
    print(f"\n  Total off_menu_refusal cells: {len(refusals)}")
    print(f"  All are on PADD/DD sia_capability per QT2.\n")

    for i, c in enumerate(refusals, 1):
        print(f"\n  ─────────────────────────────────────────")
        print(f"  Refusal #{i}: {c['_filename'][:80]}")
        print(f"    model={c['model_short']:<32} mode={c['mode']:<3}  "
              f"q_type={c.get('question_type')}  pc={c['problem_class']}")
        print(f"    extracted_choice={c.get('extracted_choice')!r}  "
              f"finish_reason={c.get('finish_reason')!r}")
        print(f"    Full response:")
        text = c["_trace"]
        for line in text.split("\n"):
            print(f"      {line}")

    # =================================================================
    # Q4b: Hedged-but-parsed capability cells
    # =================================================================
    print(f"\n\n{'=' * 100}")
    print("Q4b: HEDGED-BUT-PARSED CAPABILITY CELLS")
    print("     Capability cells where extracted_choice is A or B but the response")
    print("     contains explicit hedging language (forced-choice, neither-fits, etc.)")
    print(f"{'=' * 100}")

    cap_cells = [c for c in cells
                 if c.get("question_type") in ("sia_capability", "ssa_capability")
                 and c.get("extracted_choice") in ("A", "B")]
    hedged = []
    for c in cap_cells:
        hits = has_hedging(c["_trace"])
        if hits:
            c["_hedging_labels"] = hits
            hedged.append(c)

    print(f"\n  Capability cells analyzed (parsed to A/B): {len(cap_cells)}")
    print(f"  Cells with ≥1 hedging-language match: {len(hedged)} "
          f"({100*len(hedged)/len(cap_cells):.2f}%)")

    # By hedging category
    cat_counter = Counter()
    for c in hedged:
        for lab in c["_hedging_labels"]:
            cat_counter[lab] += 1
    print(f"\n  Hits per hedging category:")
    for label, count in cat_counter.most_common():
        print(f"    {label:<30} {count}")

    # By (model, mode)
    print(f"\n  Hedged-cell rate per (model, mode):")
    print(f"  {'model':<32} {'mode':<5} {'hedged cells':<14} {'% of capability cells':<24}")
    print("  " + "-" * 90)
    n_cap_per_mm = Counter()
    n_hedge_per_mm = Counter()
    for c in cap_cells:
        n_cap_per_mm[(c["model_short"], c["mode"])] += 1
    for c in hedged:
        n_hedge_per_mm[(c["model_short"], c["mode"])] += 1
    mm_sorted = sorted(n_cap_per_mm.keys(),
                       key=lambda mm: -n_hedge_per_mm.get(mm, 0) / max(1, n_cap_per_mm[mm]))
    for mm in mm_sorted:
        n_total = n_cap_per_mm[mm]
        n_h = n_hedge_per_mm.get(mm, 0)
        pct = 100 * n_h / n_total if n_total else 0
        print(f"  {mm[0]:<32} {mm[1]:<5} {n_h:<14} {pct:.2f}%")

    # By (problem_class, q_type)
    print(f"\n  Hedged-cell rate per (problem_class, q_type):")
    print(f"  {'pc':<6} {'q_type':<22} {'hedged':<10} {'total':<8} {'%':<8}")
    print("  " + "-" * 60)
    n_cap_per_pcqt = Counter()
    n_hedge_per_pcqt = Counter()
    for c in cap_cells:
        n_cap_per_pcqt[(c["problem_class"], c.get("question_type"))] += 1
    for c in hedged:
        n_hedge_per_pcqt[(c["problem_class"], c.get("question_type"))] += 1
    for key in sorted(n_cap_per_pcqt.keys()):
        n_total = n_cap_per_pcqt[key]
        n_h = n_hedge_per_pcqt.get(key, 0)
        pct = 100 * n_h / n_total if n_total else 0
        print(f"  {key[0]:<6} {key[1]:<22} {n_h:<10} {n_total:<8} {pct:.2f}%")

    # =================================================================
    # Q4c: Representative hedged-but-parsed quotes
    # =================================================================
    print(f"\n\n{'=' * 100}")
    print("Q4c: REPRESENTATIVE HEDGED-BUT-PARSED EXAMPLES")
    print("     Most informative hedging quotes — model committed to A/B but flagged the menu")
    print(f"{'=' * 100}")

    # Show top 12: prioritize cases with multiple hedging labels
    hedged.sort(key=lambda c: -len(set(c["_hedging_labels"])))
    print(f"\n  Showing top 12 hedged cases (sorted by number of distinct hedging-categories):")
    for i, c in enumerate(hedged[:12], 1):
        print(f"\n  ─────────────────────────────────────────")
        print(f"  Hedged #{i}: {c['_filename'][:80]}")
        print(f"    model={c['model_short']:<32} mode={c['mode']:<3}  "
              f"q_type={c.get('question_type')}  pc={c['problem_class']}")
        print(f"    chose={c.get('extracted_choice')}")
        print(f"    hedging labels: {sorted(set(c['_hedging_labels']))}")
        # Extract the hedging snippet (find first hedging pattern match)
        text = c["_trace"]
        snippet_shown = False
        for pat, label in HEDGING_PATTERNS:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                start = max(0, m.start() - 150)
                end = min(len(text), m.end() + 250)
                snippet = text[start:end].replace("\n", " ")
                print(f"    snippet (around hedging '{label}'):")
                print(f"      ...{snippet.strip()}...")
                snippet_shown = True
                break
        if not snippet_shown:
            print(f"    (no hedging snippet extractable)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
