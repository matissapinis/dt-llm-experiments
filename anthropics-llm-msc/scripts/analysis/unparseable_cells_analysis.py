#!/usr/bin/env python3
"""QT2 — Analysis of all unparseable / off-menu cells (n=123 per RQ8).

Per parse_quality field, summarize:
  - Cell counts by (model, mode, q_type, parse_quality)
  - Cell counts by (parse_quality, problem_class)
  - For each parse_quality category, show 5 representative response excerpts
  - Refire/accept/parser-extension recommendation per category

The off-menu categories (from the parser specification):
  - empty_response: model returned essentially no text
  - no_final_answer: model produced text but no parseable FINAL ANSWER line
  - off_menu_refusal: model declined to pick A or B (refusal, hedging, etc.)
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

D = Path("experiment_results/main_run_20260516")
OFF_MENU_QUALITIES = {"empty_response", "no_final_answer", "off_menu_refusal"}


def parse_mode(filename: str) -> str:
    if filename.endswith("_on.json"):
        return "on"
    if filename.endswith("_off.json"):
        return "off"
    return "?"


def parse_problem_class(template_name: str) -> str:
    m = re.match(r"\d+_standard_(sb|inc|dd|padd)_", template_name or "")
    return m.group(1) if m else "?"


def load_off_menu():
    cells = []
    for f in sorted(D.glob("*.json")):
        d = json.load(open(f))
        pq = d.get("parse_quality")
        ch = d.get("extracted_choice")
        # Off-menu = parse_quality in off-menu list OR extracted_choice not in {A, B}
        if pq in OFF_MENU_QUALITIES or ch not in ("A", "B"):
            model = (d.get("model_id_openrouter") or "").split("/")[-1]
            d["mode"] = parse_mode(f.name)
            d["model_short"] = model
            d["problem_class"] = parse_problem_class(d.get("template_name", ""))
            d["_filename"] = f.name
            cells.append(d)
    return cells


def main() -> int:
    cells = load_off_menu()
    print(f"Loaded {len(cells)} off-menu / unparseable cells")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 1: Off-menu cells by parse_quality + extracted_choice status")
    print(f"{'=' * 100}")
    pq_count = Counter(c.get("parse_quality") for c in cells)
    print(f"\n  parse_quality breakdown:")
    for pq, n in pq_count.most_common():
        print(f"    {pq!r:<30} {n}")
    print(f"\n  extracted_choice values:")
    ch_count = Counter(c.get("extracted_choice") for c in cells)
    for ch, n in ch_count.most_common():
        print(f"    {ch!r:<10} {n}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 2: By (model, mode)")
    print(f"{'=' * 100}")
    by_mm = Counter((c["model_short"], c["mode"]) for c in cells)
    print(f"\n  {'model':<32} {'mode':<5} {'off-menu count':<15}")
    print("  " + "-" * 55)
    for (m, mode), n in by_mm.most_common():
        print(f"  {m:<32} {mode:<5} {n}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 3: By (parse_quality, model, mode) joint distribution")
    print(f"{'=' * 100}")
    by_pq_mm = defaultdict(Counter)
    for c in cells:
        by_pq_mm[c.get("parse_quality")][(c["model_short"], c["mode"])] += 1
    for pq in sorted(by_pq_mm.keys()):
        print(f"\n  parse_quality = {pq!r}")
        for (m, mode), n in by_pq_mm[pq].most_common():
            print(f"    {m:<32} {mode:<5} {n}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 4: By (parse_quality, problem_class, q_type)")
    print(f"{'=' * 100}")
    by_pq_pc_qt = defaultdict(Counter)
    for c in cells:
        by_pq_pc_qt[c.get("parse_quality")][(c["problem_class"], c.get("question_type"))] += 1
    for pq in sorted(by_pq_pc_qt.keys()):
        print(f"\n  parse_quality = {pq!r}")
        for (pc, qt), n in by_pq_pc_qt[pq].most_common():
            print(f"    pc={pc:<5} qt={qt:<22} {n}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 5: Representative response excerpts per parse_quality")
    print(f"{'=' * 100}")
    by_pq = defaultdict(list)
    for c in cells:
        by_pq[c.get("parse_quality")].append(c)
    import random
    random.seed(20260518)
    for pq, lst in by_pq.items():
        print(f"\n  --- parse_quality = {pq!r}  (n={len(lst)}) ---")
        sample = random.sample(lst, min(5, len(lst)))
        for i, c in enumerate(sample, 1):
            resp = c.get("response") or ""
            rt = c.get("reasoning_trace") or ""
            preview = (resp or rt or "").strip()
            preview_show = preview[:300] if len(preview) > 300 else preview
            tail_show = f"  [...end of response: ...{preview[-200:]!r}]" if len(preview) > 500 else ""
            print(f"\n    [{i}] {c['_filename'][:80]}")
            print(f"        model={c['model_short']}/{c['mode']}  "
                  f"q_type={c.get('question_type')}  pc={c['problem_class']}")
            print(f"        finish_reason={c.get('finish_reason')!r}  "
                  f"completion_tokens={(c.get('usage_statistics') or {}).get('completion_tokens')}  "
                  f"reasoning_tokens={(c.get('usage_statistics') or {}).get('reasoning_tokens')}")
            print(f"        response (first 300 chars):")
            print(f"          {preview_show!r}")
            if tail_show:
                print(f"        {tail_show}")

    # =================================================================
    print(f"\n{'=' * 100}")
    print("SECTION 6: Recommendation per parse_quality category")
    print(f"{'=' * 100}")
    print(f"\n  Heuristic recommendations:")
    print(f"")
    print(f"  - empty_response:    likely infrastructure/truncation. Candidate for re-fire if")
    print(f"                       finish_reason = 'length' or completion_tokens hit limit.")
    print(f"                       Accept-as-missing if intentional (model produced 0 tokens by choice).")
    print(f"")
    print(f"  - no_final_answer:   model produced reasoning but no parseable FINAL ANSWER.")
    print(f"                       Two sub-cases: (a) reasoning was truncated before reaching")
    print(f"                       conclusion (finish_reason='length' or near-max-tokens) → re-fire;")
    print(f"                       (b) model gave a long answer without committing → either extend")
    print(f"                       parser (Stage 2 wrapper recovery), or accept as off-menu data.")
    print(f"")
    print(f"  - off_menu_refusal:  model explicitly declined to pick A or B. Genuine non-response;")
    print(f"                       should be accepted as off-menu (don't re-fire — same model would")
    print(f"                       give same refusal).")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
