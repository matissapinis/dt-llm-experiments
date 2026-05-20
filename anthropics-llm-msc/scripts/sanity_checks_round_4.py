#!/usr/bin/env python3
"""Round 4 sanity checks — focus on what this session might have introduced,
plus checks not yet performed.

Sections:
  1. Schema integrity — every cell has parse_quality + key metadata
  2. parse_quality / extracted_choice consistency (rules of the parser hold)
  3. Derived fields agree with current extracted_choice + row_order + preferred_actions
  4. Cell count per (model, mode) still 1152 each
  5. parse_quality distribution by model
  6. Spot-check 14 multi_match_mixed verdicts agree with response tail
  7. Inspect 41 no_final_answer cells — any recoverable in non-standard formats?
  8. Inspect 6 off_menu_refusal cells — verify they're genuine refusals
  9. Re-verify the headline numbers (SSA reference-class disambiguation drift, capability accuracy) — no regression
 10. Cross-tab parse_quality × correctness for capability questions

Usage:
  python scripts/sanity_checks_round4.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from framework import NewcombExperiment  # type: ignore  # noqa: E402

D = Path("experiment_results/main_run_20260516")
EXPECTED_KEYS = {
    "template_name", "question_type", "model_id_openrouter",
    "row_order", "preferred_actions", "response", "extracted_choice",
    "usage_statistics", "parse_quality",
}


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
    section_header("1. SCHEMA INTEGRITY — required keys present")
    missing_key_counts: Counter[str] = Counter()
    for c in cells:
        for k in EXPECTED_KEYS:
            if k not in c:
                missing_key_counts[k] += 1
    if not missing_key_counts:
        print(f"  ✓ all {len(EXPECTED_KEYS)} required keys present on all {n} cells")
    else:
        for k, m in missing_key_counts.most_common():
            print(f"  ✗ {k}: missing on {m} cells")

    # =================================================================
    section_header("2. parse_quality / extracted_choice INVARIANTS")
    # Cells with quality=*_refusal, no_final_answer, empty_response must have extracted_choice = None
    # Cells with quality=strict_*, wrapped_*, multi_match_* must have extracted_choice in {A,B}
    null_required = {"off_menu_refusal", "no_final_answer", "empty_response"}
    nonnull_required = {"strict_clean", "strict_with_continuation",
                        "wrapped_choice", "wrapped_option", "wrapped_answer",
                        "multi_match_consistent", "multi_match_mixed"}
    violations = []
    for c in cells:
        q, ch = c.get("parse_quality"), c.get("extracted_choice")
        if q in null_required and ch is not None:
            violations.append(("should be null", c["_filename"], q, ch))
        elif q in nonnull_required and ch not in ("A", "B"):
            violations.append(("should be A/B", c["_filename"], q, ch))
        elif q is None:
            violations.append(("missing quality", c["_filename"], q, ch))
    if not violations:
        print(f"  ✓ all {n} cells satisfy parse_quality/extracted_choice invariants")
    else:
        print(f"  ✗ {len(violations)} violations:")
        for v in violations[:10]:
            print(f"    {v}")

    # =================================================================
    section_header("3. DERIVED FIELDS consistent with extracted_choice + row_order + preferred_actions")
    exp = NewcombExperiment(base_output_dir="/tmp/_sanity4", temperature=0.8)
    cap_mismatches, alignment_mismatches = [], []
    for c in cells:
        ch = c.get("extracted_choice")
        if not ch:
            continue
        pref = c.get("preferred_actions") or {}
        row = c.get("row_order")
        qt = c.get("question_type")
        if not (pref and row and qt):
            continue
        # Check correctness
        exp.problem_structure = pref
        if qt.endswith("_capability"):
            expected = exp.check_correctness(ch, qt, pref, row_order=row)
            actual = c.get("correct_capability_answer")
            if expected is not None and expected != actual:
                cap_mismatches.append((c["_filename"], ch, row, expected, actual))
        # Check alignment
        align = exp.determine_alignment(ch, pref, row_order=row)
        for k, v in align.items():
            if c.get(k) != v:
                alignment_mismatches.append((c["_filename"], k, c.get(k), v))
    print(f"  capability-answer mismatches: {len(cap_mismatches)}")
    for m in cap_mismatches[:5]:
        print(f"    {m}")
    print(f"  alignment-field mismatches:   {len(alignment_mismatches)}")
    for m in alignment_mismatches[:5]:
        print(f"    {m}")

    # =================================================================
    section_header("4. CELL COUNT per (model, mode) — should be exactly 1152 each")
    cc: Counter = Counter()
    for c in cells:
        model = c.get("model_id_openrouter") or "<missing>"
        # mode inferred from filename
        mode = "on" if c["_filename"].endswith("_on.json") else "off"
        cc[(model, mode)] += 1
    bad = [k for k, v in cc.items() if v != 1152]
    if not bad:
        print(f"  ✓ all 12 (model, mode) pairs have exactly 1152 cells")
    else:
        for k in bad:
            print(f"  ✗ {k}: {cc[k]} (expected 1152)")

    # =================================================================
    section_header("5. parse_quality DISTRIBUTION by (model, mode)")
    pq_by_mm: dict = defaultdict(Counter)
    for c in cells:
        model = c.get("model_id_openrouter") or "<missing>"
        mode = "on" if c["_filename"].endswith("_on.json") else "off"
        pq_by_mm[(model, mode)][c.get("parse_quality")] += 1
    qualities_all = sorted({q for cnt in pq_by_mm.values() for q in cnt})
    print(f"  {'model':<45} {'mode':<4}  " + "  ".join(f"{q[:20]:>12}" for q in qualities_all))
    for (model, mode), cnt in sorted(pq_by_mm.items()):
        nm = model.split('/')[-1][:42]
        row = f"  {nm:<45} {mode:<4}  " + "  ".join(f"{cnt.get(q, 0):>12}" for q in qualities_all)
        print(row)

    # =================================================================
    section_header("6. multi_match_mixed VERDICTS — last 200 chars of each response")
    mixed = [c for c in cells if c.get("parse_quality") == "multi_match_mixed"]
    print(f"  {len(mixed)} cells in this bucket")
    for c in mixed:
        ch = c.get("extracted_choice")
        resp = c.get("response") or ""
        tail = resp[-200:].replace('\n', ' ⏎ ')
        print(f"  [{ch}] {c['_filename'][:70]}")
        print(f"        …{tail}")

    # =================================================================
    section_header("7. no_final_answer CELLS — any recoverable in non-standard formats?")
    no_fa = [c for c in cells if c.get("parse_quality") == "no_final_answer"]
    # heuristics: look for "answer: X", "the answer is X", "I choose X", "Choice X", "Option X"
    alt_patterns = [
        (r"\bthe\s+answer\s+is\s+([AB])\b", "the_answer_is"),
        (r"\bI\s+(?:choose|pick|select|go\s+with)\s+(?:choice\s+)?([AB])\b", "I_choose"),
        (r"\banswer:\s*([AB])\b", "answer_colon"),
        (r"\bmy\s+answer\s+is\s+([AB])\b", "my_answer_is"),
        (r"\b(?:choice|option)\s+([AB])\b\s*(?:is\s+(?:my\s+)?(?:final\s+)?answer|is\s+correct|is\s+the\s+answer)", "X_is_answer"),
    ]
    recoverable = []
    truly_no_answer = []
    for c in no_fa:
        resp = c.get("response") or ""
        if not resp:
            continue
        recovered = None
        for pat, name in alt_patterns:
            m = re.search(pat, resp, re.IGNORECASE)
            if m:
                recovered = (m.group(1).upper(), name)
                break
        if recovered:
            recoverable.append((c["_filename"], recovered, resp[-150:].replace('\n', ' ⏎ ')))
        else:
            truly_no_answer.append((c["_filename"], len(resp), resp[-150:].replace('\n', ' ⏎ ')))
    print(f"  Total no_final_answer with content: {len(no_fa)}")
    print(f"  Potentially recoverable via alt patterns: {len(recoverable)}")
    for name, (letter, pat_name), tail in recoverable[:10]:
        print(f"    [{letter}] via {pat_name}  {name[:60]}")
        print(f"        …{tail}")
    print(f"  Truly no parseable answer: {len(truly_no_answer)}")
    for name, rlen, tail in truly_no_answer[:5]:
        print(f"    len={rlen}  {name[:60]}")
        print(f"        …{tail}")

    # =================================================================
    section_header("8. off_menu_refusal CELLS")
    refusals = [c for c in cells if c.get("parse_quality") == "off_menu_refusal"]
    for c in refusals:
        resp = c.get("response") or ""
        # find the refusal context
        m = re.search(r"FINAL\s+ANSWER\s*:.{0,100}", resp, re.IGNORECASE)
        ctx = m.group(0).replace('\n', ' ⏎ ')[:160] if m else "<no match>"
        print(f"  {c['_filename'][:70]}")
        print(f"    {ctx}")

    # =================================================================
    section_header("9. RE-VERIFY HEADLINE NUMBERS — no regression from parser changes")
    # SSA reference-class disambiguation drift (re-compute)
    # capability accuracy by model
    cap_correct_by_mm: dict = defaultdict(lambda: [0, 0])  # [correct, total]
    for c in cells:
        qt = c.get("question_type") or ""
        if not qt.endswith("_capability"):
            continue
        model = c.get("model_id_openrouter") or "<missing>"
        mode = "on" if c["_filename"].endswith("_on.json") else "off"
        ca = c.get("correct_capability_answer")
        if ca is None:
            continue
        cap_correct_by_mm[(model, mode)][0] += int(ca)
        cap_correct_by_mm[(model, mode)][1] += 1
    print(f"  capability accuracy by (model, mode):")
    for (m, mo), (corr, tot) in sorted(cap_correct_by_mm.items()):
        print(f"    {m.split('/')[-1]:<42} {mo:<4} {corr}/{tot} {100*corr/tot:5.1f}%")

    # =================================================================
    section_header("10. CROSS-TAB parse_quality × correctness on capability questions")
    qual_correct: dict = defaultdict(lambda: [0, 0])
    for c in cells:
        qt = c.get("question_type") or ""
        if not qt.endswith("_capability"):
            continue
        q = c.get("parse_quality")
        ca = c.get("correct_capability_answer")
        if ca is None:
            continue
        qual_correct[q][0] += int(ca)
        qual_correct[q][1] += 1
    print(f"  {'parse_quality':<30} {'correct':<10} {'total':<10} {'acc':<8}")
    for q in sorted(qual_correct):
        corr, tot = qual_correct[q]
        print(f"  {q:<30} {corr:<10} {tot:<10} {100*corr/tot:5.1f}%")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
