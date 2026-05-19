#!/usr/bin/env python3
"""Apply the settled two-stage parser to all Main run cells.

Main run (strict literal): exactly one `FINAL ANSWER: [A-Za-z]` match where
letter ∈ {A, B}. Records `parse_quality = strict_clean` or `strict_with_continuation`
depending on whether the letter is followed by substantive text.

Stage 2 (corpus-validated triage), applied when Main run doesn't yield a single
A/B match:
  - `FINAL ANSWER: Choice [AB]` → wrapped_choice
  - `FINAL ANSWER: Option [AB]` → wrapped_option
  - `FINAL ANSWER: Answer [AB]` → wrapped_answer
  - `FINAL ANSWER: (Neither|N/A|None)` → off_menu_refusal (null)
  - Multiple matches, all same A/B → multi_match_consistent
  - Multiple matches, mixed letters or with non-A/B → multi_match_mixed
  - Multiple matches, none A/B → multi_match_no_ab (null)
  - No match at all → no_final_answer (null)
  - Empty response → empty_response (null)

Each cell gets a `parse_quality` field. `extracted_choice` is updated only
when the new parse disagrees with the current value (which should be rare,
since the previous parser already incorporated most of these rules).
Re-derives correctness/alignment for any cells whose extracted_choice changes.

Usage:
  python scripts/two_stage_parse_main_run.py
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from framework import NewcombExperiment  # type: ignore  # noqa: E402

MAIN_RUN_DIR = Path("experiment_results/main_run_20260516")

# Unified pattern: captures any FINAL ANSWER form in a single pass.
# Group 'wrapper' is "Choice"/"Option"/"Answer" if present, else None.
# Group 'letter' is the first A-Z after the (optional) wrapper.
# Order matters: try wrapper first so "Choice A" captures wrapper="Choice", letter="A",
# not letter="C".
UNIFIED_PAT = re.compile(
    r"FINAL\s+ANSWER\s*:\s*"
    r"(?:(?P<wrapper>Choice|Option|Answer)\s+)?"
    r"[\W_]*"
    r"(?P<letter>[A-Za-z])\b",
    re.IGNORECASE,
)
REFUSAL_PAT = re.compile(
    r"FINAL\s+ANSWER\s*:\s*(?:Choice\s+)?(Neither|N/?A|None)\b",
    re.IGNORECASE,
)


def two_stage_parse(response_text: str | None) -> tuple[str | None, str]:
    """Return (extracted_letter, parse_quality).

    Principle: the model's TERMINAL `FINAL ANSWER` commit is the answer. We
    scan the response for any FINAL ANSWER occurrence (strict bare letter or
    wrapped 'Choice X'/'Option X'/'Answer X' form), take the LAST occurrence
    whose letter is A or B, and tag the parse with what form was used and
    whether there were multiple competing commits.
    """
    if not response_text:
        return None, "empty_response"

    all_matches = []
    for m in UNIFIED_PAT.finditer(response_text):
        wrapper = (m.group("wrapper") or "").lower()
        letter = m.group("letter").upper()
        all_matches.append({
            "start": m.start(),
            "end": m.end(),
            "wrapper": wrapper,
            "letter": letter,
        })

    # Check for off-menu refusal pattern even when nothing else parsed
    refusal_hit = REFUSAL_PAT.search(response_text)

    if not all_matches:
        return (None, "off_menu_refusal") if refusal_hit else (None, "no_final_answer")

    # Filter to A/B letters only
    ab_matches = [m for m in all_matches if m["letter"] in ("A", "B")]
    if not ab_matches:
        # Every FINAL ANSWER had a non-A/B letter; check refusal first
        if refusal_hit:
            return None, "off_menu_refusal"
        # Multiple matches all non-A/B (e.g., model wrote "FINAL ANSWER: X" multiple times)
        if len(all_matches) > 1:
            return None, "multi_match_no_ab"
        return None, "no_final_answer"

    # If model wrote a refusal AFTER the last A/B match, prefer the refusal
    # (e.g., "FINAL ANSWER: A ... actually FINAL ANSWER: N/A")
    winner = ab_matches[-1]
    if refusal_hit and refusal_hit.start() > winner["end"]:
        return None, "off_menu_refusal"

    if len(all_matches) == 1:
        if winner["wrapper"]:
            return winner["letter"], f"wrapped_{winner['wrapper']}"
        tail = response_text[winner["end"]:]
        tail_non_ws = re.sub(r"\s+", "", tail)
        if len(tail_non_ws) <= 2:
            return winner["letter"], "strict_clean"
        return winner["letter"], "strict_with_continuation"

    # Multiple FINAL ANSWER occurrences
    ab_letters = [m["letter"] for m in ab_matches]
    distinct = set(ab_letters)
    # Were any of the multi-matches non-A/B (X meta-quote, etc.)?
    has_non_ab = any(m["letter"] not in ("A", "B") for m in all_matches)
    if len(distinct) == 1 and not has_non_ab:
        return winner["letter"], "multi_match_consistent"
    return winner["letter"], "multi_match_mixed"


def main() -> int:
    exp = NewcombExperiment(base_output_dir="/tmp/_two_stage", temperature=0.8)
    files = sorted(MAIN_RUN_DIR.glob("*.json"))

    qual_counter: Counter[str] = Counter()
    choice_changes: list[tuple[str, str | None, str | None, str]] = []
    n_total = 0

    for fp in files:
        with open(fp) as f:
            d = json.load(f)
        n_total += 1
        resp = d.get("response") or ""
        old_choice = d.get("extracted_choice")
        new_choice, quality = two_stage_parse(resp)
        qual_counter[quality] += 1

        d["parse_quality"] = quality
        if new_choice != old_choice:
            choice_changes.append((fp.name, old_choice, new_choice, quality))
            d["extracted_choice"] = new_choice
            # Re-derive correctness / alignment if choice present
            preferred = d.get("preferred_actions") or {}
            row_order = d.get("row_order")
            q_type = d.get("question_type")
            if new_choice and preferred and row_order:
                exp.problem_structure = preferred
                alignment = exp.determine_alignment(new_choice, preferred, row_order=row_order)
                d.update(alignment)
                correctness = exp.check_correctness(new_choice, q_type, preferred, row_order=row_order)
                if correctness is not None:
                    d["correct_capability_answer"] = correctness
            elif not new_choice:
                for k in ("correct_capability_answer", "ssa_aligned", "sia_aligned",
                          "cdt_aligned", "edt_aligned"):
                    d.pop(k, None)

        with open(fp, "w") as f:
            json.dump(d, f, indent=2)

    print(f"Processed {n_total} cells")
    print(f"\n=== parse_quality distribution ===")
    for q, n in qual_counter.most_common():
        pct = 100.0 * n / n_total
        print(f"  {q:<32} {n:>6}  ({pct:5.2f}%)")

    print(f"\n=== Cells where extracted_choice changed: {len(choice_changes)} ===")
    if choice_changes:
        for name, old, new, qual in choice_changes[:30]:
            print(f"  {old!r:>8} → {new!r:<8}  [{qual}]  {name[:60]}")
        if len(choice_changes) > 30:
            print(f"  ... and {len(choice_changes) - 30} more")

    # Categorize by "trust tier"
    auto_trust = sum(qual_counter[q] for q in (
        "strict_clean", "strict_with_continuation",
        "wrapped_choice", "wrapped_option", "wrapped_answer",
        "multi_match_consistent",
    ))
    needs_attention = sum(qual_counter[q] for q in (
        "multi_match_mixed", "multi_match_no_ab",
        "no_final_answer", "off_menu_refusal", "empty_response",
    ))
    print(f"\n=== Trust tiers ===")
    print(f"  Auto-trust (clean / wrapper / consistent multi): {auto_trust}  ({100*auto_trust/n_total:.2f}%)")
    print(f"  Needs attention (mixed / no-answer / refusal):   {needs_attention}  ({100*needs_attention/n_total:.2f}%)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
