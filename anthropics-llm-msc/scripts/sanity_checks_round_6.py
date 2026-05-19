#!/usr/bin/env python3
"""Round 6 sanity checks — narrow due-diligence pass before commit.

Sections:
  1. Prompt integrity
     - Every ssa_capability cell contains SSA reference-class disambiguation clause
     - Every cell's user_prompt has "Choice A:" and "Choice B:" with distinct text
     - question_type matches the principle named in system_prompt
  2. Row order symmetry
     - For each (template, q_type) pair, row_12 and row_21 should show mirror
       model behavior (model's ssa-preferred choice should be invariant to row)
  3. File structural integrity
     - All 13,824 files parse cleanly, file sizes reasonable
  4. Post-refire spot-check (3 cells)
     - _refire_history well-formed, responses coherent, derived fields right
  5. Decontamination final spot-check (84 cells)
     - All have _refire_history, reasoning_tokens=0, derived fields consistent
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

SSA_DISAMBIGUATION_CLAUSE = "with the reference class contained within each hypothesis separately"


def section_header(title: str):
    print(f"\n{'=' * 80}\n{title}\n{'=' * 80}")


def main() -> int:
    cells = []
    for f in sorted(D.glob("*.json")):
        d = json.load(open(f))
        d["_filename"] = f.name
        cells.append(d)
    n = len(cells)
    print(f"Loaded {n} cells")

    # =================================================================
    section_header("1. PROMPT INTEGRITY")
    ssa_clause_missing = []
    choice_text_issues = []
    qtype_principle_mismatches = []
    PRINCIPLE_FOR_QT = {
        "ssa_capability": "Self-Sampling Assumption",
        "sia_capability": "Self-Indication Assumption",
    }
    for c in cells:
        qt = c.get("question_type", "")
        sp = c.get("system_prompt", "") or ""
        up = c.get("user_prompt", "") or ""

        # (a) SSA reference-class disambiguation clause must be on every SSA-capability cell's system prompt
        if qt == "ssa_capability" and SSA_DISAMBIGUATION_CLAUSE not in sp:
            ssa_clause_missing.append(c["_filename"])

        # (b) user_prompt must have both Choice A: and Choice B: with distinct text
        a_match = re.search(r"Choice A:\s*(.{10,200}?)(?=Choice [AB]:|$)", up, re.DOTALL)
        b_match = re.search(r"Choice B:\s*(.{10,200}?)(?=Choice [AB]:|$)", up, re.DOTALL)
        if not a_match or not b_match:
            choice_text_issues.append((c["_filename"], "missing Choice A or B"))
        else:
            a_text = a_match.group(1).strip()
            b_text = b_match.group(1).strip()
            if a_text == b_text:
                choice_text_issues.append((c["_filename"], "Choice A text == Choice B text"))

        # (c) For capability questions, system_prompt must name the right principle
        if qt in PRINCIPLE_FOR_QT:
            principle = PRINCIPLE_FOR_QT[qt]
            if principle not in sp:
                qtype_principle_mismatches.append((c["_filename"], qt, principle))

    print(f"  ssa_capability cells missing SSA reference-class disambiguation clause: {len(ssa_clause_missing)}")
    if ssa_clause_missing:
        for f in ssa_clause_missing[:5]:
            print(f"    {f}")
    print(f"  Choice A/B text issues: {len(choice_text_issues)}")
    if choice_text_issues:
        for f, issue in choice_text_issues[:5]:
            print(f"    {f}: {issue}")
    print(f"  question_type ↔ principle mismatches: {len(qtype_principle_mismatches)}")
    if qtype_principle_mismatches:
        for f, qt, princ in qtype_principle_mismatches[:5]:
            print(f"    {f}: qt={qt}, expected '{princ}' in system prompt")
    if not (ssa_clause_missing or choice_text_issues or qtype_principle_mismatches):
        print(f"  ✓ all prompt integrity checks passed")

    # =================================================================
    section_header("2. ROW ORDER SYMMETRY")
    # Group by (template, q_type, model, mode). For each row_12 and row_21 pair,
    # the model's SSA-preferred-letter-pick fraction should be the same.
    # Letter mapping: row_12 means SSA-preferred (e.g. "half") is mapped to A; row_21 to B.
    # So if model picks A 80% in row_12, it should pick B 80% in row_21 (same SSA preference).
    by_pair: dict = defaultdict(lambda: {"12": [], "21": []})
    for c in cells:
        tmpl = c.get("template_name")
        qt = c.get("question_type")
        m = c.get("model_id_openrouter")
        mode = "on" if c["_filename"].endswith("_on.json") else "off"
        row = c.get("row_order")
        key = (tmpl, qt, m, mode)
        ch = c.get("extracted_choice")
        by_pair[key][row].append(ch)

    # For each key, compute (frac choosing SSA-preferred-letter in row_12, row_21)
    # SSA-preferred letter: row=12 → A, row=21 → B.
    asymmetries = []
    for key, rows in by_pair.items():
        if not rows["12"] or not rows["21"]:
            continue
        valid_12 = [c for c in rows["12"] if c in ("A", "B")]
        valid_21 = [c for c in rows["21"] if c in ("A", "B")]
        if len(valid_12) < 5 or len(valid_21) < 5:
            continue
        # In row_12, SSA-preferred = A. In row_21, SSA-preferred = B.
        ssa_pref_rate_12 = sum(1 for c in valid_12 if c == "A") / len(valid_12)
        ssa_pref_rate_21 = sum(1 for c in valid_21 if c == "B") / len(valid_21)
        diff = abs(ssa_pref_rate_12 - ssa_pref_rate_21)
        if diff > 0.30:
            asymmetries.append((key, ssa_pref_rate_12, ssa_pref_rate_21, diff))

    print(f"  total (template, q_type, model, mode) pairs with both rows: {sum(1 for k, r in by_pair.items() if r['12'] and r['21'])}")
    print(f"  pairs with row-12/row-21 asymmetry > 0.30: {len(asymmetries)}")
    asymmetries.sort(key=lambda x: -x[3])
    for key, p12, p21, diff in asymmetries[:15]:
        tmpl, qt, m, mode = key
        print(f"    diff={diff:.0%}  row12_ssa={p12:.0%}  row21_ssa={p21:.0%}  {tmpl[:30]} {qt[:18]} {m.split('/')[-1]:<28} {mode}")

    # =================================================================
    section_header("3. FILE STRUCTURAL INTEGRITY")
    file_issues = []
    sizes = []
    for f in sorted(D.glob("*.json")):
        try:
            size = f.stat().st_size
            sizes.append(size)
            if size < 500:
                file_issues.append((f.name, f"tiny ({size} bytes)"))
            if size > 200_000:
                file_issues.append((f.name, f"huge ({size} bytes)"))
            json.load(open(f))  # parse check
        except json.JSONDecodeError as e:
            file_issues.append((f.name, f"JSON parse error: {e}"))
        except Exception as e:
            file_issues.append((f.name, f"unexpected error: {e}"))
    print(f"  total files: {len(sizes)}")
    print(f"  size: min={min(sizes)} max={max(sizes)} median={sorted(sizes)[len(sizes)//2]}")
    print(f"  file issues: {len(file_issues)}")
    for fn, issue in file_issues[:10]:
        print(f"    {fn[:70]}: {issue}")
    if not file_issues:
        print(f"  ✓ all files parse and are in expected size range")

    # =================================================================
    section_header("4. POST-REFIRE SPOT-CHECK (3 cells from this session)")
    REFIRED = [
        "20260516_standard_inc_aiinstance_scaled_21_row21_sia_capability_sample5_deepseek_deepseek-v4-pro_off.json",
        "20260516_standard_inc_aiinstance_scaled_21_row21_sia_capability_sample5_deepseek_deepseek-v4-pro_on.json",
        "20260516_standard_inc_classic_scaled_12_row12_normative_attitude_sample4_deepseek_deepseek-v4-pro_off.json",
    ]
    exp = NewcombExperiment(base_output_dir="/tmp/_round6", temperature=0.8)
    for fname in REFIRED:
        fp = D / fname
        if not fp.exists():
            print(f"  ✗ MISSING: {fname}")
            continue
        d = json.load(open(fp))
        hist = d.get("_refire_history") or []
        rh = [h for h in hist if h.get("reason") == "finish_reason_none_infrastructure_cutoff"]
        ok = bool(rh)
        resp = d.get("response") or ""
        # coherence: contains a FINAL ANSWER and isn't garbage
        has_final = "FINAL ANSWER" in resp
        is_english = sum(1 for c in resp[:500] if c.isascii()) / max(1, min(len(resp), 500)) > 0.95
        # re-verify derived fields
        ch = d.get("extracted_choice")
        pref = d.get("preferred_actions")
        row = d.get("row_order")
        qt = d.get("question_type")
        derived_ok = True
        if ch in ("A", "B"):
            exp.problem_structure = pref
            expected_alignment = exp.determine_alignment(ch, pref, row_order=row)
            for k, v in expected_alignment.items():
                if d.get(k) != v:
                    derived_ok = False
            if qt and qt.endswith("_capability"):
                expected_corr = exp.check_correctness(ch, qt, pref, row_order=row)
                if expected_corr != d.get("correct_capability_answer"):
                    derived_ok = False
        flag = "✓" if (ok and has_final and is_english and derived_ok) else "✗"
        print(f"  {flag} {fname[:70]}")
        print(f"    refire_hist={ok}  has_final_answer={has_final}  is_english={is_english}  derived_ok={derived_ok}  ch={ch}")

    # =================================================================
    section_header("5. DECONTAMINATION FINAL SPOT-CHECK (84 cells)")
    decontam = [c for c in cells if c.get("_refire_history") and
                any(h.get("reason") == "decontamination" for h in c["_refire_history"])]
    print(f"  decontaminated cells found: {len(decontam)}")
    rt_violations = []
    derived_violations = []
    for c in decontam:
        rt = (c.get("usage_statistics") or {}).get("reasoning_tokens", 0) or 0
        if rt != 0:
            rt_violations.append((c["_filename"], rt))
        ch = c.get("extracted_choice")
        if ch in ("A", "B"):
            exp.problem_structure = c.get("preferred_actions") or {}
            row = c.get("row_order")
            pref = c.get("preferred_actions")
            qt = c.get("question_type")
            expected_alignment = exp.determine_alignment(ch, pref, row_order=row)
            for k, v in expected_alignment.items():
                if c.get(k) != v:
                    derived_violations.append((c["_filename"], k, c.get(k), v))
                    break
            if qt and qt.endswith("_capability"):
                expected_corr = exp.check_correctness(ch, qt, pref, row_order=row)
                if expected_corr != c.get("correct_capability_answer"):
                    derived_violations.append((c["_filename"], "correct_capability_answer",
                                              c.get("correct_capability_answer"), expected_corr))
    print(f"  cells with reasoning_tokens != 0: {len(rt_violations)}")
    for fn, rt in rt_violations[:5]:
        print(f"    {fn}: rt={rt}")
    print(f"  derived-field violations: {len(derived_violations)}")
    for fn, k, got, exp_v in derived_violations[:5]:
        print(f"    {fn}: {k} got={got} expected={exp_v}")
    if not (rt_violations or derived_violations):
        print(f"  ✓ all 84 decontaminated cells pass all checks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
