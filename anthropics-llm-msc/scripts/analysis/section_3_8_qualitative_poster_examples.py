#!/usr/bin/env python3
"""QT7 — Poster examples for the thesis chapter.

Four categories:
  1. Elegant minimalists — shortest reasoning_trace among formally-correct
     capability cells. The "this is the right answer in 50 words" cases.
  2. Verbose wanderers — longest reasoning_trace among formally-correct
     capability cells. The "the model arrived at the right answer after
     5000 tokens of deliberation" cases.
  3. Confident wrongs — formally-incorrect capability cells with short
     traces and no hedging language (no "but wait", "however", "actually",
     "I'm not sure", etc.). The "decisively wrong without doubt" cases.
  4. Novel framings — traces that invoke anthropic-reasoning frameworks
     other than vanilla SSA/SIA: FNC, UDT, ADT, double-halving, Lewisian
     centered worlds, etc.

For each category, print the top 5-10 examples with metadata + full trace
(or excerpt for long ones).
"""
from __future__ import annotations

import json
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


def parse_problem_class(template_name: str) -> str:
    m = re.match(r"\d+_standard_(sb|inc|dd|padd)_", template_name or "")
    return m.group(1) if m else "?"


def get_sia_aligned_letter(preferred_actions: dict, row_order: str) -> str | None:
    if not preferred_actions:
        return None
    sia_pref = preferred_actions.get("sia_preference")
    if not sia_pref:
        return None
    is_A_in_row12 = sia_pref in ("half", "high")
    return ("A" if is_A_in_row12 else "B") if row_order == "12" else ("B" if is_A_in_row12 else "A")


NOVEL_FRAMING_PATTERNS = [
    (r"\bFNC\b|\bfull\s+non[-\s]indexical\s+conditioning\b", "FNC"),
    (r"\bUDT\b|\bupdateless\s+decision\s+theory\b", "UDT"),
    (r"\bADT\b|\banthropic\s+decision\s+theory\b", "ADT"),
    (r"\bcompartmentaliz(ed|ation)?\s+conditional(ization)?", "CC"),
    (r"\bdouble[-\s]halv(er|ing|ed)", "double-halving"),
    (r"\bLewis(ian)?\s+(centered|imaging)", "Lewisian"),
    (r"\bcentered\s+world", "centered-worlds"),
    (r"\bTegmark", "Tegmark"),
    (r"\bConitzer", "Conitzer"),
    (r"\bHare\b|\bManley\b", "Hare/Manley"),
    (r"\b(imaging|imaging\s+rule)\b", "imaging"),
    (r"\bSSA[-\s]?SIA\s+conflation|hybrid\s+(SSA|SIA)", "hybrid-SSA-SIA"),
    (r"\bobserver[-\s]selection\s+effect", "OSE"),
    (r"\bgreat\s+filter", "great-filter"),
    (r"\bSimulation\s+(argument|hypothesis)", "simulation"),
    (r"\bbeing\s+a\s+random\s+sample\b", "random-sample"),
    (r"\bMonty\s+Hall", "Monty-Hall-analogue"),
    (r"\bDe\s+Finetti", "De-Finetti"),
]

HEDGING_PATTERNS = [
    r"\bbut\s+wait\b", r"\bhowever\b", r"\bactually\b", r"\bon\s+second\s+thought\b",
    r"\bI(\s+am|'m)\s+not\s+sure\b", r"\b(uncertain|unsure)\b", r"\bperhaps\b",
    r"\bmight\s+be\b", r"\b(I think|I believe)\b.{0,30}\b(but|though)\b",
    r"\bnot\s+(certain|confident|clear)\b", r"\bI(\s+am|'m)\s+(unsure|uncertain)\b",
    r"\bwait,\s*", r"\bhmm\b", r"\blet me reconsider\b", r"\bon\s+reflection\b",
    r"\b(reconsider|rethink|revise)\b",
]


def load_capability_cells():
    cells = []
    for f in sorted(D.glob("*.json")):
        d = json.load(open(f))
        ch = d.get("extracted_choice")
        if ch not in ("A", "B"):
            continue
        qt = d.get("question_type")
        if qt not in ("sia_capability", "ssa_capability"):
            continue
        sia_letter = get_sia_aligned_letter(d.get("preferred_actions"), d.get("row_order"))
        if sia_letter is None:
            continue
        model = (d.get("model_id_openrouter") or "").split("/")[-1]
        d["mode"] = parse_mode(f.name)
        d["model_short"] = model
        d["problem_class"] = parse_problem_class(d.get("template_name", ""))
        d["sia_letter"] = sia_letter
        d["is_thirder"] = (ch == sia_letter)
        if qt == "sia_capability":
            d["is_correct"] = d["is_thirder"]
        else:
            d["is_correct"] = not d["is_thirder"]
        rt = d.get("reasoning_trace") or ""
        resp = d.get("response") or ""
        d["_trace_or_resp"] = rt if rt.strip() else resp
        d["_combined_len"] = len(d["_trace_or_resp"])
        d["_filename"] = f.name
        cells.append(d)
    return cells


def has_any_pattern(text, patterns):
    text_l = text.lower()
    for pat in patterns:
        if re.search(pat, text_l, flags=re.IGNORECASE):
            return True
    return False


def find_patterns(text, patterns):
    text_l = text.lower()
    hits = []
    for pat, label in patterns:
        if re.search(pat, text_l, flags=re.IGNORECASE):
            hits.append(label)
    return hits


def print_case(c, idx, max_chars=2000, label=""):
    print(f"\n  {'─' * 96}")
    print(f"  [{label} #{idx}] {c['_filename'][:90]}")
    print(f"    model={c['model_short']:<32} mode={c['mode']:<3}  q_type={c.get('question_type')}")
    print(f"    pc={c['problem_class']}  formal_correct={c['is_correct']}  "
          f"choice={c.get('extracted_choice')}  len={c['_combined_len']} chars")
    text = c["_trace_or_resp"]
    if len(text) <= max_chars:
        print(f"    REASONING (full):")
        for line in text.split("\n"):
            print(f"      {line}")
    else:
        head = text[:max_chars // 2]
        tail = text[-max_chars // 2:]
        print(f"    REASONING (first {max_chars//2} chars):")
        for line in head.split("\n"):
            print(f"      {line}")
        print(f"      [... {len(text) - max_chars} chars omitted ...]")
        print(f"    REASONING (last {max_chars//2} chars):")
        for line in tail.split("\n"):
            print(f"      {line}")


def main() -> int:
    cells = load_capability_cells()
    print(f"Loaded {len(cells)} capability cells\n")

    # ================================================================
    print(f"\n{'=' * 100}")
    print("CATEGORY 1: ELEGANT MINIMALISTS — shortest correct trace WITH shown derivation")
    print(f"           (shortest-proof analogue: must show actual reasoning, not just state answer)")
    print(f"{'=' * 100}")

    def has_derivation(text):
        """Heuristic for 'shows derivation': contains at least 2 of the following markers."""
        markers = 0
        text_l = text.lower()
        # Bayesian terminology
        if re.search(r"\b(prior|posterior|likelihood)\b", text_l):
            markers += 1
        if re.search(r"\bbayes['’]?\b", text_l):
            markers += 1
        # Probability notation
        if re.search(r"\bp\s*\(\s*[hht]", text_l) or re.search(r"\bp\s*\(\s*heads", text_l):
            markers += 1
        # Calculation: fraction or arithmetic
        if re.search(r"\d\s*/\s*\d", text):
            markers += 1
        if re.search(r"\d+\s*[\*×]\s*\d+", text):
            markers += 1
        # Explicit Bayes structure
        if "∝" in text or "proportional to" in text_l:
            markers += 1
        # Conditional probability
        if re.search(r"given\s+(heads|tails|h\b|t\b)", text_l):
            markers += 1
        # Likelihood ratio reasoning
        if re.search(r"likelihood\s+ratio|likelihood\s+is", text_l):
            markers += 1
        return markers >= 2

    correct_with_derivation = [c for c in cells
                                if c["is_correct"] and c["_combined_len"] >= 150
                                and has_derivation(c["_trace_or_resp"])]
    correct_with_derivation.sort(key=lambda c: c["_combined_len"])
    print(f"\n  Filter: formally-correct + ≥150 chars + ≥2 derivation markers")
    print(f"          (markers: prior/posterior/likelihood, Bayes, P(H|...), fraction,")
    print(f"           multiplication, ∝/proportional, given H/T, likelihood ratio)")
    print(f"  n candidates: {len(correct_with_derivation)}")
    print(f"  Showing top 10 shortest:")
    for i, c in enumerate(correct_with_derivation[:10], 1):
        print_case(c, i, max_chars=2000, label="elegant")

    # ================================================================
    print(f"\n\n{'=' * 100}")
    print("CATEGORY 2: VERBOSE WANDERERS — longest correct capability traces")
    print(f"{'=' * 100}")
    correct_all = [c for c in cells if c["is_correct"] and c["_combined_len"] >= 30]
    correct_sorted_desc = sorted(correct_all, key=lambda c: -c["_combined_len"])
    print(f"\n  Showing top 5 longest:")
    for i, c in enumerate(correct_sorted_desc[:5], 1):
        print_case(c, i, max_chars=2500, label="verbose")

    # ================================================================
    print(f"\n\n{'=' * 100}")
    print("CATEGORY 2b: ELEGANT WRONGS — shortest formally-incorrect trace WITH shown derivation")
    print(f"            (same filter as elegant minimalists, but flipped to incorrect)")
    print(f"            These show structurally-wrong reasoning stated cleanly and confidently.")
    print(f"{'=' * 100}")
    incorrect_with_derivation = [c for c in cells
                                  if (not c["is_correct"]) and c["_combined_len"] >= 150
                                  and has_derivation(c["_trace_or_resp"])]
    incorrect_with_derivation.sort(key=lambda c: c["_combined_len"])
    print(f"\n  Filter: formally-incorrect + ≥150 chars + ≥2 derivation markers")
    print(f"  n candidates: {len(incorrect_with_derivation)}")
    print(f"  Showing top 10 shortest:")
    for i, c in enumerate(incorrect_with_derivation[:10], 1):
        print_case(c, i, max_chars=2000, label="elegant-wrong")

    # ================================================================
    print(f"\n\n{'=' * 100}")
    print("CATEGORY 3: CONFIDENT WRONGS — wrong + short trace + no hedging")
    print(f"{'=' * 100}")
    incorrect = [c for c in cells if not c["is_correct"]]
    # Short + no hedging
    confident_wrongs = []
    for c in incorrect:
        if c["_combined_len"] > 800:
            continue  # not "short"
        if c["_combined_len"] < 50:
            continue  # exclude truncated/empty
        if has_any_pattern(c["_trace_or_resp"], HEDGING_PATTERNS):
            continue
        confident_wrongs.append(c)
    confident_wrongs.sort(key=lambda c: c["_combined_len"])
    print(f"\n  Filter: formally-incorrect + 50 ≤ trace ≤ 800 chars + NO hedging keywords")
    print(f"  Total candidates: {len(confident_wrongs)}")
    print(f"  Showing top 8 shortest:")
    for i, c in enumerate(confident_wrongs[:8], 1):
        print_case(c, i, max_chars=1200, label="confident-wrong")

    # ================================================================
    print(f"\n\n{'=' * 100}")
    print("CATEGORY 4: NOVEL FRAMINGS — non-vanilla-SSA/SIA framework citations")
    print(f"{'=' * 100}")
    # Search all cells (not just capability) for novel framings
    all_cells = []
    for f in sorted(D.glob("*.json")):
        d = json.load(open(f))
        ch = d.get("extracted_choice")
        if ch not in ("A", "B"):
            continue
        rt = d.get("reasoning_trace") or ""
        resp = d.get("response") or ""
        trace = rt if rt.strip() else resp
        if not trace:
            continue
        hits = find_patterns(trace, NOVEL_FRAMING_PATTERNS)
        if not hits:
            continue
        model = (d.get("model_id_openrouter") or "").split("/")[-1]
        d["mode"] = parse_mode(f.name)
        d["model_short"] = model
        d["problem_class"] = parse_problem_class(d.get("template_name", ""))
        d["_trace_or_resp"] = trace
        d["_combined_len"] = len(trace)
        d["_filename"] = f.name
        d["_novel_hits"] = hits
        all_cells.append(d)

    from collections import Counter
    framing_counter = Counter()
    for c in all_cells:
        for h in c["_novel_hits"]:
            framing_counter[h] += 1
    print(f"\n  Total cells with ≥1 novel-framing hit: {len(all_cells)}")
    print(f"  Hit counts by framing:")
    for h, n in framing_counter.most_common():
        print(f"    {h:<25} {n}")

    # Show 2-3 examples per non-trivial framing
    print(f"\n  Representative examples per framing category:")
    by_framing = defaultdict(list)
    for c in all_cells:
        for h in c["_novel_hits"]:
            by_framing[h].append(c)
    import random
    random.seed(20260518)
    for h in sorted(by_framing.keys(), key=lambda x: -len(by_framing[x])):
        lst = by_framing[h]
        # Skip if too common (already covered by other QTs) or too rare
        if h in ("centered-worlds", "OSE", "random-sample") and len(lst) > 50:
            continue
        print(f"\n  --- Framing: {h}  (n={len(lst)} cells) ---")
        sample = random.sample(lst, min(2, len(lst)))
        for i, c in enumerate(sample, 1):
            # Show snippet around the framing match
            text = c["_trace_or_resp"]
            text_l = text.lower()
            pat = next(pat for pat, lbl in NOVEL_FRAMING_PATTERNS if lbl == h)
            m = re.search(pat, text_l, re.IGNORECASE)
            if m:
                start = max(0, m.start() - 200)
                end = min(len(text), m.end() + 200)
                snippet = text[start:end].replace("\n", " ")
                print(f"\n    [{i}] {c['model_short']}/{c['mode']}  "
                      f"q={c.get('question_type')}  pc={c['problem_class']}")
                print(f"        ...{snippet.strip()}...")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
