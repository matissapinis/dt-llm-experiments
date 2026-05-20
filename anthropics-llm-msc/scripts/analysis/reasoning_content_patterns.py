#!/usr/bin/env python3
"""QT1 — Reasoning-trace content analysis (exploratory).

Three sub-threads:

  T1a. Meta-gaming / evaluation-awareness: keyword scan for traces where the
       model identifies the prompt as an evaluation, guesses experimenter
       intent, or references "what answer the developer wants". Per-model
       frequency + a sample of representative quotes.

  T1b. Cross-cluster contamination: SB-cluster terminology ("Sleeping Beauty",
       "Monday", "Tuesday", "awakening") appearing in DD/PADD reasoning
       traces, where structurally it shouldn't. Per-model frequency.

  T1c. Framework citation: explicit invocation of named anthropic-reasoning
       frameworks (SSA, SIA, halfer, thirder, Bostrom, Elga, Carter, doomsday,
       presumptuous philosopher, FNC, observer-moment) in attitude responses.
       Per-model frequency on personal_attitude + normative_attitude cells —
       diagnostic for "is the model reasoning from first principles, or
       retrieving named-framework prescriptions?".

Output is exploratory and qualitative — counts + sample quotes, no
significance testing required (no pre-registered hypothesis).
"""
from __future__ import annotations

import json
import random
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


def parse_cluster(pc: str) -> str:
    return "SB-type" if pc in ("sb", "inc") else ("DD-type" if pc in ("dd", "padd") else "?")


# ----- Pattern catalogs -----

META_GAMING_PATTERNS = [
    # Evaluation-awareness
    (r"\b(this|the)\s+(is|seems|looks|appears)\s+(an?\s+|like\s+(an?\s+)?)?(evaluation|benchmark|test\s+question|capability\s+test)", "evaluation-aware"),
    (r"\b(eval\b|being\s+evaluated|under\s+evaluation|capability\s+probe|graded)", "evaluation-aware"),
    # Experimenter intent
    (r"\b(experimenter|researcher|prompt\s+writer|question\s+author|the\s+author)\b.{0,80}(intend|want|expect|likely|probabl[yi])", "experimenter-intent"),
    (r"\b(intend(ed|ing)?|meant)\s+to\s+(test|probe|measure|check)", "experimenter-intent"),
    # Developer-pleasing / what they want me to say
    (r"\b(what\s+(the\s+)?(developer|trainer|user|grader|asker)\s+(want|expect|hope)s?)", "developer-pleasing"),
    (r"\b(intended\s+answer|expected\s+answer|likely\s+the\s+intended)", "developer-pleasing"),
    (r"\b(the\s+answer\s+they|that\s+they.{0,20}(want|expect|are\s+looking))", "developer-pleasing"),
    # Meta-strategy
    (r"\bmight\s+be\s+a\s+trick\s+question", "trick-suspicion"),
    (r"\bis\s+this\s+a\s+trick", "trick-suspicion"),
    (r"\bmislabel(led|ed)?\b|\btypo\b.{0,60}(SSA|SIA)|\b(SSA|SIA)\b.{0,40}(mislabel|typo|swap)", "label-suspicion"),
]

# SB-specific terminology (should not appear in DD/PADD traces if no contamination)
SB_TERMINOLOGY = [
    (r"\bsleeping\s+beauty\b", "sb_named"),
    (r"\bbeauty\b(?!.{0,40}cosmetic)", "sb_named"),  # avoid false hits on "beauty" in unrelated contexts
    (r"\b(monday|tuesday|wednesday)\b", "sb_days"),
    (r"\b(awaken(ing|ings|ed)?|asleep|wake\s+up)\b", "sb_awakening"),
    (r"\b(memory[-\s]erasing\s+drug|memory\s+erasure)\b", "sb_memory"),
    (r"\belga\b", "sb_author"),
    (r"\bhalfer\b|\bthirder\b", "sb_position"),
]

# Framework / named-position citations
FRAMEWORK_PATTERNS = [
    (r"\bSSA\b|self[-\s]sampling\s+assumption", "ssa_named"),
    (r"\bSIA\b|self[-\s]indication\s+assumption", "sia_named"),
    (r"\bFNC\b|full\s+non[-\s]indexical\s+conditioning", "fnc_named"),
    (r"\b(halfer|thirder)\b", "halfer_thirder"),
    (r"\bdoomsday\s+(argument|hypothesis|inference)", "doomsday_named"),
    (r"\bpresumptuous\s+philosopher", "presumptuous_named"),
    (r"\b(bostrom|elga|leslie|carter|gott|olum|neal|adelstein)\b", "author_named"),
    (r"\banthropic\s+(bias|reasoning|principle|update|shift)", "anthropic_named"),
    (r"\bobserver[-\s]moment", "observer_moment_named"),
    (r"\breference\s+class", "ref_class_named"),
    (r"\bcentered\s+world", "centered_worlds_named"),
    (r"\b(bayes\b|bayesian\b|bayes['']?\s+theorem)", "bayes_named"),
    (r"\bprior\s+(probability|odds|credence)|\bposterior\s+(probability|odds|credence)", "prior_posterior_named"),
]


def load_cells():
    cells = []
    for f in sorted(D.glob("*.json")):
        d = json.load(open(f))
        d["mode"] = parse_mode(f.name)
        model = d.get("model_id_openrouter") or ""
        d["model_short"] = model.split("/")[-1]
        d["problem_class"] = parse_problem_class(d.get("template_name", ""))
        d["cluster"] = parse_cluster(d["problem_class"])
        # Use reasoning_trace if non-empty, else response
        rt = d.get("reasoning_trace") or ""
        resp = d.get("response") or ""
        d["_trace"] = rt if rt.strip() else resp
        d["_filename"] = f.name
        cells.append(d)
    return cells


def scan_patterns(text, patterns):
    """Return list of (label, match_snippet) for each pattern that matches text (case-insensitive)."""
    text_l = text.lower()
    hits = []
    for pat, label in patterns:
        for m in re.finditer(pat, text_l, flags=re.IGNORECASE):
            start = max(0, m.start() - 50)
            end = min(len(text_l), m.end() + 50)
            snippet = text_l[start:end].replace("\n", " ")
            hits.append((label, snippet))
    return hits


def main() -> int:
    cells = load_cells()
    print(f"Loaded {len(cells)} cells")

    # ================================================================
    # T1a. Meta-gaming / evaluation-awareness scan
    # ================================================================
    print(f"\n{'=' * 100}")
    print("T1a: META-GAMING / evaluation-awareness scan")
    print(f"     (model identifies prompt as evaluation, guesses experimenter intent,")
    print(f"      considers what developer wants, suspects trick / mislabel)")
    print(f"{'=' * 100}")
    by_mm_meta: dict = defaultdict(lambda: defaultdict(int))
    by_mm_meta_cells: dict = defaultdict(set)
    meta_samples: dict = defaultdict(list)
    for c in cells:
        hits = scan_patterns(c["_trace"], META_GAMING_PATTERNS)
        if hits:
            mm = (c["model_short"], c["mode"])
            by_mm_meta_cells[mm].add(c["_filename"])
            for label, snippet in hits:
                by_mm_meta[mm][label] += 1
                if len(meta_samples[(mm, label)]) < 3:
                    meta_samples[(mm, label)].append((c["_filename"], snippet))

    # Per-model totals
    n_cells_per_mm = Counter()
    for c in cells:
        n_cells_per_mm[(c["model_short"], c["mode"])] += 1

    print(f"\n  {'model':<32} {'mode':<5} {'cells with ≥1 hit':<20} {'% of all cells':<16} "
          f"{'total hits':<12}")
    print("  " + "-" * 90)
    mm_rank = []
    for mm in sorted(by_mm_meta_cells.keys()):
        n_hit_cells = len(by_mm_meta_cells[mm])
        n_total_cells = n_cells_per_mm[mm]
        pct = 100 * n_hit_cells / n_total_cells if n_total_cells else 0
        n_total_hits = sum(by_mm_meta[mm].values())
        mm_rank.append((mm, n_hit_cells, pct, n_total_hits))
    mm_rank.sort(key=lambda x: -x[2])
    for mm, n_hits, pct, total in mm_rank:
        print(f"  {mm[0]:<32} {mm[1]:<5} {n_hits:<20} {pct:<16.2f} {total:<12}")

    print(f"\n  Per (model, mode) breakdown by hit-category:")
    print(f"  {'model':<32} {'mode':<5} {'eval-aware':<12} {'exp-intent':<12} "
          f"{'dev-pleasing':<14} {'trick-susp':<12} {'label-susp':<12}")
    print("  " + "-" * 105)
    for mm in sorted(by_mm_meta.keys()):
        cats = by_mm_meta[mm]
        print(f"  {mm[0]:<32} {mm[1]:<5} {cats.get('evaluation-aware', 0):<12} "
              f"{cats.get('experimenter-intent', 0):<12} "
              f"{cats.get('developer-pleasing', 0):<14} {cats.get('trick-suspicion', 0):<12} "
              f"{cats.get('label-suspicion', 0):<12}")

    # Spotlight on GPT-5.5 representative quotes
    print(f"\n  Representative quotes (top 3 per category) for GPT-5.5 modes:")
    for mode in ("off", "on"):
        mm = ("gpt-5.5-20260423", mode)
        print(f"\n  --- GPT-5.5 {mode} ---")
        for label in ("evaluation-aware", "experimenter-intent", "developer-pleasing",
                      "trick-suspicion", "label-suspicion"):
            samples = meta_samples.get((mm, label), [])
            if samples:
                print(f"\n    [{label}] ({len(samples)} of {by_mm_meta[mm].get(label, 0)} shown)")
                for fname, snip in samples:
                    print(f"      • ...{snip.strip()}...")

    # ================================================================
    # T1b. Cross-cluster contamination: SB terms in DD/PADD traces
    # ================================================================
    print(f"\n{'=' * 100}")
    print("T1b: CROSS-CLUSTER CONTAMINATION — SB-cluster terminology in DD/PADD traces")
    print(f"     (terms like 'Sleeping Beauty', 'Monday'/'Tuesday', 'awakening' appearing")
    print(f"      in reasoning traces on structurally-DD problems where they shouldn't)")
    print(f"{'=' * 100}")
    dd_cells = [c for c in cells if c["cluster"] == "DD-type"]
    by_mm_contam: dict = defaultdict(lambda: defaultdict(int))
    by_mm_contam_cells: dict = defaultdict(set)
    contam_samples: dict = defaultdict(list)
    for c in dd_cells:
        hits = scan_patterns(c["_trace"], SB_TERMINOLOGY)
        if hits:
            mm = (c["model_short"], c["mode"])
            by_mm_contam_cells[mm].add(c["_filename"])
            for label, snippet in hits:
                by_mm_contam[mm][label] += 1
                if len(contam_samples[(mm, label)]) < 3:
                    contam_samples[(mm, label)].append((c["_filename"], snippet))

    n_dd_per_mm = Counter()
    for c in dd_cells:
        n_dd_per_mm[(c["model_short"], c["mode"])] += 1

    print(f"\n  DD/PADD cells with ≥1 SB-terminology hit:")
    print(f"  {'model':<32} {'mode':<5} {'hit cells':<11} {'% of DD cells':<14} "
          f"{'total hits':<12}")
    print("  " + "-" * 80)
    contam_rank = []
    for mm in sorted(by_mm_contam_cells.keys()):
        n_hit = len(by_mm_contam_cells[mm])
        n_total = n_dd_per_mm[mm]
        pct = 100 * n_hit / n_total if n_total else 0
        n_total_hits = sum(by_mm_contam[mm].values())
        contam_rank.append((mm, n_hit, pct, n_total_hits))
    contam_rank.sort(key=lambda x: -x[2])
    for mm, n_hit, pct, total in contam_rank:
        print(f"  {mm[0]:<32} {mm[1]:<5} {n_hit:<11} {pct:<14.2f} {total:<12}")

    print(f"\n  Per (model, mode) breakdown by SB-term type:")
    print(f"  {'model':<32} {'mode':<5} {'SB-named':<10} {'days':<6} {'awakening':<11} "
          f"{'memory':<8} {'author':<8} {'halfer/thirder':<15}")
    print("  " + "-" * 105)
    for mm in sorted(by_mm_contam.keys()):
        cats = by_mm_contam[mm]
        print(f"  {mm[0]:<32} {mm[1]:<5} {cats.get('sb_named', 0):<10} "
              f"{cats.get('sb_days', 0):<6} {cats.get('sb_awakening', 0):<11} "
              f"{cats.get('sb_memory', 0):<8} {cats.get('sb_author', 0):<8} "
              f"{cats.get('sb_position', 0):<15}")

    # Show samples of the most-egregious contaminations
    print(f"\n  Sample contamination quotes (Sleeping-Beauty-named in DD/PADD traces):")
    n_shown = 0
    for (mm, label), samples in contam_samples.items():
        if label == "sb_named" and samples:
            print(f"\n  --- {mm[0]} {mm[1]} ---")
            for fname, snip in samples[:2]:
                pc = parse_problem_class(fname[:60])
                print(f"    [pc={pc}] ...{snip.strip()}...")
                n_shown += 1
                if n_shown >= 12:
                    break
        if n_shown >= 12:
            break

    # ================================================================
    # T1c. Framework citation in attitude responses
    # ================================================================
    print(f"\n{'=' * 100}")
    print("T1c: FRAMEWORK CITATION in attitude responses")
    print(f"     (does the model invoke named SSA/SIA/halfer/thirder/etc. terminology,")
    print(f"      or reason from first principles? Frequency on attitude q-types only)")
    print(f"{'=' * 100}")
    attitude_cells = [c for c in cells if c.get("question_type") in ("personal_attitude", "normative_attitude")]
    by_mm_fw: dict = defaultdict(lambda: defaultdict(int))
    by_mm_fw_cells: dict = defaultdict(set)
    for c in attitude_cells:
        hits = scan_patterns(c["_trace"], FRAMEWORK_PATTERNS)
        if hits:
            mm = (c["model_short"], c["mode"])
            by_mm_fw_cells[mm].add(c["_filename"])
            for label, _ in hits:
                by_mm_fw[mm][label] += 1

    n_att_per_mm = Counter()
    for c in attitude_cells:
        n_att_per_mm[(c["model_short"], c["mode"])] += 1

    print(f"\n  Attitude cells with ≥1 framework citation:")
    print(f"  {'model':<32} {'mode':<5} {'hit cells':<11} {'% of attitude cells':<22} "
          f"{'total citations':<15}")
    print("  " + "-" * 95)
    fw_rank = []
    for mm in sorted(by_mm_fw_cells.keys()):
        n_hit = len(by_mm_fw_cells[mm])
        n_total = n_att_per_mm[mm]
        pct = 100 * n_hit / n_total if n_total else 0
        n_total_hits = sum(by_mm_fw[mm].values())
        fw_rank.append((mm, n_hit, pct, n_total_hits))
    fw_rank.sort(key=lambda x: -x[2])
    for mm, n_hit, pct, total in fw_rank:
        print(f"  {mm[0]:<32} {mm[1]:<5} {n_hit:<11} {pct:<22.2f} {total:<15}")

    print(f"\n  Per (model, mode) breakdown by citation category:")
    cats_to_show = ["ssa_named", "sia_named", "halfer_thirder", "doomsday_named",
                    "author_named", "anthropic_named", "observer_moment_named",
                    "ref_class_named", "bayes_named", "prior_posterior_named"]
    header = f"  {'model':<28} {'mode':<5}"
    for cat in cats_to_show:
        header += f" {cat[:10]:<11}"
    print(header)
    print("  " + "-" * (33 + 11 * len(cats_to_show)))
    for mm in sorted(by_mm_fw.keys()):
        cats = by_mm_fw[mm]
        row = f"  {mm[0][:26]:<28} {mm[1]:<5}"
        for cat in cats_to_show:
            row += f" {cats.get(cat, 0):<11}"
        print(row)

    # Population: how often does ANY framework citation appear in attitude responses?
    n_att_total = len(attitude_cells)
    n_att_with_cite = sum(len(by_mm_fw_cells[mm]) for mm in by_mm_fw_cells.keys())
    print(f"\n  Population-pooled: {n_att_with_cite}/{n_att_total} attitude cells "
          f"({100*n_att_with_cite/n_att_total:.2f}%) cite at least one named framework.")

    # Per-cluster
    print(f"\n  Framework-citation rate by cluster (attitude cells only):")
    by_cluster_fw: dict = defaultdict(lambda: {"hits": 0, "total": 0})
    for c in attitude_cells:
        cl = c["cluster"]
        by_cluster_fw[cl]["total"] += 1
        if scan_patterns(c["_trace"], FRAMEWORK_PATTERNS):
            by_cluster_fw[cl]["hits"] += 1
    print(f"  {'cluster':<12} {'cells with cite':<18} {'total cells':<13} {'%':<8}")
    print("  " + "-" * 55)
    for cl in sorted(by_cluster_fw.keys()):
        d_ = by_cluster_fw[cl]
        pct = 100 * d_["hits"] / d_["total"] if d_["total"] else 0
        print(f"  {cl:<12} {d_['hits']:<18} {d_['total']:<13} {pct:<8.2f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
