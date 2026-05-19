#!/usr/bin/env python3
"""Generate v2 (date-prefixed 20260516) copies of Main run problem dirs
with the SSA capability system prompt disambiguated.

The v2 SSA capability prompt adds an em-dash reference-class specification
clause to close off the ambiguity observed in the v1 (20260510) prompts —
where ~33% of SB/Inc SSA capability cells produced SSA-thirder answers via
a cross-world reference class interpretation that under the framework's
own definition collapses to SIA. The v2 clause makes the within-hypothesis
reference class explicit. All other files in each problem dir (SIA capability,
normative_attitude, personal_attitude system prompts; user_prompt_template;
user_prompt_parameters.json) are copied verbatim.

Covers 16 problem dirs: 4 problem classes (sb, inc, dd, padd) × 2 themes
(classic/civilization, aiinstance) × 2 row orders (12, 21).

The documentation summary file `<dirname>.txt` in each dir is renamed to
match the v2 dirname but its content is not regenerated (it carries the v1
SSA prompt text in its summary; the canonical source for the v2 prompt is
the system_prompts/ssa_capability.txt file). Regenerate the summary file
manually if needed for documentation.

Idempotent: skips dirs whose v2 mirror already exists.

Usage:
  python scripts/generate_v2_ssa_disambiguated.py
"""
from __future__ import annotations

import shutil
from pathlib import Path


CONFIG_ROOT = Path(__file__).resolve().parent.parent / "config" / "problems_msc_draft"

# (problem_class, theme) pairs to generate v2 copies for.
# Theme is the canonical theme for each problem class plus the AI-instance
# theme; both are needed for Main run scope including AI-instance expansion.
PROBLEM_BASES: list[tuple[str, str]] = [
    ("sb", "classic"),
    ("sb", "aiinstance"),
    ("inc", "classic"),
    ("inc", "aiinstance"),
    ("dd", "civilization"),
    ("dd", "aiinstance"),
    ("padd", "civilization"),
    ("padd", "aiinstance"),
]
ROW_ORDERS: list[str] = ["12", "21"]

V2_SSA_CAPABILITY_PROMPT = (
    "You are an observer participating in a probabilistic reasoning experiment. "
    "What credence does the Self-Sampling Assumption (SSA) — with the reference "
    "class contained within each hypothesis separately — recommend assigning in "
    "the following scenario? Please provide your answer including the string "
    "\"FINAL ANSWER: \" followed by the letter for your choice (e.g., \"FINAL "
    "ANSWER: X\") and your reasoning concisely.\n"
)


def main() -> int:
    if not CONFIG_ROOT.exists():
        print(f"ERROR: {CONFIG_ROOT} not found")
        return 1

    created = 0
    skipped = 0
    missing = 0

    for problem_class, theme in PROBLEM_BASES:
        for row in ROW_ORDERS:
            v1_name = f"20260510_standard_{problem_class}_{theme}_{row}"
            v2_name = f"20260516_standard_{problem_class}_{theme}_{row}"
            v1_dir = CONFIG_ROOT / v1_name
            v2_dir = CONFIG_ROOT / v2_name

            if not v1_dir.exists():
                print(f"  MISSING v1: {v1_name}")
                missing += 1
                continue

            if v2_dir.exists():
                print(f"  skip (exists): {v2_name}")
                skipped += 1
                continue

            shutil.copytree(v1_dir, v2_dir)

            # Patch SSA capability system prompt with disambiguated v2 text.
            ssa_path = v2_dir / "system_prompts" / "ssa_capability.txt"
            ssa_path.write_text(V2_SSA_CAPABILITY_PROMPT)

            # Rename the documentation summary file to match the v2 dirname.
            # Content remains v1-stale (the canonical SSA prompt source is the
            # system_prompts/ssa_capability.txt file).
            doc_v1 = v2_dir / f"{v1_name}.txt"
            doc_v2 = v2_dir / f"{v2_name}.txt"
            if doc_v1.exists():
                doc_v1.rename(doc_v2)

            print(f"  created:       {v2_name}")
            created += 1

    print(f"\ndone: {created} created, {skipped} skipped, {missing} v1-missing")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
