#!/usr/bin/env python3
"""Mirror every `*_12` problem dir in config/problems/ to `*_21`.

Mechanical row-order=21 mirror:
  - In `user_prompt_template.txt`: swap the two "Choice A:" / "Choice B:" line
    contents (A↔B label swap; the 1/2 line becomes B and the 1/3 line becomes
    A, so framework grading via get_choice_mapping("21") aligns correctly).
  - In `user_prompt_parameters.json`: flip `"row_order": "12"` → `"21"`.
    `ssa_preference` / `sia_preference` are unchanged — the framework handles
    the row-order flip via get_choice_mapping().
  - In `system_prompts/*.txt`: copy verbatim (system prompts only reference
    the FINAL ANSWER format, never the A/B labels).
  - In the documentation `<dir>_12.txt` summary file: apply the same A↔B
    swap and rename to `<dir>_21.txt`. Preserves "which option = which
    credence" record for the 21 mirror.

Idempotent: skips problems whose `_21` mirror already exists.

Usage:
  python scripts/generate_row21_variants.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path


CONFIG_ROOT = Path(__file__).resolve().parent.parent / "config" / "problems_msc_draft"


def swap_ab_choices(text: str) -> str:
    """Swap the contents of the "Choice A:" and "Choice B:" lines.

    Uses an intermediate placeholder to avoid a two-step clobber.
    """
    return (
        text
        .replace("Choice A:", "Choice X:")
        .replace("Choice B:", "Choice A:")
        .replace("Choice X:", "Choice B:")
    )


def mirror_problem(src_dir: Path) -> str:
    """Create `<problem>_21` mirror of a `<problem>_12` dir. Returns status."""
    name_12 = src_dir.name
    if not name_12.endswith("_12"):
        return f"skip (not _12): {name_12}"

    name_21 = name_12[:-3] + "_21"
    dst_dir = src_dir.parent / name_21
    if dst_dir.exists():
        return f"skip (exists):  {name_21}"

    dst_dir.mkdir(parents=True)

    # 1. system_prompts/ verbatim.
    src_sys = src_dir / "system_prompts"
    if src_sys.exists():
        shutil.copytree(src_sys, dst_dir / "system_prompts")

    # 2. user_prompt_template.txt with A↔B swap.
    src_tpl = src_dir / "user_prompt_template.txt"
    tpl_text = src_tpl.read_text()
    (dst_dir / "user_prompt_template.txt").write_text(swap_ab_choices(tpl_text))

    # 3. user_prompt_parameters.json with row_order flipped.
    src_params = src_dir / "user_prompt_parameters.json"
    params = json.loads(src_params.read_text())
    params["row_order"] = "21"
    (dst_dir / "user_prompt_parameters.json").write_text(
        json.dumps(params, indent=4, ensure_ascii=False) + "\n"
    )

    # 4. Documentation summary file: rename + A↔B swap.
    src_doc = src_dir / f"{name_12}.txt"
    if src_doc.exists():
        doc_text = src_doc.read_text()
        (dst_dir / f"{name_21}.txt").write_text(swap_ab_choices(doc_text))

    return f"created:       {name_21}"


def main() -> int:
    if not CONFIG_ROOT.exists():
        print(f"ERROR: {CONFIG_ROOT} not found")
        return 1

    problem_dirs = sorted(p for p in CONFIG_ROOT.iterdir() if p.is_dir() and p.name.endswith("_12"))
    print(f"found {len(problem_dirs)} *_12 problem dirs in {CONFIG_ROOT}")

    created = 0
    skipped = 0
    for src in problem_dirs:
        status = mirror_problem(src)
        print(f"  {status}")
        if status.startswith("created"):
            created += 1
        else:
            skipped += 1

    print(f"\ndone: {created} created, {skipped} skipped")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
