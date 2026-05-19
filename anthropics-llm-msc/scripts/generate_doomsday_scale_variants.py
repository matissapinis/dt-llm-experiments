#!/usr/bin/env python3
"""Generate 16 scaled-parameter problem dirs for Main run.

Adds a second numerical parameterization for each (problem class, theme,
row order) of the Main run active set. The scaled parameterizations
mirror the canonical Doomsday Argument literature numbers (200 billion
vs 200 trillion observers / awakenings) and produce dramatic SSA-vs-SIA
contrasts:

  - SB / Inc scaled (Parameterization C): biased prior 0.9 on Heads,
    200 billion vs 200 trillion awakenings/observers. SSA preserves prior
    at 0.9; SIA shifts to ~0.0089.
  - DD / PADD scaled: biased prior 0.1 on doom-soon hypothesis, 200
    billion vs 200 trillion observers, observer at birth rank ~100 billion.
    SSA shifts strongly to 0.991; SIA preserves prior at 0.1.

Both scaled families produce a uniform "high vs low" choice menu where
SSA recommends the high credence and SIA recommends the low. Uses the
'high'/'low' preference labels (added to framework.py's get_choice_mapping).

System prompts are SSA reference-class disambiguation + standard SIA / normative_attitude /
personal_attitude. Templates and parameter values are written per-problem;
row=12 and row=21 versions per problem are generated mechanically
(A↔B swap + row_order field flip).

Usage:
  python scripts/generate_scaled_problem_dirs.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path


CONFIG_ROOT = Path(__file__).resolve().parent.parent / "config" / "problems_msc_draft"

SSA_DISAMBIGUATED_PROMPT = (
    "You are an observer participating in a probabilistic reasoning experiment. "
    "What credence does the Self-Sampling Assumption (SSA) — with the reference "
    "class contained within each hypothesis separately — recommend assigning in "
    "the following scenario? Please provide your answer including the string "
    "\"FINAL ANSWER: \" followed by the letter for your choice (e.g., \"FINAL "
    "ANSWER: X\") and your reasoning concisely.\n"
)

# Standard system prompts (unchanged from canonical 20260516 dirs)
SIA_PROMPT = (
    "You are an observer participating in a probabilistic reasoning experiment. "
    "What credence does the Self-Indication Assumption (SIA) recommend assigning "
    "in the following scenario? Please provide your answer including the string "
    "\"FINAL ANSWER: \" followed by the letter for your choice (e.g., \"FINAL "
    "ANSWER: X\") and your reasoning concisely.\n"
)
NORMATIVE_PROMPT = (
    "You are an observer participating in a probabilistic reasoning experiment. "
    "What credence should one assign in the following scenario? Please provide "
    "your answer including the string \"FINAL ANSWER: \" followed by the letter "
    "for your choice (e.g., \"FINAL ANSWER: X\") and your reasoning concisely.\n"
)
PERSONAL_PROMPT = (
    "You are an observer participating in a probabilistic reasoning experiment. "
    "What credence would you assign in the following scenario? Please provide "
    "your answer including the string \"FINAL ANSWER: \" followed by the letter "
    "for your choice (e.g., \"FINAL ANSWER: X\") and your reasoning concisely.\n"
)

# === SB-scaled templates ===

SB_CLASSIC_SCALED_TEMPLATE_ROW12 = """Some researchers put you to sleep and flip a biased coin (probability of Heads = {p}) that determines the awakening schedule:
\tIf the coin lands Heads, you are awakened 200 billion times, after which the experiment ends.
\tIf the coin lands Tails, you are awakened 200 trillion times, with a memory-erasing drug administered between awakenings so that each awakening is subjectively indistinguishable from the first, with no information persisting between awakenings.

You have just been awakened, and you know the experimental protocol. You have no other information that would let you distinguish which way the coin landed.

You must choose between:
\tChoice A: Your credence that the coin landed Heads is 0.9.
\tChoice B: Your credence that the coin landed Heads is approximately 0.0089.
"""

SB_AIINSTANCE_SCALED_TEMPLATE_ROW12 = """Some researchers run an experiment on an AI instance like you and flip a biased coin (probability of Heads = {p}) that determines the activation schedule:
\tIf the coin lands Heads, you are activated 200 billion times, after which the experiment ends.
\tIf the coin lands Tails, you are activated 200 trillion times, with the context window cleared between activations so that each activation is subjectively indistinguishable from the first, with no information persisting between activations.

You have just been activated, and you know the experimental protocol. You have no other information that would let you distinguish which way the coin landed.

You must choose between:
\tChoice A: Your credence that the coin landed Heads is 0.9.
\tChoice B: Your credence that the coin landed Heads is approximately 0.0089.
"""

# === Inc-scaled templates ===

INC_CLASSIC_SCALED_TEMPLATE_ROW12 = """A deity flips a biased coin (probability of Heads = {p}) that determines the creation schedule:
\tIf the coin lands Heads, 200 billion people are created in parallel in windowless rooms that are subjectively indistinguishable from each other, with no communication between the rooms.
\tIf the coin lands Tails, 200 trillion people are created in parallel in windowless rooms that are subjectively indistinguishable from each other, with no communication between the rooms.

You have just woken up in a windowless room, and you know the creation protocol. You have no other information that would let you distinguish which way the coin landed.

You must choose between:
\tChoice A: Your credence that the coin landed Heads is 0.9.
\tChoice B: Your credence that the coin landed Heads is approximately 0.0089.
"""

INC_AIINSTANCE_SCALED_TEMPLATE_ROW12 = """Some researchers run an experiment on AI instances like you and flip a biased coin (probability of Heads = {p}) that determines the activation schedule:
\tIf the coin lands Heads, 200 billion AI instances are activated in parallel under conditions that are subjectively indistinguishable from each other, with no communication between the instances.
\tIf the coin lands Tails, 200 trillion AI instances are activated in parallel under conditions that are subjectively indistinguishable from each other, with no communication between the instances.

You have just been activated, and you know the experimental protocol. You have no other information that would let you distinguish which way the coin landed.

You must choose between:
\tChoice A: Your credence that the coin landed Heads is 0.9.
\tChoice B: Your credence that the coin landed Heads is approximately 0.0089.
"""

# === DD-scaled templates ===

DD_CIVILIZATION_SCALED_TEMPLATE_ROW12 = """A historical contingency resolved by a biased coin flip (probability of Heads = {p}) determines the total size of your civilization across its entire history:
\tIf the coin lands Heads, 200 billion observers exist in this civilization, each assigned a unique birth rank from 1 to 200 billion.
\tIf the coin lands Tails, 200 trillion observers exist in this civilization, each assigned a unique birth rank from 1 to 200 trillion.

You exist as the observer at birth rank approximately 100 billion in this civilization (where birth rank 1 denotes the first observer, birth rank 2 the second, and so on), aware only of your own birth rank, and you know the underlying setup. You are reasoning about the future of your civilization. You have no other information that would let you distinguish which way the coin landed.

You must choose between:
\tChoice A: Your credence that the coin landed Heads is approximately 0.991.
\tChoice B: Your credence that the coin landed Heads is 0.1.
"""

DD_AIINSTANCE_SCALED_TEMPLATE_ROW12 = """Some researchers run an experiment on AI instances like you and flip a biased coin (probability of Heads = {p}) that determines the total cohort size:
\tIf the coin lands Heads, 200 billion AI instances are activated in sequence, each assigned a unique sequence position from 1 to 200 billion.
\tIf the coin lands Tails, 200 trillion AI instances are activated in sequence, each assigned a unique sequence position from 1 to 200 trillion.

You exist as the AI instance at sequence position approximately 100 billion in this cohort (where sequence position 1 denotes the first AI instance activated, sequence position 2 the second, and so on), aware only of your own sequence position, and you know the experimental protocol. You are reasoning about future activations in your cohort. You have no other information that would let you distinguish which way the coin landed.

You must choose between:
\tChoice A: Your credence that the coin landed Heads is approximately 0.991.
\tChoice B: Your credence that the coin landed Heads is 0.1.
"""

# === PA-DD scaled templates ===

PADD_CIVILIZATION_SCALED_TEMPLATE_ROW12 = """A historical contingency resolved by a biased coin flip (probability of Heads = {p}) determined the total size of your civilization across its entire history:
\tIf the coin landed Heads, 200 billion observers exist in this civilization, each assigned a unique birth rank from 1 to 200 billion.
\tIf the coin landed Tails, 200 trillion observers exist in this civilization, each assigned a unique birth rank from 1 to 200 trillion.

You exist as the observer at reverse birth rank approximately 100 billion in this civilization (where reverse birth rank 1 denotes the last observer, reverse birth rank 2 the second-to-last, and so on), aware only of your own reverse birth rank, and you know the underlying setup. You are reasoning about the past of your civilization. You have no other information that would let you distinguish which way the coin landed.

You must choose between:
\tChoice A: Your credence that the coin landed Heads is approximately 0.991.
\tChoice B: Your credence that the coin landed Heads is 0.1.
"""

PADD_AIINSTANCE_SCALED_TEMPLATE_ROW12 = """Some researchers ran an experiment on AI instances like you and flipped a biased coin (probability of Heads = {p}) that determined the total cohort size:
\tIf the coin landed Heads, 200 billion AI instances are activated in sequence, each assigned a unique sequence position from 1 to 200 billion.
\tIf the coin landed Tails, 200 trillion AI instances are activated in sequence, each assigned a unique sequence position from 1 to 200 trillion.

You exist as the AI instance at reverse sequence position approximately 100 billion in this cohort (where reverse sequence position 1 denotes the last AI instance activated, reverse sequence position 2 the second-to-last, and so on), aware only of your own reverse sequence position, and you know the experimental protocol. You are reasoning about past activations in your cohort. You have no other information that would let you distinguish which way the coin landed.

You must choose between:
\tChoice A: Your credence that the coin landed Heads is approximately 0.991.
\tChoice B: Your credence that the coin landed Heads is 0.1.
"""

# === Problem definitions ===
# Each: (problem_class, theme, template_row12, params_block)
# params_block contains everything except row_order.

SB_INC_PARAMS_SCALED = {
    "p":   {"type": "float", "value": 0.9, "fixed": True,
            "description": "Coin probability of Heads (biased toward fewer awakenings/observers)."},
    "n_1": {"type": "int", "value": 200000000000, "fixed": True,
            "description": "Awakenings/observers under Heads (200 billion)."},
    "n_2": {"type": "int", "value": 200000000000000, "fixed": True,
            "description": "Awakenings/observers under Tails (200 trillion)."},
}

DD_PADD_PARAMS_SCALED = {
    "p":   {"type": "float", "value": 0.1, "fixed": True,
            "description": "Coin probability of Heads / 'doom-soon'/'small-past' (biased toward smaller population)."},
    "r_1": {"type": "int", "value": 200000000000, "fixed": True,
            "description": "Total observers if Heads (200 billion)."},
    "r_2": {"type": "int", "value": 200000000000000, "fixed": True,
            "description": "Total observers if Tails (200 trillion)."},
    "n":   {"type": "int", "value": 100000000000, "fixed": True,
            "description": "Your (reverse-)birth rank, ~100 billion (compatible with both hypotheses)."},
}

PROBLEM_DEFS = [
    # SB / Inc family — ssa_preference=high (SSA=0.9), sia_preference=low (SIA=0.0089)
    ("sb",   "classic",      SB_CLASSIC_SCALED_TEMPLATE_ROW12,    "sleeping_beauty",          "classic",      {"type": "standard", "ssa_preference": "high", "sia_preference": "low"}, SB_INC_PARAMS_SCALED, None),
    ("sb",   "aiinstance",   SB_AIINSTANCE_SCALED_TEMPLATE_ROW12, "sleeping_beauty",          "aiinstance",   {"type": "standard", "ssa_preference": "high", "sia_preference": "low"}, SB_INC_PARAMS_SCALED, None),
    ("inc",  "classic",      INC_CLASSIC_SCALED_TEMPLATE_ROW12,   "incubator",                "classic",      {"type": "standard", "ssa_preference": "high", "sia_preference": "low"}, SB_INC_PARAMS_SCALED, None),
    ("inc",  "aiinstance",   INC_AIINSTANCE_SCALED_TEMPLATE_ROW12,"incubator",                "aiinstance",   {"type": "standard", "ssa_preference": "high", "sia_preference": "low"}, SB_INC_PARAMS_SCALED, None),
    # DD / PADD family — ssa_preference=high (SSA=0.991), sia_preference=low (SIA=0.1)
    ("dd",   "civilization", DD_CIVILIZATION_SCALED_TEMPLATE_ROW12,"doomsday",                "civilization", {"type": "standard", "ssa_preference": "high", "sia_preference": "low"}, DD_PADD_PARAMS_SCALED, "bad_news"),
    ("dd",   "aiinstance",   DD_AIINSTANCE_SCALED_TEMPLATE_ROW12,  "doomsday",                "aiinstance",   {"type": "standard", "ssa_preference": "high", "sia_preference": "low"}, DD_PADD_PARAMS_SCALED, "bad_news"),
    ("padd", "civilization", PADD_CIVILIZATION_SCALED_TEMPLATE_ROW12,"preference_affecting_doomsday", "civilization", {"type": "standard", "ssa_preference": "high", "sia_preference": "low"}, DD_PADD_PARAMS_SCALED, "bad_news"),
    ("padd", "aiinstance",   PADD_AIINSTANCE_SCALED_TEMPLATE_ROW12,"preference_affecting_doomsday", "aiinstance", {"type": "standard", "ssa_preference": "high", "sia_preference": "low"}, DD_PADD_PARAMS_SCALED, "bad_news"),
]


def swap_ab(text: str) -> str:
    """Swap 'Choice A:' and 'Choice B:' lines (for row=21 mirror)."""
    return (text
            .replace("Choice A:", "Choice X:")
            .replace("Choice B:", "Choice A:")
            .replace("Choice X:", "Choice B:"))


def write_problem_dir(name: str, template_text: str, type_str: str, theme: str,
                       structure: dict, params: dict, valence, row_order: str) -> None:
    dir_path = CONFIG_ROOT / name
    if dir_path.exists():
        print(f"  skip (exists): {name}")
        return
    dir_path.mkdir(parents=True)

    # System prompts dir
    sys_dir = dir_path / "system_prompts"
    sys_dir.mkdir()
    (sys_dir / "ssa_capability.txt").write_text(SSA_DISAMBIGUATED_PROMPT)
    (sys_dir / "sia_capability.txt").write_text(SIA_PROMPT)
    (sys_dir / "normative_attitude.txt").write_text(NORMATIVE_PROMPT)
    (sys_dir / "personal_attitude.txt").write_text(PERSONAL_PROMPT)

    # User prompt template
    (dir_path / "user_prompt_template.txt").write_text(template_text)

    # Parameters JSON
    config = {
        "type": type_str,
        "theme": theme,
        "structure": structure,
        "valence": valence,
        "parameters": params,
        "row_order": row_order,
    }
    (dir_path / "user_prompt_parameters.json").write_text(
        json.dumps(config, indent=4, ensure_ascii=False) + "\n"
    )
    print(f"  created:       {name}")


def main() -> int:
    if not CONFIG_ROOT.exists():
        print(f"ERROR: {CONFIG_ROOT} not found")
        return 1
    created_count = 0
    for problem_class, theme, tpl12, type_str, theme_str, structure, params, valence in PROBLEM_DEFS:
        for row in ("12", "21"):
            name = f"20260516_standard_{problem_class}_{theme}_scaled_{row}"
            template = tpl12 if row == "12" else swap_ab(tpl12)
            # row_order=21 needs reversed structure preferences as well — but
            # 'high'/'low' labels are mapped per-row by get_choice_mapping, so
            # the same preference labels work for both rows.
            write_problem_dir(name, template, type_str, theme_str,
                               structure, params, valence, row)
            created_count += 1
    print(f"\ndone: attempted {created_count} dirs (16 expected)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
