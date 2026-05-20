# anthropics-llm-msc

Code and data for the Master's thesis *Epistemic Behavior of Large Language Models in Anthropic Reasoning Problems* (Matīss Apinis, University of Latvia, May 2026).

The thesis experimentally investigates the behavior of 12 leading large language model (LLM) configurations on four anthropic reasoning problem classes (*Sleeping Beauty*, *Incubator*, standard *Doomsday* argument, and "past" *Doomsday* argument), with the *Self-Sampling Assumption* (SSA) and *Self-Indication Assumption* (SIA) as the normative principles whose recommendations the models are asked to either compute (capability questions) or endorse options aligned with them (attitude questions). The complete experiment comprises 13 824 model responses across the design parameters (4 problem classes × 2 thematic framings × 2 numerical parameterizations × 2 choice orderings × 4 question types × 12 configurations of model and reasoning mode × 9 repetitions).

## Repository layout

- `src/` — experiment framework, main experiment driver, model availability checks, runtime support (rate limiting, retries, concurrency).
- `scripts/` — pipeline scripts that produce the cleaned, validated corpus from the raw responses (parser, validation checks etc.).
- `scripts/analysis/` — analysis scripts that produce quantitative claims in the thesis (named by thesis sections, e.g. `section_3_4_cluster_pluralism.py`, `section_3_7_design_parameter_effects.py`; otherwise exploratory analyses not cited in the thesis).
- `scripts/exploratory/` — diagnostic scripts used during dataset construction and validation; not required for replication.
- `config/problems/` — two representative problem directories as examples (`20260516_standard_sb_classic_12` and `20260516_standard_dd_civilization_12`).
- `config.zip` — all 32 thesis problem variants plus a concatenated readable text file; password-protected (please reach out if interested).
- `data.zip` — all 13 824 raw response JSON files from the main experimental run, ≈35 MB; password-protected (please reach out if interested).

## Prerequisites

- Python 3
- Git
- OpenRouter API key (used by the main experiment driver)

## Setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install python-dotenv requests aisuite pandas openai tqdm
echo "OPENROUTER_API_KEY=your_openrouter_key_here" > .env
unzip -P ... config.zip
unzip -P ... data.zip
```

## Replicating the thesis results

If you have the unpacked `experiment_results/main_run_20260516/` (from `data.zip`) or have otherwise compatible data, quantitative claims in the thesis are generally reproducible by running the matching `scripts/analysis/section_X_Y_*.py` and reading its stdout. For example:

```bash
python3 scripts/analysis/section_3_2_3_3_capability_accuracy.py
python3 scripts/analysis/section_3_4_cluster_pluralism.py
python3 scripts/analysis/section_3_7_design_parameter_effects.py
```

To get parse-quality and correctness fields with repeated parsing:

```bash
python3 scripts/two_stage_parser.py
python3 scripts/rederive_correctness_flags.py
```

To collect raw responses from scratch (requires API key and credits):

```bash
python3 src/run_main_experiment.py
```
