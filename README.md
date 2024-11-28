# Decision Theory LLM Experiments

Note: Early work-in-progress – a basic initial framework for decision theory experiments with LLMs.

This repository contains a framework for running descriptive decision theory experiments with large language models (LLMs), specifically focusing on exploring LLM behavior in Newcomblike decision problems (e.g., for Newcomb's problem its near-classic formulations or various reformulations by themes, by payoff structures and by payoff values).

## Prerequisites

- Python 3.x
- Git
- LLM provider API keys (e.g., OpenAI's, Anthropic's)

## Setup

1. Clone the repository:
```bash
git clone https://github.com/matissapinis/dt-llm-experiments.git
cd dt-llm-experiments
```

2. Create and activate virtual environment:
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install required packages:
```bash
pip install 'aisuite[all]' pandas python-dotenv
```

4. Create `.env` file in the root directory and add your API keys, e.g.:
```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

5. Optional: Try the added IPYNB notebook with example usage by importing it into Google Colab.

## Running experiments and usage

To run experiments:

### 1. Local environment
```bash
python src/run_experiment.py
```

### 2. Google Colab notebook
An IPYNB example notebook is provided for running experiments in Google Colaboratory with the same-supporting functionality.

These will:
1. Validate availability of configured LLM models.
2. Load all available decision problem configurations.
3. Run experiments for each model on each problem the chosen number of times.
4. Save responses with full metadata to JSON files.
5. Display example responses and completion summary.

## Project structure

```
dt-llm-experiments/
├── src/
│   ├── framework.py          # Core experiment framework
│   └── run_experiment.py     # Local experiment runner script
├── config/
│   └── problems/             # Decision problem configurations
│       ├── newcomb_money/
│       │   ├── system_prompt.txt
│       │   ├── user_prompt_template.txt
│       │   └── user_prompt_parameters.json
│       ├── newcomb_marriage/
│       │   ├── system_prompt.txt
│       │   ├── user_prompt_template.txt
│       │   └── user_prompt_parameters.json
│       └── [other_variations]/
├── experiment_results/       # Experiment outputs
├── .env                      # API keys (not in git)
├── dt-llm-experiments-example-notebook.ipynb     # Example Google Colab notebook
...
└── README.md
```

## Features

- Support for multiple LLM providers (e.g., OpenAI, Anthropic via aisuite package).
- Configurable problem variations via templates and parameters.
- Model availability validation before experiment runs (with some minimal token use).
- Structured output with full experiment metadata.
- Compatible with both local environments and Google Colab.

## Development status

Early prototype with working implementation of:
- Framework for running batched experiments.
- Model validation and error handling.
- Configurable problem variations.
- Result saving with metadata.
- Local environment and Google Colab compatibility.

## To-do

### 1. Problem structure standardization
- [ ] Implement clear structure for decision problem variations:
  - 1. Problem type (e.g., Newcomb's Problem, Psychological Twin Prisoner's Dilemma).
  - 2. Problem theme (e.g., money-box, QALY-marriage).
  - 3. Problem structure:
    - 1. Standard (EDT → one-box, CDT → two-box)
    - 2. Inverse (EDT → two-box, CDT → one-box)
    - 3. EDT-indifferent (EDT → indifferent, CDT → two-box)
    - 4. CDT-indifferent (EDT → one-box, CDT → indifferent)
    - 5. Fully indifferent (control) (EDT → indifferent, CDT → indifferent)
    - 6. Other combinatorial choices ({DT → preferred action})

### 2. Parameter generation
- [ ] Add support for randomized parameter generation:
  - Theme-appropriate intuitive value ranges and granularities (e.g., money-box in multiple-incrementable $100K-$10M range, QALY-marriage in 1-incrementable 1-70 QALY range).

### 3. Other potential improvements
- [ ] Response parsing and quantitative or qualitative analysis assistance (to search for insights):
  - Payoff-specific patterns.
  - Theme-specific patterns.
  - Structure-specific patterns.
  - Problem-specific patterns.
  - Consistency within models.
  - Comparison across models at frontier between providers.
  - Comparison across models on scaling between frontier and legacy.
  - Decision theory alignment.
  - Notable reasoning patterns.
  - ...
- [ ] Improve immediate readability of preview responses (to ease quick-impressionable viewing).
- [ ] Flexibility for running experiments on subset of decision problem configurations (to switch via code, not file management).
- [ ] Include in decision configuration preferred normative ground-truth – DT-calculated utilities, reasoning (to clarify normative-descriptive comparison).
- [ ] Any needed support for local or Colab results visualization (to generate graphs).
- [ ] Automated tests for framework components (to test for errors).
- [ ] Support for batch experiment scheduling (to handle API rate limits, quotas).
- [ ] Experiment cost estimation (to avoid compute budgeting issues).
- [ ] Clean up example Google Colab notebook to remove unintended private decision problem artifacts (to avoid leaking latent evaluation information).
