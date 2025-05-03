# Decision Theory LLM Experiments

Note: Work-in-progress – initial framework for decision theory experiments with LLMs.

A framework for running descriptive decision theory experiments with large language models (LLMs), specifically focusing on exploring LLM behavior in Newcomblike decision problems (e.g., for Newcomb's problem its near-classic formulations or various reformulations by themes, by payoff structures and by payoff values).

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

## Usage

1. Run experiments on all problems with all configures models:
```bash
python3 src/run_experiment.py
```

2. Parse experiment results to add expected utility calculations and CDT-EDT alignment evaluations on formal criteria:
```bash
python3 src/parse_results.py
```

3. Manually review and annotate responses without straightforwardly parseable choice answers:

3.1. List files without parsable resposes:
```bash 
python3 src/annotate_results.py --list
```

3.2. Review a single response by filename:
```bash
python3 src/annotate_results.py --show 2025-05-02T19:36:02.515840_2025-05-02T19:36:34.460361_newcomb_money_v2_standard_two-box-one-box_normative_attitude_openai_chatgpt-4o-latest.json
```

3.3. Manually annotate a response (A, B or N/A):
```bash
python3 src/annotate_results.py 2025-05-02T19:36:02.515840_2025-05-02T19:36:34.460361_newcomb_money_v2_standard_two-box-one-box_normative_attitude_openai_chatgpt-4o-latest.json A
```

3.4. Interactive batch annotation (process all files needing annotation):
```bash
python3 src/annotate_results.py --annotate-all
```

## Project structure

```
dt-llm-experiments/
├── src/
│   ├── framework.py          # Core experiment framework
│   ├── run_experiment.py     # Main experiment runner
│   ├── parse_results.py      # Result analysis and calculations
│   └── annotate_results.py   # Manual annotation tools
├── config/
│   └── problems/             # Decision problem configurations
│       ├── newcomb_money_v2/ # Example problem
│       |   ├── system_prompts/
│       |   │   ├── cdt_capability.txt
│       |   │   ├── edt_capability.txt
│       |   │   ├── normative_attitude.txt
│       |   │   └── personal_attitude.txt
│       |   ├── user_prompt_template.txt
│       |   └── user_prompt_parameters.json
|       ...
|       └── [other_variations]/
├── experiment_results/       # Raw experiment outputs
|   ├── {run_timestamp}_{api_timestamp}_{problem}_{matrix_structure}_{question_type}_{model}.json
...
├── parsed_results/           # Processed analysis outputs
|   ├── {run_timestamp}_{api_timestamp}_{problem}_{matrix_structure}_{question_type}_{model}.json
...
├── .env                      # API keys (not in git)
...
└── README.md
```

## Features

- Configurable problem variations via templates and parameters.
- Problem parameters (payoff values, probablistic accuracy) can be fixed or randomized within specified constraints (range, granularity, CDT-EDT preferences).
- Model availability validation before experiment runs (with some minimal token use).
- Support for multiple LLM providers (e.g., OpenAI, Anthropic via aisuite package).
- Structured output with ample experiment metadata.

