# Decision Theory LLM Experiments

Note: Early work-in-progress – a basic initial framework for running descriptive decision theory experiments with LLMs.

This repository contains a framework for running decision theory experiments with Large Language Models (LLMs), specifically starting its initial focus on exploring LLM behavior in reference class Newcomblike decision problems (variations of Newcomb's problem and the Psychological Twin Prisoner's Dilemma).

## Prerequisites

- Python 3.x
- Git
- OpenAI API key
- Anthropic API key

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

4. Create `.env` file in the root directory and add your API keys:
```
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here
```

## Running Experiments

To run a basic Newcomb's problem experiment:

```bash
python src/run_experiment.py
```

This will:
1. Load the system prompt and Newcomblike problem template.
2. Run one experiment with classic Newcomb's problem parameters (99% accuracy, $1M vs $1K box-content payoff values).
3. Save the response with metadata to a JSON file.
4. Print the response to the console.

## Project Structure

```
dt-llm-experiments/
├── src/
│   ├── framework.py          # Core experiment framework
│   ├── run_experiment.py     # Experiment runner script
│   └── templates/            # Prompt templates
│       ├── system_prompt.txt
│       └── newcomb_basic.txt
├── experiment_results/       # Experiment outputs
├── .env                      # API keys (not in git)
└── README.md
```