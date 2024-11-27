# ./src/run_experiment.py

from framework import NewcombExperiment
from pathlib import Path

def main():
    # Initialize experiment:
    experiment = NewcombExperiment(
        base_output_dir="experiment_results",
        temperature=0.8,
        max_tokens=400,
        random_seed=42
    )
    
    # Load problem configuration (includes system prompt):
    experiment.load_problem("newcomb_marriage")  # Or "newcomb_money" and so on.
    
    # Set models:
    experiment.set_models([
        "anthropic:claude-3-5-sonnet-20241022"
    ])
    
    # Run experiments:
    results = experiment.run_experiments(
        param_config=experiment.param_config,
        repeats_per_model=1,
        display_examples=True
    )

if __name__ == "__main__":
    main()