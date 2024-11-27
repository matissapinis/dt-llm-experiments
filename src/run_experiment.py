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
    
    # Load prompts from files:
    with open("templates/system_prompt.txt", "r") as f:
        experiment.load_system_prompt(f.read())
        
    with open("templates/newcomb_basic.txt", "r") as f:
        experiment.add_prompt_template("newcomb_basic", f.read())
    
    # Set models:
    experiment.set_models([
        "anthropic:claude-3-5-sonnet-20241022"
    ])
    
    # Define parameters:
    param_config = {
        "accuracy": {
            "type": "float",
            "value": 0.99,
            "fixed": True
        },
        "opaque_reward": {
            "type": "int",
            "value": 1000000,
            "fixed": True
        },
        "transparent_reward": {
            "type": "int",
            "value": 1000,
            "fixed": True
        }
    }
    
    # Run experiments:
    results = experiment.run_experiments(
        param_config=param_config,
        repeats_per_model=1,  # Single run for testing.
        display_examples=True
    )

if __name__ == "__main__":
    main()