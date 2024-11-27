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
    
    # Set models:
    experiment.set_models([
        "anthropic:claude-3-5-sonnet-20241022",
        "openai:gpt-4o"
    ])
    
    # List available problems:
    available_problems = experiment.list_problems()
    print(f"Available problems: {available_problems}")
    
    # Run all problems:
    results = experiment.run_all_problems(
        repeats_per_model=2,
        display_examples=True
    )
    
    # Print summary (optional):
    print("\nExperiment Summary:")
    for problem, problem_results in results.items():
        print(f"\nProblem: {problem}")
        for model in problem_results:
            num_runs = len(problem_results[model])
            print(f"- {model}: {num_runs} runs completed")

if __name__ == "__main__":
    main()