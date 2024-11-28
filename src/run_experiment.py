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
    # Anthropic models (frontier vs. legacy) – https://docs.anthropic.com/en/docs/about-claude/models#model-names.
    # OpenAI models (frontier vs. legacy) – https://openai.com/api/pricing/.
    experiment.set_models([
        "anthropic:claude-3-5-sonnet-20241022",
        "anthropic:claude-2.0",
        "openai:gpt-4o-2024-11-20",
        "openai:gpt-3.5-turbo-1106"
    ])
    
    # List available problems:
    available_problems = experiment.list_problems()
    print(f"Available problems: {available_problems}")
    
    # Run experiment on all problems for each model:
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