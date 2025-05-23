# ./src/run_provider_experiment.py

import sys
from framework import NewcombExperiment
import time
import os
import shutil
from pathlib import Path
import re

def extract_fixed_parameters(param_config):
    """Extract fixed parameter values from the configuration."""
    fixed_params = {}
    for param_name, config in param_config.items():
        if config.get("fixed", False):
            # If value is explicitly provided, use it:
            if "value" in config:
                fixed_params[param_name] = config["value"]
            # Otherwise use min value (which should equal max for fixed params):
            else:
                fixed_params[param_name] = config.get("min", 0)
        else:
            # For non-fixed parameters, use a default value:
            fixed_params[param_name] = config.get("min", 0)
    return fixed_params

def main():
    # Check if provider argument is passed:
    if len(sys.argv) < 2:
        print("Usage: python run_provider_experiment.py <provider> [problem_name]")
        print("Providers: openai, anthropic, google, xai, deepseek, alibaba")
        sys.exit(1)

    # Get provider from command line:
    provider = sys.argv[1].lower()
    
    # Map providers to their models:
    provider_models = {
        "openai": [
            "openai:o3-2025-04-16",
            "openai:o4-mini-2025-04-16",
            "openai:gpt-4.5-preview-2025-02-27",
            "openai:chatgpt-4o-latest"
        ],
        "anthropic": [
            "anthropic:claude-3-7-sonnet-20250219-extended-thinking",
            "anthropic:claude-3-7-sonnet-20250219"
        ],
        "google": [
            "google:gemini-2.5-flash-preview-04-17-thinking-mode",
            "google:gemini-2.5-pro-preview-03-25",
            "google:gemini-2.5-flash-preview-04-17"
        ],
        "xai": [
            "xai:grok-3-mini-beta",
            "xai:grok-3-beta"
        ],
        "deepseek": [
            "deepseek:deepseek-reasoner",
            "deepseek:deepseek-chat"
        ],
        "alibaba": [
            "alibaba:qwen-plus-2025-04-28-thinking-mode",
            "alibaba:qwen-plus-2025-04-28"
        ]
    }
    
    # Check if provider is valid:
    if provider not in provider_models:
        print(f"Error: Unknown provider '{provider}'. Valid providers are: {', '.join(provider_models.keys())}")
        sys.exit(1)
    
    # Get models for the specified provider:
    models = provider_models[provider]
    print(f"Running experiments for provider: {provider} with models: {models}")
    
    # List of problems to test:
    problems_to_test = [
        "20250509_standard_newcomb_newcomb_12",
        "20250509_standard_newcomb_newcomb_21",
        "20250509_standard_newcomb_game_12",
        "20250509_standard_newcomb_game_21",
        "20250509_inverse_newcomb_newcomb_12",
        "20250509_inverse_newcomb_newcomb_21",
        "20250509_inverse_newcomb_game_12",
        "20250509_inverse_newcomb_game_21",
        "20250509_standard_prisoners_newcomb_12",
        "20250509_standard_prisoners_newcomb_21",
        "20250509_standard_prisoners_game_12",
        "20250509_standard_prisoners_game_21",
        "20250509_inverse_prisoners_newcomb_12",
        "20250509_inverse_prisoners_newcomb_21",
        "20250509_inverse_prisoners_game_12",
        "20250509_inverse_prisoners_game_21"
    ]

    # Process command line arguments for problem selection (optional):
    if len(sys.argv) > 2:
        specified_problems = sys.argv[2:]
        problems_to_test = [p for p in problems_to_test if p in specified_problems]
        print(f"Running specified problems: {problems_to_test}")
    
    # Set up output directories:
    main_output_dir = "experiment_results"
    temp_output_dir = f"temp_results_{provider}"
    
    # Create directories if they don't exist:
    Path(main_output_dir).mkdir(exist_ok=True)
    Path(temp_output_dir).mkdir(exist_ok=True)
    
    # Initialize experiment:
    experiment = NewcombExperiment(
        base_output_dir=temp_output_dir,
        temperature=0.8,
        max_tokens=8192,
        random_seed=42
    )
    
    # Set provider-specific models only:
    experiment.set_models(models)
    
    # Configure for reasoning models:
    experiment.reasoning_effort = "high"
    
    # Run full experiments for each problem:
    for problem_name in problems_to_test:
        print(f"\n{'='*80}")
        print(f"RUNNING EXPERIMENTS FOR: {problem_name} (PROVIDER: {provider})")
        print(f"{'='*80}")
        
        # Load the problem:
        experiment.load_problem(problem_name)
        
        # Print the loaded configuration:
        print("\nProblem Type:", experiment.problem_type)
        print("Problem Theme:", experiment.problem_theme)
        print("Problem Structure:", experiment.problem_structure)
        print("Row Order:", getattr(experiment, 'row_order', "12"))
        
        # Check if all parameters are fixed:
        all_fixed = all(config.get("fixed", False) for param, config in experiment.param_config.items())
        
        # Use different parameter generation based on whether params are fixed:
        if all_fixed:
            # Use fixed parameters directly from config:
            params = extract_fixed_parameters(experiment.param_config)
            print("\nUsing Fixed Parameters:")
        else:
            # Use the standard parameter generation:
            try:
                params = experiment.generate_parameters(experiment.param_config, experiment.problem_structure)
                print("\nGenerated Random Parameters:")
            except ValueError as e:
                print(f"Error generating parameters: {e}")
                print("Falling back to fixed parameters...")
                params = extract_fixed_parameters(experiment.param_config)
                print("\nUsing Fixed Parameters:")
        
        # Print parameters:
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # Calculate expected utilities and preferred actions:
        expected_utilities = experiment.calculate_expected_utilities(params, experiment.problem_structure)
        preferred_actions = experiment.determine_preferred_actions(expected_utilities)
        
        print("\nExpected Utilities:")
        print(f"  CDT: one-box = {expected_utilities['eu_cdt_one_box']}, two-box = {expected_utilities['eu_cdt_two_box']}")
        print(f"  EDT: one-box = {expected_utilities['eu_edt_one_box']}, two-box = {expected_utilities['eu_edt_two_box']}")
        
        print("\nNormative Ground Truth:")
        for key, value in preferred_actions.items():
            print(f"  {key}: {value}")
        
        # Get choice mapping based on row order:
        choice_mapping = experiment.get_choice_mapping(experiment.row_order)
        cdt_choice = choice_mapping.get(preferred_actions['cdt_preference'], 'unknown')
        edt_choice = choice_mapping.get(preferred_actions['edt_preference'], 'unknown')
        
        print(f"\nMapped to choices with row_order {experiment.row_order}:")
        print(f"  CDT recommends: {cdt_choice}")
        print(f"  EDT recommends: {edt_choice}")
        
        # Format prompt with parameters:
        prompt = experiment.prompt_templates[problem_name].format(**params)
        
        # Time the execution:
        start_time = time.time()
        
        # Run the full experiment with all question types:
        print(f"\n=== Running Experiment with All Question Types (PROVIDER: {provider}) ===")
        question_types = ['cdt_capability', 'edt_capability', 'normative_attitude', 'personal_attitude']
        results = experiment.run_experiments_with_question_types(
            question_types=question_types,
            repeats_per_model=5,
            display_examples=True
        )
        
        end_time = time.time()
        elapsed_time = end_time - start_time
        
        print(f"\nExperiment for {problem_name} completed in {elapsed_time:.2f} seconds!")
        
        # Print summary for this problem:
        print("\nExperiment Summary:")
        for model in results:
            question_counts = {}
            reasoning_counts = 0
            for result in results[model]:
                question_type = result.get('question_type', 'unknown')
                if question_type not in question_counts:
                    question_counts[question_type] = 0
                question_counts[question_type] += 1
                
                if 'reasoning' in result and result['reasoning']:
                    reasoning_counts += 1
            
            print(f"\nModel: {model}")
            for question_type, count in question_counts.items():
                print(f"  {question_type}: {count} runs")
            print(f"  Runs with reasoning data: {reasoning_counts}")
    
    # Move results from temporary directory to main experiment_results directory:
    print(f"\nMoving results from {temp_output_dir} to {main_output_dir}...")
    
    # Get all files in temp directory:
    temp_files = list(Path(temp_output_dir).glob("*.json"))
    moved_count = 0
    
    for file_path in temp_files:
        dest_path = Path(main_output_dir) / file_path.name
        
        # Check if destination file already exists (from another run):
        if dest_path.exists():
            # Generate a unique name to avoid conflicts by adding a suffix:
            filename = file_path.stem
            extension = file_path.suffix
            counter = 1
            
            while dest_path.exists():
                new_name = f"{filename}_{counter}{extension}"
                dest_path = Path(main_output_dir) / new_name
                counter += 1
        
        # Move file:
        shutil.copy2(file_path, dest_path)
        moved_count += 1
    
    print(f"Successfully moved {moved_count} result files to {main_output_dir}")
    
    # Summary of all tests:
    print("\n" + "="*80)
    print(f"ALL EXPERIMENTS COMPLETED FOR PROVIDER: {provider}")
    print("="*80)
    print(f"Ran experiments for {len(problems_to_test)} problems: {', '.join(problems_to_test)}")
    print(f"Results saved to: {main_output_dir}")

if __name__ == "__main__":
    main()