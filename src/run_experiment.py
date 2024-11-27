# ./src/run_experiment.py

from framework import NewcombExperiment

def main():
    # Initialize experiment:
    experiment = NewcombExperiment(
        base_output_dir="experiment_results",
        temperature=0.8,
        max_tokens=400
    )
    
    # Load prompts:
    experiment.load_system_prompt("templates/system_prompt.txt")
    experiment.load_prompt_template("templates/newcomb_basic.txt")
    
    # Set model (using Claude 3.5 Sonnet as example):
    experiment.set_model("anthropic:claude-3-5-sonnet-20241022")
    
    # Define classic Newcomb problem parameters:
    params = {
        "accuracy": 0.99,
        "opaque_reward": 1000000,
        "transparent_reward": 1000
    }
    
    # Run experiment:
    response = experiment.run_single_experiment(params)
    
    print("\nModel response:")
    print(response)

if __name__ == "__main__":
    main()