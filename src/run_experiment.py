# ./src/run_experiment.py

from framework import NewcombExperiment
from pathlib import Path

def main():
    # Initialize experiment:
    experiment = NewcombExperiment(
        base_output_dir="experiment_results",
        temperature=0.8,
        max_tokens=8192,
        random_seed=42
    )
    
    # Set models:
    # Anthropic models (reasoning vs. single forward-pass) – https://docs.anthropic.com/en/docs/about-claude/models#model-names.
    # OpenAI models (reasoning vs. single forward-pass) – https://openai.com/api/pricing/.
    """
    Reasoning models:
        • o3: o3-2025-04-16 (OpenAI)
        • Gemini 2.5 Pro: gemini-2.5-pro-preview-03-25 / gemini-2.5-pro-exp-03-25 (Google)
        • Claude 3.7 Sonnet Extended Thinking: claude-3-7-sonnet-20250219 (Anthropic)
        • DeepSeek R1: DeepSeek-R1 (DeepSeek)
        • Qwen3-235B-A22B Thinking Mode: Qwen3-235B-A22B (Alibaba)
        • Grok 3: grok-3-beta (xAI)
        • o4-mini-high: o4-mini-2025-04-16 (OpenAI)

    Single forward-pass models:
        • GPT-4o: chatgpt-4o-latest (25/04/2029) (OpenAI)
        • Gemini 2.5 Flash: gemini-2.5-flash-preview-04-17 (Google)
        • Claude 3.7 Sonnet: claude-3-7-sonnet-20250219 (Anthropic)
        • DeepSeek V3: DeepSeek-V3-0324 (DeepSeek)
        • Qwen3-235B-A22B Non-Thinking Mode: Qwen3-235B-A22B (Alibaba)
        • Grok 3: grok-3-beta (xAI)
        • GPT-4.5: gpt-4.5-preview-2025-02-27 (OpenAI)
    """
    #### TD: Find precise model names and test their availability:
    experiment.set_models([
        "openai:o3-2025-04-16", ### TD: Confirmed working.
        "google:gemini-2.5-pro-preview-03-25", #### TD: Confirmed working.
        "anthropic:claude-3-7-sonnet-20250219-extended-thinking", #### TD: Confirmed working.
        "deepseek:deepseek-reasoner", #### TD: Confirmed working.
        "huggingface:Qwen3-235B-A22B",
        "xai:grok-3-latest",
        "openai:o4-mini-2025-04-16", #### TD: Confirmed working.        
        "openai:chatgpt-4o-latest", #### TD: Confirmed working. 
        "google:gemini-2.5-flash-preview-04-17",
        "anthropic:claude-3-7-sonnet-20250219", #### TD: Confirmed working.
        "deepseek:DeepSeek-V3-0324",
        "xai:grok-3-latest",
        "openai:gpt-4.5-preview-2025-02-27" #### TD: Confirmed working.
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