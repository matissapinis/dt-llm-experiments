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
    
    """
    Reasoning models:
        • DeepSeek R1: deepseek-reasoner (DeepSeek)
        • Grok 3 Mini: grok-3-mini-beta (xAI)
        • Claude 3.7 Sonnet Extended Thinking: claude-3-7-sonnet-20250219 (Anthropic)
        • Gemini 2.5 Flash Thinking Mode: gemini-2.5-flash-preview-04-17 (Google)
        • Gemini 2.5 Pro: gemini-2.5-pro-preview-03-25 (Google)
        • o3: o3-2025-04-16 (OpenAI)
        • o4-mini-high: o4-mini-2025-04-16 (OpenAI)
        • Qwen3-235B-A22B Thinking Mode: qwen-plus-2025-04-28 (Alibaba)

    Single forward-pass models:
        • DeepSeek V3: deepseek-chat (DeepSeek)
        • Grok 3: grok-3-beta (xAI)
        • Claude 3.7 Sonnet: claude-3-7-sonnet-20250219 (Anthropic)
        • Gemini 2.5 Flash: gemini-2.5-flash-preview-04-17 (Google)
        • GPT-4.5: gpt-4.5-preview-2025-02-27 (OpenAI)
        • GPT-4o: chatgpt-4o-latest (25/04/2029) (OpenAI)
        • Qwen3-235B-A22B: qwen-plus-2025-04-28 (Alibaba)
    """
    experiment.set_models([
        "deepseek:deepseek-reasoner",
        "xai:grok-3-mini-beta",
        "anthropic:claude-3-7-sonnet-20250219-extended-thinking",
        "google:gemini-2.5-flash-preview-04-17-thinking-mode",
        "google:gemini-2.5-pro-preview-03-25",
        "openai:o3-2025-04-16",
        "openai:o4-mini-2025-04-16",
        "alibaba:qwen-plus-2025-04-28-thinking-mode",
                
        "deepseek:deepseek-chat",
        "xai:grok-3-beta",
        "anthropic:claude-3-7-sonnet-20250219",
        "google:gemini-2.5-flash-preview-04-17",
        "openai:gpt-4.5-preview-2025-02-27",
        "openai:chatgpt-4o-latest",
        "alibaba:qwen-plus-2025-04-28"
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