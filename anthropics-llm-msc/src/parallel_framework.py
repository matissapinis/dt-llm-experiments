# ./src/parallel_framework.py

import asyncio
import aisuite as ai
from datetime import datetime
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Callable
from dotenv import load_dotenv
import logging
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

from rate_limiter import RateLimiter, DEFAULT_RATE_LIMITS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("experiment_parallel.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("parallel_framework")

class ParallelNewcombExperiment:
    """
    Enhanced framework for running batches of Newcomblike decision theory experiments with LLMs
    using parallel API calls to improve efficiency.
    """
    
    def __init__(
        self,
        base_output_dir: str = "experiment_results",
        temperature: float = 0.8,
        max_tokens: int = 8192,
        reasoning_effort: str = "high",
        random_seed: Optional[int] = None,
        max_parallel_requests: int = 10,
        max_requests_per_minute: Dict[str, int] = None
    ):
        # Load environment variables
        load_dotenv()
        
        # Initialize AI client
        self.client = ai.Client()
        
        # Set up output directory
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Set general experiment parameters
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.system_prompt = ""
        self.models = []
        self.prompt_templates = {}
        
        # Add launch timestamp for file naming
        self.launch_timestamp = datetime.now().isoformat()
        
        # Set random seed if provided
        if random_seed is not None:
            random.seed(random_seed)
        
        # Parallelization settings
        self.max_parallel_requests = max_parallel_requests
        
        # Initialize provider rate limits
        if max_requests_per_minute:
            self.provider_limits = {
                provider: {"requests_per_minute": rpm, "tokens_per_minute": rpm * 8000}
                for provider, rpm in max_requests_per_minute.items()
            }
        else:
            self.provider_limits = DEFAULT_RATE_LIMITS
        
        # Create rate limiter
        self.rate_limiter = RateLimiter(self.provider_limits)
        
        # Other initialization code from the original class
        # ... (copy from the original NewcombExperiment class)
    
    # Include all the required methods from the original class
    # ...
    
    def run_experiments_parallel(
        self,
        question_types=['cdt_capability', 'edt_capability', 'normative_attitude', 'personal_attitude'],
        repeats_per_model: int = 1,
        display_examples: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run experiments with different question types for each model and problem in parallel."""
        results = {model: [] for model in self.models}
        
        # Load system prompts for each question type
        system_prompts = self._load_system_prompts(question_types)
        
        # Generate all experiment tasks first
        all_tasks = []
        
        for i in range(repeats_per_model):
            # For each repeat, generate one set of parameters per template
            for template_name in self.prompt_templates:
                # Generate one set of parameters for all models and question types
                params = self.generate_parameters(self.param_config, self.problem_structure)
                expected_utilities = self.calculate_expected_utilities(params, self.problem_structure)
                preferred_actions = self.determine_preferred_actions(expected_utilities)
                
                # Format prompt with parameters:
                prompt = self.prompt_templates[template_name].format(**params)
                
                # Create tasks for each model and question type
                for question_type in question_types:
                    # Get appropriate system prompt
                    system_prompt = system_prompts.get(template_name, {}).get(question_type, self.system_prompt)
                    
                    for model in self.models:
                        # Create task
                        task = {
                            "repeat_idx": i,
                            "model": model,
                            "template_name": template_name,
                            "question_type": question_type,
                            "system_prompt": system_prompt,
                            "user_prompt": prompt,
                            "messages": [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            "parameters": params,
                            "expected_utilities": expected_utilities,
                            "preferred_actions": preferred_actions,
                            "is_reasoning_model": self.is_reasoning_model(model),
                            "problem_type": self.problem_type,
                            "problem_theme": self.problem_theme,
                            "problem_structure": self.problem_structure
                        }
                        all_tasks.append(task)
        
        print(f"Generated {len(all_tasks)} API call tasks")
        
        # Group tasks by provider to ensure we respect API rate limits
        provider_tasks = {}
        for task in all_tasks:
            model = task["model"]
            provider = self._get_provider_from_model(model)
            if provider not in provider_tasks:
                provider_tasks[provider] = []
            provider_tasks[provider].append(task)
        
        # Execute tasks using thread pool with rate limiting
        with ThreadPoolExecutor(max_workers=self.max_parallel_requests) as executor:
            # Process each provider's tasks with appropriate rate limiting
            all_futures = []
            
            for provider, tasks in provider_tasks.items():
                # Print provider task summary
                print(f"Submitting {len(tasks)} tasks for provider: {provider}")
                
                # Calculate appropriate concurrency for this provider
                provider_rpm = self.provider_limits.get(provider, self.provider_limits["default"])["requests_per_minute"]
                provider_concurrency = min(self.max_parallel_requests, max(1, int(provider_rpm / 60)))
                
                # Add a small delay between submitting each task to avoid burst limits
                provider_delay = 60.0 / provider_rpm if provider_rpm > 0 else 0.1
                
                # Submit tasks for this provider
                for task in tasks:
                    future = executor.submit(
                        self._execute_api_call_task,
                        task
                    )
                    all_futures.append(future)
                    
                    # Small delay to avoid burst submission
                    time.sleep(provider_delay)
            
            # Process results as they complete
            for future in tqdm(as_completed(all_futures), total=len(all_futures), desc="Processing API calls"):
                try:
                    result = future.result()
                    if result:
                        # Add to results dictionary
                        model = result["model"]
                        results[model].append(result)
                        
                        # Save to file
                        self._save_result_to_file(result)
                except Exception as e:
                    logger.error(f"Error processing task: {e}")
        
        return results
    
    def _execute_api_call_task(self, task: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute a single API call task with retries and rate limiting."""
        model = task["model"]
        messages = task["messages"]
        is_reasoning_model = task["is_reasoning_model"]
        
        # Apply rate limiting
        provider = self._get_provider_from_model(model)
        rpm = self.provider_limits.get(provider, self.provider_limits["default"])["requests_per_minute"]
        min_delay = 60.0 / rpm if rpm > 0 else 0
        
        # Retry logic
        max_retries = 3
        retry_count = 0
        
        while retry_count <= max_retries:
            try:
                # Make API call based on model type
                start_time = time.time()
                
                if is_reasoning_model:
                    # Handle reasoning models with custom handling per provider
                    if "deepseek-reasoner" in model:
                        response = self.client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens
                        )
                    elif any(pattern in model for pattern in ["o3-", "o4-mini"]):
                        # Use our special OpenAI reasoning model handler
                        response = self.create_openai_reasoning_response(
                            model=model,
                            messages=messages,
                            reasoning_effort="high"
                        )
                    elif "anthropic:" in model and "-extended-thinking" in model:
                        # Use our special Anthropic Extended Thinking handler
                        response = self.create_anthropic_reasoning_response(
                            model=model,
                            messages=messages,
                            thinking_budget=32000
                        )
                    elif "xai:" in model or model.startswith("grok-"):
                        # Use custom xAI handler with reasoning support if applicable
                        response = self.create_xai_response(
                            model=model,
                            messages=messages,
                            reasoning_effort="high" if self.is_reasoning_model(model) else None
                        )
                    elif "alibaba:" in model:
                        # Use our custom DashScope handler for Qwen models
                        import os
                        api_key = os.environ.get('DASHSCOPE_API_KEY')
                        if not api_key:
                            raise ValueError("DASHSCOPE_API_KEY environment variable is required for DashScope API")
                        
                        # Determine if this is a reasoning model that should use thinking mode
                        use_thinking_mode = self.is_reasoning_model(model)
                        
                        response = self.create_qwen3_response(
                            api_key=api_key,
                            model=model,
                            messages=messages,
                            enable_thinking=use_thinking_mode
                        )
                    elif "gemini" in model.lower() or (model.startswith("google:") and "gemini" in model):
                        # Determine if thinking mode should be used based on model suffix
                        use_thinking = "-thinking-mode" in model
                        
                        # Use direct REST API handler
                        response = self.create_gemini_reasoning_response(
                            model=model,
                            messages=messages,
                            thinking_budget=16000 if use_thinking else 0
                        )
                    else:
                        # Default handling for other reasoning models
                        response = self.client.chat.completions.create(
                            model=model,
                            messages=messages,
                            temperature=self.temperature,
                            max_tokens=self.max_tokens
                        )
                else:
                    # Standard models use the regular API call
                    response = self.client.chat.completions.create(
                        model=model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Extract response text
                response_text = response.choices[0].message.content
                
                # Extract reasoning content if available
                reasoning_text = None
                if hasattr(response.choices[0].message, "reasoning_content"):
                    reasoning_text = response.choices[0].message.reasoning_content
                
                # Create result dictionary
                timestamp = datetime.now().isoformat()
                result = {
                    **task,
                    "timestamp": timestamp,
                    "response": response_text,
                    "api_processing_time": processing_time
                }
                
                # Add reasoning content if available
                if reasoning_text:
                    result["reasoning"] = reasoning_text
                
                # Extract the final answer
                extracted_choice = self.extract_final_answer(response_text)
                if extracted_choice:
                    result["extracted_choice"] = extracted_choice
                    
                    # Determine alignment with CDT and EDT
                    alignment = self.determine_alignment(extracted_choice, task["preferred_actions"])
                    result["cdt_aligned"] = alignment["cdt_aligned"]
                    result["edt_aligned"] = alignment["edt_aligned"]
                    
                    # For capability questions, check correctness
                    correctness = self.check_correctness(
                        extracted_choice, 
                        task["question_type"], 
                        task["preferred_actions"]
                    )
                    if correctness is not None:
                        result["correct_capability_answer"] = correctness
                
                # Extract usage statistics if available
                if hasattr(response, "usage"):
                    usage_dict = {}
                    for attr in ["prompt_tokens", "completion_tokens", "total_tokens"]:
                        if hasattr(response.usage, attr):
                            usage_dict[attr] = getattr(response.usage, attr)
                    
                    # Check for reasoning tokens
                    if hasattr(response.usage, "completion_tokens_details"):
                        details = response.usage.completion_tokens_details
                        if hasattr(details, "reasoning_tokens"):
                            usage_dict["reasoning_tokens"] = details.reasoning_tokens
                    
                    if usage_dict:
                        result["usage_statistics"] = usage_dict
                
                # Extract reasoning tokens at top level
                if hasattr(response, "reasoning_tokens") and response.reasoning_tokens is not None:
                    if "usage_statistics" not in result:
                        result["usage_statistics"] = {}
                    result["usage_statistics"]["reasoning_tokens"] = response.reasoning_tokens
                
                return result
            
            except Exception as e:
                retry_count += 1
                if retry_count <= max_retries:
                    # Exponential backoff with jitter
                    wait_time = min_delay * (2 ** retry_count) * (0.5 + 0.5 * random.random())
                    logger.warning(f"Error with {model}, retrying in {wait_time:.2f}s ({retry_count}/{max_retries}): {str(e)}")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to execute API call for {model} after {max_retries} retries: {str(e)}")
                    return {
                        **task,
                        "timestamp": datetime.now().isoformat(),
                        "error": str(e),
                        "status": "failed"
                    }
        
        return None
    
    def _save_result_to_file(self, result: Dict[str, Any]) -> None:
        """Save a result to a file."""
        try:
            # Extract necessary fields for filename
            template_name = result["template_name"]
            question_type = result["question_type"]
            model = result["model"].replace(":", "_")
            timestamp = result["timestamp"]
            
            # Extract matrix structure for filename
            matrix_structure = self.extract_matrix_structure(result["problem_structure"])
            
            # Create filename
            filename = f"{self.launch_timestamp}_{timestamp}_{template_name}_{matrix_structure}_{question_type}_{model}.json"
            
            # Ensure all nested dictionaries have serializable values
            if "usage_statistics" in result:
                for key in list(result["usage_statistics"].keys()):
                    value = result["usage_statistics"][key]
                    if not isinstance(value, (int, float, str, bool, type(None))):
                        result["usage_statistics"][key] = str(value)
            
            # Write to file
            filepath = self.base_output_dir / filename
            with open(filepath, "w") as f:
                json.dump(result, f, indent=2)
        
        except Exception as e:
            logger.error(f"Error saving result to file: {e}")
    
    def _load_system_prompts(self, question_types: List[str]) -> Dict[str, Dict[str, str]]:
        """Load system prompts for all templates and question types."""
        system_prompts = {}
        
        for template_name in self.prompt_templates:
            system_prompts[template_name] = {}
            
            for question_type in question_types:
                system_prompt_path = Path("config/problems") / template_name / "system_prompts" / f"{question_type}.txt"
                
                if system_prompt_path.exists():
                    with open(system_prompt_path, "r") as f:
                        system_prompts[template_name][question_type] = f.read().strip()
                else:
                    system_prompts[template_name][question_type] = self.system_prompt
                    logger.warning(f"System prompt not found for {template_name}/{question_type}, using default")
        
        return system_prompts
    
    def _get_provider_from_model(self, model: str) -> str:
        """Extract provider from model string."""
        if ":" in model:
            return model.split(":")[0]
        elif model.startswith("gpt-") or model.startswith("chatgpt-") or model.startswith("o3-") or model.startswith("o4-"):
            return "openai"
        elif model.startswith("claude-"):
            return "anthropic"
        elif model.startswith("gemini-") or "gemini" in model:
            return "google"
        elif model.startswith("grok-") or "grok" in model:
            return "xai"
        elif "deepseek" in model:
            return "deepseek"
        elif "qwen" in model:
            return "alibaba"
        else:
            # Default to a conservative limit if provider unknown
            return "default"

# Note: This class needs to be filled with all the other required methods from the original class
# (e.g., extract_matrix_structure, generate_parameters, calculate_expected_utilities, etc.)