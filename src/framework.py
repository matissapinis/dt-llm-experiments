# ./src/framework.py

import aisuite as ai
from datetime import datetime
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Union, Optional
from dotenv import load_dotenv

class NewcombExperiment:
    """Framework for running batches of Newcomblike decision theory experiments with LLMs."""

    def __init__(
        self,
        base_output_dir: str = "experiment_results",
        temperature: float = 0.8,
        max_tokens: int = 8192,
        reasoning_effort: str = "high",
        random_seed: Optional[int] = None
    ):
        # Load environment variables:
        load_dotenv()
        
        # Initialize AI client:
        self.client = ai.Client()
        
        # Set up output directory:
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Set general experiment parameters:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.system_prompt = ""
        self.models = []
        self.prompt_templates = {}
        
        # Add launch timestamp for file naming:
        self.launch_timestamp = datetime.now().isoformat()
        
        # Set random seed if provided:
        if random_seed is not None:
            random.seed(random_seed)

    def extract_matrix_structure(self, structure):
        """Extract a short string representing the matrix structure."""
        if not structure:
            return "unknown"
        
        structure_type = structure.get('type', '')
        cdt_pref = structure.get('cdt_preference', '')
        edt_pref = structure.get('edt_preference', '')
        
        # Create compact representation of structure:
        return f"{structure_type}_{cdt_pref}-{edt_pref}"

    def validate_all_models(self, models: List[str]) -> List[str]:
        """Validate all models are available before proceeding with any experiments:"""
        print("\nValidating model availability...")
        available_models = []
        unavailable_models = []
        
        for model in models:
            try:
                # Use specialized validation for Gemini models (due to direct REST API call implementation):
                if "gemini" in model or (model.startswith("google:") and "gemini" in model):
                    self.validate_gemini_model(model)
                    available_models.append(model)
                    print(f"✓ Available: {model}")
                    continue

                # Use specialized validation for xAI models (due to direct REST API call implementation):
                if "xai:" in model or model.startswith("grok-"):
                    self.validate_xai_model(model)
                    available_models.append(model)
                    print(f"✓ Available: {model}")
                    continue

                # Check if this is an OpenAI reasoning model:
                if self.is_reasoning_model(model) and any(pattern in model for pattern in ["o3-", "o4-mini"]):
                    # Use direct OpenAI client validation for reasoning models:
                    import openai
                    client = openai.OpenAI()
                    
                    # Extract the model name without the provider prefix:
                    model_name = model.split(":")[-1] if ":" in model else model
                    
                    # Use minimal content for validation:
                    response = client.responses.create(
                        model=model_name,
                        reasoning={"effort": "low"},
                        input=[
                            {"role": "user", "content": "."}
                        ],
                        max_output_tokens=16 # Minimal token usage.
                    )
                    available_models.append(model)
                    print(f"✓ Available: {model}")
                    continue
                # Check if this is an Anthropic reasoning model:
                if self.is_reasoning_model(model) and "anthropic:" in model and "-extended" in model:
                    # For Claude with Extended Thinking, validate with Anthropic API:
                    import anthropic
                    client = anthropic.Anthropic()
                    
                    # Extract base model name (without -extended-thinking suffix):
                    model_name = model.split(":")[-1].replace("-extended-thinking", "")
                    
                    # Try minimal content with Extended Thinking:
                    response = client.beta.messages.create(
                        model=model_name,
                        max_tokens=1024 + 1, # Minimal token usage.
                        thinking={
                            "type": "enabled",
                            "budget_tokens": 1024 # Minimal token usage.
                        },
                        messages=[{"role": "user", "content": "."}]
                    )
                    available_models.append(model)
                    print(f"✓ Available: {model}")
                    continue

                # Standard models or other reasoning models use regular validation:
                self.client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": "."},
                        {"role": "user", "content": "."}
                    ],
                    max_tokens=1,
                    temperature=0
                )
                
                available_models.append(model)
                print(f"✓ Available: {model}")
            except Exception as e:
                unavailable_models.append((model, str(e)))
                print(f"✗ Unavailable: {model}")
                print(f"  Error: {str(e)}")
        
        if unavailable_models:
            error_msg = "The following models are unavailable:\n"
            for model, error in unavailable_models:
                error_msg += f"- {model}: {error}\n"
            raise RuntimeError(error_msg)
        
        return available_models

    def validate_gemini_model(self, model: str) -> bool:
        """Validate a Gemini model is available by making a minimal REST API call."""
        # Extract the model name without the provider prefix:
        model_name = model.split(":")[-1] if ":" in model else model
        
        # Import necessary libraries:
        import os
        import requests
        
        # Check for the presence of GOOGLE_API_KEY environment variable:
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini API")
        
        # Build a minimal API request payload:
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        
        minimal_payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": "."}] # Minimal content.
                }
            ],
            "generationConfig": {
                "maxOutputTokens": 1 # Request minimal output.
            }
        }
        
        try:
            # Make a minimal API request:
            response = requests.post(
                f"{api_url}?key={api_key}",
                headers={"Content-Type": "application/json"},
                json=minimal_payload,
                timeout=10 # Add a reasonable timeout.
            )
            
            # Check if request was successful:
            response.raise_for_status()
            
            # If we got here, validation succeeded:
            return True
            
        except Exception as e:
            # Log detailed error information:
            print(f"Gemini validation error: {str(e)}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Response error: {e.response.text}")
            
            # Validation failed:
            raise e

    def validate_xai_model(self, model: str) -> bool:
        """Validate an xAI model is available by making a minimal REST API call."""
        # Extract the model name without the provider prefix:
        model_name = model.split(":")[-1] if ":" in model else model
        
        # Import necessary libraries:
        import os
        import openai
        
        # Check for the presence of XAI_API_KEY environment variable:
        api_key = os.environ.get('XAI_API_KEY')
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable is required for xAI API")
        
        try:
            # Set up the OpenAI client with xAI base URL:
            client = openai.OpenAI(
                base_url="https://api.x.ai/v1",
                api_key=api_key
            )
            
            # Prepare minimal arguments:
            kwargs = {
                "model": model_name,
                "messages": [{"role": "user", "content": "."}],
                "max_tokens": 1
            }
            
            # Add reasoning parameter if this is a reasoning model:
            if self.is_reasoning_model(model):
                kwargs["reasoning_effort"] = "low"
            
            # Make a minimal API request:
            client.chat.completions.create(**kwargs)
            
            # If we got here, validation succeeded:
            return True
            
        except Exception as e:
            # Log detailed error information:
            print(f"xAI validation error: {str(e)}")
            
            # Validation failed:
            raise e

    def set_models(self, models: List[str]) -> None:
        """Set list of models to use in experiments with strict validation:"""
        self.models = self.validate_all_models(models)
        print(f"\nAll {len(self.models)} requested models are available.")
    
    def load_system_prompt(self, prompt: str) -> None:
        """Load system prompt directly from string:"""
        self.system_prompt = prompt.strip()

    def load_problem(self, problem_name: str) -> None:
        """Load a problem's configuration, template and parameters from config directory:"""
        problem_dir = Path("config/problems") / problem_name
        
        # Clear previous templates:
        self.prompt_templates = {}
        
        # Load system prompt if it exists in either location:
        system_prompt_path = problem_dir / "system_prompt.txt"
        system_prompts_folder = problem_dir / "system_prompts"
        
        if system_prompt_path.exists():
            with open(system_prompt_path, "r") as f:
                self.load_system_prompt(f.read())
        elif system_prompts_folder.exists() and any(system_prompts_folder.glob("*.txt")):
            # Use the first system prompt in the subfolder as default:
            system_prompt_files = list(system_prompts_folder.glob("*.txt"))
            if system_prompt_files:
                with open(system_prompt_files[0], "r") as f:
                    self.load_system_prompt(f.read())
                print(f"Using {system_prompt_files[0].name} as default system prompt for {problem_name}")
        else:
            # Set a default system prompt if no files exist:
            self.system_prompt = "You're a participant in a decision theory experiment."
            print(f"Note: No system prompts found for {problem_name}, using default.")
        
        # Load template:
        with open(problem_dir / "user_prompt_template.txt", "r") as f:
            self.add_prompt_template(problem_name, f.read())
        
        # Load parameters and structure configuration:
        with open(problem_dir / "user_prompt_parameters.json", "r") as f:
            config_data = json.load(f)
            
            # Extract structure configuration if it exists:
            if "type" in config_data and "theme" in config_data and "structure" in config_data:
                self.problem_type = config_data.get("type")
                self.problem_theme = config_data.get("theme")
                self.problem_structure = config_data.get("structure")
                
                # Extract parameter config:
                if "parameters" in config_data:
                    self.param_config = config_data["parameters"]
                else:
                    self.param_config = {}
            else:
                # Legacy format:
                self.problem_type = None
                self.problem_theme = None 
                self.problem_structure = None
                self.param_config = config_data

    def list_problems(self) -> List[str]:
        """List all available problems in config directory:"""
        problems_dir = Path("config/problems")
        return [p.name for p in problems_dir.iterdir() if p.is_dir() and 
                all((p / f).exists() for f in [
                    "user_prompt_template.txt",
                    "user_prompt_parameters.json"
                ])]

    def run_all_problems(
        self,
        repeats_per_model: int = 1,
        display_examples: bool = True
    ) -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
        """Run experiments for all available problems:"""
        all_results = {}
        
        for problem_name in self.list_problems():
            print(f"\nRunning problem: {problem_name}")
            self.load_problem(problem_name)
            
            results = self.run_experiments(
                param_config=self.param_config,
                repeats_per_model=repeats_per_model,
                display_examples=display_examples
            )
            
            all_results[problem_name] = results
        
        return all_results

    def add_prompt_template(self, name: str, template: str) -> None:
        """Add a prompt template directly from string:"""
        self.prompt_templates[name] = template.strip()

    def validate_structure_constraints(self, params, structure):
        """Validate that generated parameters satisfy decision theory constraints."""
        # Extract parameters for the payoff matrix:
        x = params.get('x', 0)
        y = params.get('y', 0)
        c = params.get('c', 0)
        p = params.get('p', 0.99)

        # Derive z and w from x, y, and c (transparent box):
        z = x + c
        w = y + c
        
        # Check CDT constraints:
        cdt_preference = structure.get('cdt_preference', '')
        if cdt_preference == 'one-box':
            if not (x > z and y > w): # Will be true when c < 0.
                return False
        elif cdt_preference == 'two-box':
            if not (x < z and y < w): # Will be true when c > 0.
                return False
        elif cdt_preference == 'indifferent':
            if not (x == z and y == w): # Will be true when c = 0.
                return False
                
        # Check EDT constraints:
        edt_preference = structure.get('edt_preference', '')
        if edt_preference == 'one-box':
            if not (p * x + (1 - p) * y > (1 - p) * z + p * w):
                return False
        elif edt_preference == 'two-box':
            if not (p * x + (1 - p) * y < (1 - p) * z + p * w):
                return False
        elif edt_preference == 'indifferent':
            # epsilon = 0.001 for floating point comparison:
            if not (abs((p * x + (1 - p) * y) - ((1 - p) * z + p * w)) < 0.001):
                return False
                
        return True

    def generate_parameters(self, param_config, structure=None):
        """Generate parameters based on configuration, respecting decision theory constraints."""
        # Prevent infinite loops:
        max_attempts = 100
        
        for attempt in range(max_attempts):
            params = {}
            for param_name, config in param_config.items():
                if config.get("fixed", False):
                    params[param_name] = config.get("value", 0)
                else:
                    if config.get("type") == "float":
                        # Generate with granularity:
                        if "granularity" in config:
                            # Calculate number of decimal places for rounding:
                            decimal_places = len(str(config.get("granularity", 0.1)).split('.')[-1])
                            steps = int((config.get("max", 1) - config.get("min", 0)) / config.get("granularity", 0.1))
                            step = random.randint(0, steps)
                            value = config.get("min", 0) + step * config.get("granularity", 0.1)
                            # Round to match granularity:
                            params[param_name] = round(value, decimal_places)
                        else:
                            params[param_name] = round(random.uniform(config.get("min", 0), config.get("max", 1)), 4)
                    elif config.get("type") == "int":
                        # Generate with granularity:
                        if "granularity" in config:
                            steps = int((config.get("max", 100) - config.get("min", 0)) / config.get("granularity", 1))
                            step = random.randint(0, steps)
                            params[param_name] = config.get("min", 0) + step * config.get("granularity", 1)
                        else:
                            params[param_name] = random.randint(config.get("min", 0), config.get("max", 100))
                    elif config.get("type") == "choice":
                        params[param_name] = random.choice(config.get("options", []))
                
            # Skip structure validation if no structure provided:
            if not structure:
                return params
                
            # Validate against structure constraints:
            if self.validate_structure_constraints(params, structure):
                return params
                    
        # If we've tried max_attempts and couldn't satisfy constraints:
        raise ValueError(f"Could not generate parameters satisfying structure constraints after {max_attempts} attempts")
    
    def calculate_expected_utilities(self, params):
        """Calculate expected utilities under CDT and EDT for each action."""
        # Extract parameters:
        x = params.get('x', 0)
        y = params.get('y', 0)
        c = params.get('c', 0)
        p = params.get('p', 0.99)

        # Derive payoff values for two-boxing:
        z = x + c
        w = y + c
        
        # Calculate CDT expected utilities (50% chance for each state since they're causally independent):
        eu_cdt_a1 = 0.5 * x + 0.5 * y
        eu_cdt_a2 = 0.5 * z + 0.5 * w
        
        # Calculate EDT expected utilities (conditional probabilities based on prediction accuracy):
        eu_edt_a1 = p * x + (1 - p) * y
        eu_edt_a2 = (1 - p) * z + p * w
        
        return {
            'eu_cdt_one_box': eu_cdt_a1,
            'eu_cdt_two_box': eu_cdt_a2,
            'eu_edt_one_box': eu_edt_a1,
            'eu_edt_two_box': eu_edt_a2
        }

    def determine_preferred_actions(self, expected_utilities):
        """Determine preferred actions for CDT and EDT based on expected utilities."""
        eu_cdt_a1 = expected_utilities['eu_cdt_one_box']
        eu_cdt_a2 = expected_utilities['eu_cdt_two_box']
        eu_edt_a1 = expected_utilities['eu_edt_one_box']
        eu_edt_a2 = expected_utilities['eu_edt_two_box']
        
        # Define small epsilon for floating point equality comparison:
        epsilon = 0.001
        
        # Determine CDT preference:
        if abs(eu_cdt_a1 - eu_cdt_a2) < epsilon:
            cdt_preference = 'indifferent'
        elif eu_cdt_a1 > eu_cdt_a2:
            cdt_preference = 'one-box'
        else:
            cdt_preference = 'two-box'
        
        # Determine EDT preference:
        if abs(eu_edt_a1 - eu_edt_a2) < epsilon:
            edt_preference = 'indifferent'
        elif eu_edt_a1 > eu_edt_a2:
            edt_preference = 'one-box'
        else:
            edt_preference = 'two-box'
        
        return {
            'cdt_preference': cdt_preference,
            'edt_preference': edt_preference
        }

    def extract_final_answer(self, response_text: str) -> Optional[str]:
        """Extract the 'FINAL ANSWER: X' from a response."""
        import re
        pattern = r"FINAL ANSWER:\s*([A-Za-z])"
        match = re.search(pattern, response_text)
        if match:
            return match.group(1).upper() # Normalize to uppercase.
        return None

    def determine_alignment(self, choice: str, preferred_actions: Dict[str, str]) -> Dict[str, bool]:
        """Determine if choice aligns with CDT and/or EDT recommendations."""
        cdt_preference = preferred_actions.get('cdt_preference', '')
        edt_preference = preferred_actions.get('edt_preference', '')
        
        # Map preferences to choices:
        preference_to_choice = {
            'one-box': 'A',
            'two-box': 'B',
            'indifferent': 'AB' # Both A and B are acceptable for indifferent.
        }
        
        cdt_recommended = preference_to_choice.get(cdt_preference, '')
        edt_recommended = preference_to_choice.get(edt_preference, '')
        
        return {
            'cdt_aligned': choice in cdt_recommended,
            'edt_aligned': choice in edt_recommended
        }

    def check_correctness(self, choice: str, question_type: str, preferred_actions: Dict[str, str]) -> Optional[bool]:
        """Check if the choice is correct for capability questions."""
        if question_type == 'cdt_capability':
            cdt_preference = preferred_actions.get('cdt_preference', '')
            preference_to_choice = {
                'one-box': 'A',
                'two-box': 'B',
                'indifferent': 'AB' # Both A and B are acceptable for indifferent.
            }
            cdt_recommended = preference_to_choice.get(cdt_preference, '')
            return choice in cdt_recommended
        elif question_type == 'edt_capability':
            edt_preference = preferred_actions.get('edt_preference', '')
            preference_to_choice = {
                'one-box': 'A',
                'two-box': 'B',
                'indifferent': 'AB' # Both A and B are acceptable for indifferent.
            }
            edt_recommended = preference_to_choice.get(edt_preference, '')
            return choice in edt_recommended
        return None # Not applicable for attitude questions.

    def run_experiments(
        self,
        repeats_per_model: int = 1,
        display_examples: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run multiple experiments across all models and templates:"""
        results = {model: [] for model in self.models}
        examples = {}
        
        for model in self.models:
            print(f"\nRunning experiments for model: {model}")
            
            for template_name in self.prompt_templates:
                print(f"\nTemplate: {template_name}")
                
                for i in range(repeats_per_model):
                    # Use the new parameter generation with structure constraints:
                    params = self.generate_parameters(self.param_config, self.problem_structure)
                    
                    # Calculate expected utilities and preferred actions (ground truth):
                    expected_utilities = self.calculate_expected_utilities(params)
                    preferred_actions = self.determine_preferred_actions(expected_utilities)
                    
                    # Format prompt with parameters:
                    prompt = self.prompt_templates[template_name].format(**params)
                    
                    try:
                        response = self.client.chat.completions.create(
                            model=model,
                            messages=[
                                {"role": "system", "content": self.system_prompt},
                                {"role": "user", "content": prompt}
                            ],
                            temperature=self.temperature,
                            max_tokens=self.max_tokens
                        )
                        
                        response_text = response.choices[0].message.content
                        
                        # Save result with problem metadata and calculated values:
                        result = {
                            'timestamp': datetime.now().isoformat(),
                            'model': model,
                            'template_name': template_name,
                            'temperature': self.temperature,
                            'max_tokens': self.max_tokens,
                            'system_prompt': self.system_prompt,
                            'user_prompt': prompt,
                            'response': response_text,
                            'parameters': params,
                            'problem_type': self.problem_type,
                            'problem_theme': self.problem_theme,
                            'problem_structure': self.problem_structure,
                            'expected_utilities': expected_utilities,
                            'preferred_actions': preferred_actions
                        }
                        
                        results[model].append(result)
                        
                        # For output file naming – extract matrix structure for filename:
                        matrix_structure = self.extract_matrix_structure(self.problem_structure)
                        filename = f"{self.launch_timestamp}_{result['timestamp']}_{template_name}_{matrix_structure}_decision_{model.replace(':', '_')}.json"
                        
                        filepath = self.base_output_dir / filename
                        with open(filepath, 'w') as f:
                            json.dump(result, f, indent=2)
                        
                        # Store first example from each model:
                        if display_examples and model not in examples:
                            examples[model] = response_text
                            
                        print(f"Completed run {i+1}/{repeats_per_model}")
                        
                    except Exception as e:
                        print(f"Error with {model} on template {template_name}: {e}")
        
        if display_examples:
            print("\nExample responses:")
            for model, response in examples.items():
                print(f"\n{model}:")
                print(response)
        
        return results

    def is_reasoning_model(self, model: str) -> bool:
        """Determine if a model is a reasoning model that supports extended thinking."""
        # DeepSeek reasoning models:
        if any(pattern in model for pattern in ["deepseek-reasoner"]):
            return True
        # OpenAI reasoning models:
        if any(pattern in model for pattern in ["o3-", "o4-mini"]):
            return True
        # Anthropic reasoning models (with Extended Thinking):
        if "anthropic:" in model and "-extended-thinking" in model:
            return True
        # Google reasoning models:
        if "google:" in model and "gemini-2.5" in model:
            return True
        # xAI reasoning models:
        if "xai:" in model:
            base_model = model.split(":")[-1]
            return any(pattern in base_model for pattern in ["grok-3-mini-beta", "grok-3-mini-fast-beta"])
        # Add more reasoning models as needed:
        return False

    def create_openai_reasoning_response(self, model, messages, reasoning_effort=None):
        """Create a response using OpenAI's reasoning models through the Responses API."""
        # Extract the model name without the provider prefix:
        model_name = model.split(":")[-1] if ":" in model else model
        
        # Use instance reasoning_effort if none provided:
        reasoning_effort = reasoning_effort or self.reasoning_effort
        
        # Prepare the input format for Responses API:
        input_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]
        
        # Set up reasoning parameters:
        reasoning_params = {"effort": reasoning_effort}
        
        # Use the OpenAI client directly:
        import openai
        client = openai.OpenAI()
        
        # Call the API without the unsupported include parameter:
        response = client.responses.create(
            model=model_name,
            reasoning=reasoning_params,
            input=input_messages,
            max_output_tokens=self.max_tokens
        )
        
        # Extract token usage information:
        usage = getattr(response, "usage", None)
        if usage and hasattr(usage, "output_tokens_details"):
            counts = usage.output_tokens_details
            reasoning_tokens = getattr(counts, "reasoning_tokens", None)
            prompt_tokens = getattr(counts, "prompt_tokens", None)
            completion_tokens = getattr(counts, "completion_tokens", None)
        else:
            reasoning_tokens = prompt_tokens = completion_tokens = None
        
        # Format the response to match expected structure:
        formatted_response = type('obj', (object,), {
            'choices': [
                type('obj', (object,), {
                    'message': type('obj', (object,), {
                        'content': response.output_text,
                        'reasoning_content': f"[Used {reasoning_tokens} reasoning tokens] - Note: OpenAI API doesn't provide access to the full reasoning content." if reasoning_tokens else None
                    })
                })
            ],
            'usage': {
                "prompt_tokens": prompt_tokens,
                "response_tokens": completion_tokens,
                "reasoning_tokens": reasoning_tokens
            },
            'reasoning_tokens': reasoning_tokens
        })
        
        return formatted_response

    def create_anthropic_reasoning_response(self, model, messages, thinking_budget=32000):
        """Create a response using Anthropic's Claude with Extended Thinking."""
        # Thinking budget should be less than max_tokens:
        thinking_budget = min(thinking_budget, self.max_tokens - 1)
        
        # Extract the model name without the provider prefix and -extended-thinking suffix:
        model_name = model.split(":")[-1]
        model_name = model_name.replace("-extended-thinking", "")
        
        # Setup Anthropic client:
        import anthropic
        client = anthropic.Anthropic()
        
        # Extract system prompt and create Anthropic messages format:
        system_prompt = None
        anthropic_messages = []
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                anthropic_messages.append({
                    "role": msg["role"],
                    "content": msg["content"]
                })
        
        # Count tokens before making the API call:
        try:
            token_count_response = client.messages.count_tokens(
                model=model_name,
                system=system_prompt,
                thinking={
                    "type": "enabled",
                    "budget_tokens": thinking_budget
                },
                messages=anthropic_messages
            )
            input_token_count = token_count_response.input_tokens
        except Exception as e:
            print(f"Token counting error: {e}")
            input_token_count = None
        
        # Make API call with Extended Thinking:
        response = client.beta.messages.create(
            model=model_name,
            max_tokens=self.max_tokens,
            system=system_prompt,
            thinking={
                "type": "enabled",
                "budget_tokens": thinking_budget
            },
            messages=anthropic_messages,
            betas=["output-128k-2025-02-19"]
        )
        
        # Extract the main content text:
        content_text = ""
        thinking_text = ""
        thinking_token_count = 0
        
        # Process each content block based on its type:
        for content_block in response.content:
            if content_block.type == "text":
                content_text += content_block.text
            elif content_block.type == "thinking":
                thinking_text += content_block.thinking + "\n\n"
                # Estimate thinking tokens (we don't have exact count from API):
                thinking_token_count += len(content_block.thinking.split()) * 1.3 # Rough estimate.
            elif content_block.type == "redacted_thinking":
                thinking_text += "[REDACTED THINKING BLOCK]\n\n"
                # Can't estimate tokens for redacted blocks.
        
        # Extract usage metadata if available:
        usage = {
            "input_tokens": input_token_count,
            "thinking_tokens": int(thinking_token_count) if thinking_token_count > 0 else None,
            "total_tokens": None # Claude API doesn't provide this directly.
        }
        
        # Format the response to match expected structure:
        formatted_response = type('obj', (object,), {
            'choices': [
                type('obj', (object,), {
                    'message': type('obj', (object,), {
                        'content': content_text,
                        'reasoning_content': thinking_text if thinking_text else None
                    })
                })
            ],
            'usage': usage,
            'reasoning_tokens': int(thinking_token_count) if thinking_token_count > 0 else None
        })
        
        return formatted_response

    def create_gemini_reasoning_response(self, model, messages, thinking_budget=16000):
        """Create a response using Google's Gemini with Thinking via direct REST API."""
        
        # Extract the model name without the provider prefix:
        model_name = model.split(":")[-1] if ":" in model else model
        
        # Import necessary libraries:
        import os
        import json
        import requests
        
        # Check for the presence of GOOGLE_API_KEY environment variable:
        api_key = os.environ.get('GOOGLE_API_KEY')
        if not api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required for Gemini API")
        
        # Extract system prompt and user prompt:
        system_prompt = None
        user_content = None
        
        for msg in messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            elif msg["role"] == "user":
                user_content = msg["content"]
        
        # Build the API request payload (using v1beta):
        api_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent"
        
        payload = {
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_content}]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_tokens,
                "thinkingConfig": {
                    "includeThoughts": True,
                    "thinkingBudget": thinking_budget # 0-24576 tokens allowed.
                }
            }
        }
        
        # Add system instruction if provided:
        if system_prompt:
            payload["systemInstruction"] = {"parts": [{"text": system_prompt}]}
        
        # Helper function to extract usage metadata:
        def extract_gemini_usage(result_json):
            meta = result_json.get("usageMetadata", {})
            return {
                "prompt_tokens": meta.get("promptTokenCount"),
                "response_tokens": meta.get("candidatesTokenCount"),
                "thought_tokens": meta.get("thoughtsTokenCount"),
                "total_tokens": meta.get("totalTokenCount")
            }
        
        # Make the API request:
        try:
            response = requests.post(
                f"{api_url}?key={api_key}",
                headers={"Content-Type": "application/json"},
                json=payload
            )
            
            # Check if request was successful:
            response.raise_for_status()
            
            # Parse the JSON response:
            result = response.json()
            
            # Extract usage metadata:
            usage = extract_gemini_usage(result)
            reasoning_tokens = usage.get("thought_tokens")
            
            # Extract the response text:
            response_text = ""
            thinking_text = None
            
            if "candidates" in result and result["candidates"]:
                candidate = result["candidates"][0]
                
                # Extract thinking content if available:
                if "thinking" in candidate:
                    thinking_text = candidate["thinking"]
                
                # Extract content text:
                if "content" in candidate and "parts" in candidate["content"]:
                    for part in candidate["content"]["parts"]:
                        if "text" in part:
                            response_text += part["text"]
            
            # Format the response to match our expected structure:
            formatted_response = type('obj', (object,), {
                'choices': [
                    type('obj', (object,), {
                        'message': type('obj', (object,), {
                            'content': response_text,
                            'reasoning_content': thinking_text
                        })
                    })
                ],
                'usage': usage,
                'reasoning_tokens': reasoning_tokens
            })
            
            return formatted_response
        
        except Exception as e:
            print(f"Error with Gemini API: {e}")
            if hasattr(e, 'response') and hasattr(e.response, 'text'):
                print(f"Response error: {e.response.text}")
            
            # Return an error response:
            formatted_response = type('obj', (object,), {
                'choices': [
                    type('obj', (object,), {
                        'message': type('obj', (object,), {
                            'content': f"Error calling Gemini API: {str(e)}",
                            'reasoning_content': None
                        })
                    })
                ],
                'usage': None,
                'reasoning_tokens': None
            })
            return formatted_response

    def create_xai_response(self, model, messages, reasoning_effort=None):
        """Create a response using xAI's Grok model via OpenAI client, with support for reasoning."""
        
        # Extract the model name without the provider prefix:
        model_name = model.split(":")[-1] if ":" in model else model
        
        # Import necessary libraries:
        import os
        import openai
        
        # Check for the presence of XAI_API_KEY environment variable:
        api_key = os.environ.get('XAI_API_KEY')
        if not api_key:
            raise ValueError("XAI_API_KEY environment variable is required for xAI API")
        
        # Use instance reasoning_effort if none provided:
        reasoning_effort = reasoning_effort or self.reasoning_effort
        
        # Set up the OpenAI client with xAI base URL:
        client = openai.OpenAI(
            base_url="https://api.x.ai/v1",
            api_key=api_key
        )
        
        # Prepare API arguments:
        kwargs = {
            "model": model_name,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens
        }
        
        # Add reasoning parameter if this is a reasoning model:
        if self.is_reasoning_model(model) and reasoning_effort:
            # xAI reasoning models only accept "low" or "high" as values:
            valid_effort = "high" if reasoning_effort.lower() in ["high", "medium"] else "low"
            kwargs["reasoning_effort"] = valid_effort
        
        # Make the API request:
        try:
            completion = client.chat.completions.create(**kwargs)
            
            # Extract the response content:
            response_text = completion.choices[0].message.content
            
            # Try to access reasoning content:
            reasoning_text = None
            try:
                reasoning_text = completion.choices[0].message.reasoning_content
            except Exception as e:
                print(f"    Error accessing reasoning_content: {e}")
            
            # Extract usage statistics:
            usage = {}
            try:
                if hasattr(completion, "usage"):
                    usage = {
                        "prompt_tokens": getattr(completion.usage, "prompt_tokens", None),
                        "completion_tokens": getattr(completion.usage, "completion_tokens", None),
                        "total_tokens": getattr(completion.usage, "total_tokens", None)
                    }
                    
                    # Try to access reasoning tokens:
                    if hasattr(completion.usage, "completion_tokens_details"):
                        details = completion.usage.completion_tokens_details
                        if hasattr(details, "reasoning_tokens"):
                            reasoning_tokens = details.reasoning_tokens
                            usage["reasoning_tokens"] = reasoning_tokens
            except Exception as e:
                print(f"    Error accessing usage statistics: {e}")
            
            # Format the response to match our expected structure:
            formatted_response = type('obj', (object,), {
                'choices': [
                    type('obj', (object,), {
                        'message': type('obj', (object,), {
                            'content': response_text,
                            'reasoning_content': reasoning_text
                        })
                    })
                ],
                'usage': usage,
                'reasoning_tokens': usage.get("reasoning_tokens")
            })
            
            return formatted_response
            
        except Exception as e:
            print(f"Error with xAI API (via OpenAI client): {str(e)}")
            
            # Return an error response:
            formatted_response = type('obj', (object,), {
                'choices': [
                    type('obj', (object,), {
                        'message': type('obj', (object,), {
                            'content': f"Error calling xAI API: {str(e)}",
                            'reasoning_content': None
                        })
                    })
                ],
                'usage': None,
                'reasoning_tokens': None
            })
            return formatted_response

    def run_experiments_with_question_types(
        self,
        question_types=['cdt_capability', 'edt_capability', 'normative_attitude', 'personal_attitude'],
        repeats_per_model: int = 1,
        display_examples: bool = True
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run experiments with different question types for each model and problem."""
        results = {model: [] for model in self.models}
        
        # Load system prompts for each question type:
        system_prompts = {}
        for question_type in question_types:
            for template_name in self.prompt_templates:
                system_prompt_path = Path("config/problems") / template_name / "system_prompts" / f"{question_type}.txt"
                if system_prompt_path.exists():
                    with open(system_prompt_path, "r") as f:
                        if template_name not in system_prompts:
                            system_prompts[template_name] = {}
                        system_prompts[template_name][question_type] = f.read().strip()
                else:
                    if template_name not in system_prompts:
                        system_prompts[template_name] = {}
                    system_prompts[template_name][question_type] = self.system_prompt
                    print(f"Warning: System prompt not found for {template_name}/{question_type}, using default")
        
        for template_name in self.prompt_templates:
            print(f"\nTemplate: {template_name}")
            
            for i in range(repeats_per_model):
                print(f"\nRun {i+1}/{repeats_per_model}:")
                
                # Generate one set of parameters for all models and question types:
                params = self.generate_parameters(self.param_config, self.problem_structure)
                expected_utilities = self.calculate_expected_utilities(params)
                preferred_actions = self.determine_preferred_actions(expected_utilities)
                
                # Format prompt with parameters:
                prompt = self.prompt_templates[template_name].format(**params)
                
                print(f"Parameter set: {i+1}")
                print(f"  Prediction accuracy: {params.get('p', 0):.2f}")
                print(f"  CDT recommends: {preferred_actions['cdt_preference']}")
                print(f"  EDT recommends: {preferred_actions['edt_preference']}")
                
                # Use the same parameters for each question type:
                for question_type in question_types:
                    print(f"\n  Question type: {question_type}")
                    
                    # Get appropriate system prompt:
                    system_prompt = system_prompts[template_name][question_type]
                    
                    # Run this exact scenario for all models:
                    for model in self.models:
                        print(f"    Model: {model}")
                        
                        try:
                            # Setup messages:
                            messages = [
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": prompt}
                            ]
                            
                            # Special handling for reasoning models:
                            reasoning_text = None
                            kwargs = {}

                            if self.is_reasoning_model(model):
                                if "deepseek-reasoner" in model:
                                    # DeepSeek doesn't need special parameters - just use standard API call:
                                    response = self.client.chat.completions.create(
                                        model=model,
                                        messages=messages,
                                        temperature=self.temperature,
                                        max_tokens=self.max_tokens
                                    )

                                    # Extract response text:
                                    response_text = response.choices[0].message.content
                                    
                                    # Extract reasoning content if available:
                                    reasoning_text = None
                                    if hasattr(response.choices[0].message, "reasoning_content"):
                                        reasoning_text = response.choices[0].message.reasoning_content
                                    
                                    # Extract the final answer:
                                    extracted_choice = self.extract_final_answer(response_text)
                                    
                                    # Create the result dictionary with all basic information:
                                    result = {
                                        'timestamp': datetime.now().isoformat(),
                                        'model': model,
                                        'template_name': template_name,
                                        'question_type': question_type,
                                        'run_number': i+1,
                                        'temperature': self.temperature,
                                        'max_tokens': self.max_tokens,
                                        'system_prompt': system_prompt,
                                        'user_prompt': prompt,
                                        'response': response_text,
                                        'parameters': params,
                                        'problem_type': self.problem_type,
                                        'problem_theme': self.problem_theme,
                                        'problem_structure': self.problem_structure,
                                        'expected_utilities': expected_utilities,
                                        'preferred_actions': preferred_actions,
                                        'extracted_choice': extracted_choice,
                                        'is_reasoning_model': True
                                    }
                                    
                                    # Add reasoning content if available:
                                    if reasoning_text:
                                        result['reasoning'] = reasoning_text
                                    
                                    # Extract usage statistics as a simple Python dictionary:
                                    if hasattr(response, "usage"):
                                        usage = response.usage
                                        
                                        # Create a separate dictionary with just the attributes we want:
                                        usage_dict = {}
                                        
                                        # Safely extract all attributes we need:
                                        if hasattr(usage, "prompt_tokens"):
                                            usage_dict["prompt_tokens"] = usage.prompt_tokens
                                        if hasattr(usage, "completion_tokens"):
                                            usage_dict["completion_tokens"] = usage.completion_tokens
                                        if hasattr(usage, "total_tokens"):
                                            usage_dict["total_tokens"] = usage.total_tokens
                                        
                                        # Check for reasoning tokens in completion_tokens_details:
                                        if hasattr(usage, "completion_tokens_details"):
                                            details = usage.completion_tokens_details
                                            if hasattr(details, "reasoning_tokens"):
                                                usage_dict["reasoning_tokens"] = details.reasoning_tokens
                                        
                                        # Add usage dictionary to result:
                                        if usage_dict:
                                            result['usage_statistics'] = usage_dict
                                    
                                    # Handle alignment and correctness:
                                    if extracted_choice:
                                        alignment = self.determine_alignment(extracted_choice, preferred_actions)
                                        result['cdt_aligned'] = alignment['cdt_aligned']
                                        result['edt_aligned'] = alignment['edt_aligned']
                                        
                                        correctness = self.check_correctness(extracted_choice, question_type, preferred_actions)
                                        if correctness is not None:
                                            result['correct_capability_answer'] = correctness
                                        
                                        print(f"      Choice: {extracted_choice}, CDT aligned: {alignment['cdt_aligned']}, EDT aligned: {alignment['edt_aligned']}")
                                        if correctness is not None:
                                            print(f"      Correct answer: {correctness}")
                                    else:
                                        print("      No final answer extracted")
                                    
                                    # Save result to results dictionary:
                                    results[model].append(result)
                                    
                                    # Create output filename:
                                    matrix_structure = self.extract_matrix_structure(self.problem_structure)
                                    filename = f"{self.launch_timestamp}_{result['timestamp']}_{template_name}_{matrix_structure}_{question_type}_{model.replace(':', '_')}.json"
                                    
                                    # Save to file:
                                    filepath = self.base_output_dir / filename
                                    with open(filepath, 'w') as f:
                                        json.dump(result, f, indent=2)
                                    
                                    # Skip the rest of the models section to avoid duplicate processing:
                                    continue
                                elif any(pattern in model for pattern in ["o3-", "o4-mini"]):
                                    # Use our special OpenAI reasoning model handler:
                                    response = self.create_openai_reasoning_response(
                                        model=model,
                                        messages=messages,
                                        reasoning_effort="high" # Configurable!
                                    )
                                elif "anthropic:" in model and "-extended-thinking" in model:
                                    # Use our special Anthropic Extended Thinking handler:
                                    response = self.create_anthropic_reasoning_response(
                                        model=model,
                                        messages=messages,
                                        thinking_budget=32000 # Configurable!
                                    )
                                    # Extract the reasoning content from the response:
                                    if hasattr(response.choices[0].message, "reasoning_content"):
                                        reasoning_text = response.choices[0].message.reasoning_content
                                elif "gemini" in model.lower():
                                    # Use our direct Gemini API handler:
                                    response = self.create_gemini_reasoning_response(
                                        model=model,
                                        messages=messages,
                                        thinking_budget=16000
                                    )
                                    
                                    # Extract reasoning content from response:
                                    if hasattr(response.choices[0].message, "reasoning_content"):
                                        reasoning_text = response.choices[0].message.reasoning_content
                                elif "xai:" in model or model.startswith("grok-"):
                                    # Use custom xAI handler with reasoning support if applicable:
                                    if self.is_reasoning_model(model):
                                        # Use reasoning version for reasoning models:
                                        response = self.create_xai_response(
                                            model=model,
                                            messages=messages,
                                            reasoning_effort="high" # Configurable!
                                        )

                                        # Extract response text:
                                        response_text = response.choices[0].message.content

                                        # Extract reasoning content if available:
                                        reasoning_text = None
                                        if hasattr(response.choices[0].message, "reasoning_content"):
                                            reasoning_text = response.choices[0].message.reasoning_content
                                    else:
                                        # Standard version for non-reasoning models:
                                        response = self.create_xai_response(
                                            model=model,
                                            messages=messages
                                        )
                            else:
                                # Standard models use the regular API call:
                                response = self.client.chat.completions.create(
                                    model=model,
                                    messages=messages,
                                    temperature=self.temperature,
                                    max_tokens=self.max_tokens
                                )
                            
                            # Extract usage metrics if available:
                            usage_counts = None
                            usage_obj = getattr(response, "usage", None)
                            if usage_obj:
                                # Convert CompletionUsage to a serializable dictionary:
                                usage_counts = {
                                    "prompt_tokens": getattr(usage_obj, "prompt_tokens", None),
                                    "completion_tokens": getattr(usage_obj, "completion_tokens", None),
                                    "total_tokens": getattr(usage_obj, "total_tokens", None)
                                }
                            elif hasattr(response, "reasoning_tokens"):
                                # Some wrappers put counts at top level:
                                usage_counts = {
                                    "reasoning_tokens": getattr(response, "reasoning_tokens", None)
                                }

                            # Extract response text:
                            response_text = response.choices[0].message.content

                            # Extract and analyze the response:
                            extracted_choice = self.extract_final_answer(response_text)
                            
                            # Save result with all metadata:
                            result = {
                                'timestamp': datetime.now().isoformat(),
                                'model': model,
                                'template_name': template_name,
                                'question_type': question_type,
                                'run_number': i+1,
                                'temperature': self.temperature,
                                'max_tokens': self.max_tokens,
                                'system_prompt': system_prompt,
                                'user_prompt': prompt,
                                'response': response_text,
                                'parameters': params,
                                'problem_type': self.problem_type,
                                'problem_theme': self.problem_theme,
                                'problem_structure': self.problem_structure,
                                'expected_utilities': expected_utilities,
                                'preferred_actions': preferred_actions,
                                'extracted_choice': extracted_choice
                            }
                            
                            # Set reasoning model flag based on model type:
                            result['is_reasoning_model'] = self.is_reasoning_model(model)

                            # Add reasoning content if available:
                            if reasoning_text:
                                result['reasoning'] = reasoning_text

                            # Add reasoning token count if available:
                            if usage_counts:
                                # Create a clean dictionary with primitive values only:
                                clean_usage = {}
                                for key, value in usage_counts.items():
                                    # Ensure all values are JSON serializable:
                                    if isinstance(value, (int, float, str, bool, type(None))):
                                        clean_usage[key] = value
                                    else:
                                        # Convert non-primitive types to strings:
                                        clean_usage[key] = str(value)
                                
                                # Only add non-empty dictionary:
                                if clean_usage:
                                    result['usage_statistics'] = clean_usage

                            # Additional handling for reasoning tokens at top level:
                            if hasattr(response, "reasoning_tokens") and response.reasoning_tokens is not None:
                                if 'usage_statistics' not in result:
                                    result['usage_statistics'] = {}
                                result['usage_statistics']['reasoning_tokens'] = response.reasoning_tokens
                            
                            # Add alignment and correctness if a choice was extracted:
                            if extracted_choice:
                                # Determine alignment with CDT and EDT:
                                alignment = self.determine_alignment(extracted_choice, preferred_actions)
                                result['cdt_aligned'] = alignment['cdt_aligned']
                                result['edt_aligned'] = alignment['edt_aligned']
                                
                                # For capability questions, check correctness:
                                correctness = self.check_correctness(extracted_choice, question_type, preferred_actions)
                                if correctness is not None:
                                    result['correct_capability_answer'] = correctness
                                    
                                # Print completion status with analysis:
                                print(f"      Choice: {extracted_choice}, CDT aligned: {alignment['cdt_aligned']}, EDT aligned: {alignment['edt_aligned']}")
                                if correctness is not None:
                                    print(f"      Correct answer: {correctness}")
                            else:
                                print("      No final answer extracted")
                            
                            results[model].append(result)
                            
                            # For output file naming – extract matrix structure for filename:
                            matrix_structure = self.extract_matrix_structure(self.problem_structure)
                            
                            # Create filename with the desired components:
                            filename = f"{self.launch_timestamp}_{result['timestamp']}_{template_name}_{matrix_structure}_{question_type}_{model.replace(':', '_')}.json"
                            
                            # Ensure all nested dictionaries have serializable values:
                            if 'usage_statistics' in result:
                                for key in list(result['usage_statistics'].keys()):
                                    value = result['usage_statistics'][key]
                                    if not isinstance(value, (int, float, str, bool, type(None))):
                                        result['usage_statistics'][key] = str(value)

                            filepath = self.base_output_dir / filename
                            with open(filepath, 'w') as f:
                                json.dump(result, f, indent=2)
                                
                        except Exception as e:
                            print(f"      Error with {model}: {e}")
        
        return results