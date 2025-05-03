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
        reasoning_effort: str = "medium",
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
                else:
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
        
        # Format the response to match expected structure:
        formatted_response = type('obj', (object,), {
            'choices': [
                type('obj', (object,), {
                    'message': type('obj', (object,), {
                        'content': response.output_text,
                        'reasoning_content': None # Will be populated if we can get reasoning content.
                    })
                })
            ]
        })
        
        # Try to extract reasoning tokens usage if available:
        try:
            if hasattr(response, 'usage') and hasattr(response.usage, 'output_tokens_details'):
                reasoning_tokens = response.usage.output_tokens_details.reasoning_tokens
                formatted_response.choices[0].message.reasoning_content = f"[Used {reasoning_tokens} reasoning tokens] - Note: OpenAI API doesn't provide access to the full reasoning content."
        except AttributeError:
            # If the attribute doesn't exist, just continue
            pass
        
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
                                    
                                    # Try to extract reasoning content:
                                    if hasattr(response.choices[0].message, "reasoning_content"):
                                        reasoning_text = response.choices[0].message.reasoning_content
                                elif any(pattern in model for pattern in ["o3-", "o4-mini"]):
                                    # Use our special OpenAI reasoning model handler:
                                    response = self.create_openai_reasoning_response(
                                        model=model,
                                        messages=messages,
                                        reasoning_effort="high"
                                    )
                            else:
                                # Standard models use the regular API call:
                                response = self.client.chat.completions.create(
                                    model=model,
                                    messages=messages,
                                    temperature=self.temperature,
                                    max_tokens=self.max_tokens
                                )
                            
                            # Extract response text correctly:
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
                            
                            # Add reasoning if available:
                            if reasoning_text:
                                result['reasoning'] = reasoning_text
                                result['is_reasoning_model'] = True
                            else:
                                result['is_reasoning_model'] = False

                            # Set reasoning model flag based on model type, not just reasoning content:
                            result['is_reasoning_model'] = self.is_reasoning_model(model)
                            if reasoning_text:
                                result['reasoning'] = reasoning_text
                            
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
                            
                            filepath = self.base_output_dir / filename
                            with open(filepath, 'w') as f:
                                json.dump(result, f, indent=2)
                                
                        except Exception as e:
                            print(f"      Error with {model}: {e}")
        
        return results