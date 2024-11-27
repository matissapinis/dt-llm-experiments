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
        max_tokens: int = 400,
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
        self.system_prompt = ""
        self.models = []
        self.prompt_templates = {}
        
        # Set random seed if provided:
        if random_seed is not None:
            random.seed(random_seed)
    
    def set_models(self, models: List[str]) -> None:
        """Set list of models to use in experiments:"""
        self.models = models
    
    def load_system_prompt(self, prompt: str) -> None:
        """Load system prompt directly from string:"""
        self.system_prompt = prompt.strip()

    def load_problem(self, problem_name: str) -> None:
        """Load a problem's system prompt, template and parameters from config directory:"""
        problem_dir = Path("config/problems") / problem_name
        
        # Clear previous templates:
        self.prompt_templates = {}
        
        # Load system prompt:
        with open(problem_dir / "system_prompt.txt", "r") as f:
            self.load_system_prompt(f.read())
        
        # Load template:
        with open(problem_dir / "user_prompt_template.txt", "r") as f:
            self.add_prompt_template(problem_name, f.read())
        
        # Load parameters:
        with open(problem_dir / "user_prompt_parameters.json", "r") as f:
            self.param_config = json.load(f)
    
    def list_problems(self) -> List[str]:
        """List all available problems in config directory:"""
        problems_dir = Path("config/problems")
        return [p.name for p in problems_dir.iterdir() if p.is_dir() and 
                all((p / f).exists() for f in [
                    "system_prompt.txt",
                    "user_prompt_template.txt",
                    "user_prompt_parameters.json"
                ])]

    def run_all_problems(
        self,
        repeats_per_model: int = 2,
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
    
    def generate_parameters(self, param_config: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Generate parameters based on configuration:"""
        params = {}
        for param_name, config in param_config.items():
            if config.get("fixed"):
                params[param_name] = config["value"]
            else:
                if config["type"] == "float":
                    params[param_name] = random.uniform(config["min"], config["max"])
                elif config["type"] == "int":
                    params[param_name] = random.randint(config["min"], config["max"])
                elif config["type"] == "choice":
                    params[param_name] = random.choice(config["options"])
        return params
    
    def run_experiments(
        self,
        param_config: Dict[str, Dict[str, Any]],
        repeats_per_model: int = 5,
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
                    params = self.generate_parameters(param_config)
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
                        
                        # Save result:
                        result = {
                            'timestamp': datetime.now().isoformat(),
                            'model': model,
                            'template_name': template_name,
                            'temperature': self.temperature,
                            'max_tokens': self.max_tokens,
                            'system_prompt': self.system_prompt,
                            'user_prompt': prompt,
                            'response': response_text,
                            'parameters': params
                        }
                        
                        results[model].append(result)
                        
                        # Save to file:
                        filename = f"{result['timestamp']}_{model.replace(':', '_')}_{template_name}.json"
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