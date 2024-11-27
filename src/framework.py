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