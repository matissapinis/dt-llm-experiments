# ./src/framework.py

import aisuite as ai
from datetime import datetime
import json
from pathlib import Path
from typing import Dict, Any
from dotenv import load_dotenv

class NewcombExperiment:
    """Simple framework for running a single Newcomblike decision theory experiment with an LLM."""
    
    def __init__(
        self,
        base_output_dir: str = "experiment_results",
        temperature: float = 0.8,
        max_tokens: int = 400
    ):
        # Load environment variables:
        load_dotenv()
        
        # Initialize AI client:
        self.client = ai.Client()
        
        # Set up output directory:
        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(exist_ok=True)
        
        # Set parameters:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.system_prompt = ""
        self.model = ""
        self.prompt_template = ""
    
    def load_system_prompt(self, filepath: str) -> None:
        """Load system prompt from file:"""
        with open(filepath, 'r') as f:
            self.system_prompt = f.read().strip()
    
    def load_prompt_template(self, filepath: str) -> None:
        """Load user prompt template from file:"""
        with open(filepath, 'r') as f:
            self.prompt_template = f.read().strip()
    
    def set_model(self, model: str) -> None:
        """Set the model to use:"""
        self.model = model
    
    def format_prompt(self, params: Dict[str, Any]) -> str:
        """Format the prompt template with given parameters:"""
        return self.prompt_template.format(**params)
    
    def save_response(
        self,
        prompt: str,
        response: str,
        params: Dict[str, Any]
    ) -> None:
        """Save response and metadata to a JSON file:"""
        timestamp = datetime.now().isoformat()
        
        metadata = {
            'timestamp': timestamp,
            'model': self.model,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'system_prompt': self.system_prompt,
            'user_prompt': prompt,
            'response': response,
            'parameters': params
        }
        
        filename = f"{timestamp}_{self.model.replace(':', '_')}_response.json"
        filepath = self.base_output_dir / filename
        
        with open(filepath, 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def run_single_experiment(self, params: Dict[str, Any]) -> str:
        """Run a single experiment with fixed parameters:"""
        # Format the prompt with parameters:
        prompt = self.format_prompt(params)
        
        try:
            # Get model response:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            response_text = response.choices[0].message.content
            
            # Save response:
            self.save_response(
                prompt=prompt,
                response=response_text,
                params=params
            )
            
            return response_text
            
        except Exception as e:
            print(f"Error running experiment: {e}")
            return None