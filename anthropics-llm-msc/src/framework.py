# ./src/framework.py

import os
import time
from datetime import datetime
import json
import random
from pathlib import Path
from typing import Dict, Any, List, Union, Optional, Tuple
from dotenv import load_dotenv
from openai import OpenAI


OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


# Roots searched (in order) when resolving a problem name to its config directory.
# Lets the same framework run BSc Newcomblike problems and MSc anthropic-reasoning
# problems without per-call path arguments:
PROBLEM_CONFIG_DIRS = [
    Path("config/problems"),
    Path("config/problems_msc_draft"),
]

# Canonical display order for question types when auto-detecting from a problem's
# system_prompts/ directory. Unknown types fall through to the end:
QUESTION_TYPE_ORDER = [
    'cdt_capability', 'edt_capability',
    'ssa_capability', 'sia_capability',
    'normative_attitude', 'personal_attitude',
]


# Per-model output and reasoning caps for the MSc lineup. Values derived from
# OpenRouter catalog (`top_provider.max_completion_tokens`) cross-checked against
# upstream provider documentation; reasoning_on max_tokens leaves ~8k headroom
# below the model output cap for the visible answer. reasoning_off = None means
# the model is reasoning-only on OR and the OFF leg should be skipped.
# reasoning_on = None means the model is used exclusively as the OFF partner
# of another model in the same provider family (the case for Gemini 3 Flash,
# which fills Gemini 3.1 Pro's OFF leg).
MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "anthropic/claude-opus-4.7":      {"max_tokens": 128000, "reasoning_off": {"enabled": False}, "reasoning_on": {"max_tokens": 119808}},
    "google/gemini-3.1-pro-preview":  {"max_tokens":  65536, "reasoning_off": None,                "reasoning_on": {"max_tokens":  57344}},
    "google/gemini-3-flash-preview":  {"max_tokens":  65536, "reasoning_off": {"enabled": False}, "reasoning_on": None},
    "openai/gpt-5.5":                 {"max_tokens": 128000, "reasoning_off": {"enabled": False}, "reasoning_on": {"effort": "high"}},
    "x-ai/grok-4.3":                  {"max_tokens": 131000, "reasoning_off": {"enabled": False}, "reasoning_on": {"effort": "high"}},
    "z-ai/glm-5.1":                   {"max_tokens": 128000, "reasoning_off": {"enabled": False}, "reasoning_on": {"max_tokens": 119808}},
    "qwen/qwen3.6-max-preview":       {"max_tokens":  65536, "reasoning_off": {"enabled": False}, "reasoning_on": {"max_tokens":  57344}},
    "moonshotai/kimi-k2.6":           {"max_tokens": 256000, "reasoning_off": {"enabled": False}, "reasoning_on": {"max_tokens": 247808}},
    "deepseek/deepseek-v4-pro":       {"max_tokens": 384000, "reasoning_off": {"enabled": False}, "reasoning_on": {"effort": "high"}},
}


class NewcombExperiment:
    """Framework for running batches of Newcomblike decision theory experiments with LLMs."""

    def __init__(
        self,
        base_output_dir: str = "experiment_results",
        temperature: float = 0.8,
        max_tokens: int = 128000,
        reasoning_config: Optional[Dict[str, Any]] = None,
        api_key: Optional[str] = None,
        random_seed: Optional[int] = None,
    ):
        """Configure framework for OpenRouter-backed runs.

        reasoning_config: OpenRouter `reasoning` extra_body, e.g.
          - None: provider default (most hybrid models reason adaptively)
          - {"enabled": False}: single forward pass, no thinking
          - {"max_tokens": 60000}: explicit large reasoning budget
          - {"effort": "high"|"medium"|"low"}: permitted cap (model decides depth)
        """
        load_dotenv()

        resolved_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not resolved_key:
            raise RuntimeError(
                "OPENROUTER_API_KEY not set. Get a key at "
                "https://openrouter.ai/settings/keys and export it, or pass api_key=..."
            )
        self.client = OpenAI(base_url=OPENROUTER_BASE_URL, api_key=resolved_key)

        self.base_output_dir = Path(base_output_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_config = reasoning_config
        self.system_prompt = ""
        self.models: List[str] = []
        self.prompt_templates: Dict[str, str] = {}
        self.problem_valence: Optional[str] = None

        self.launch_timestamp = datetime.now().isoformat()

        if random_seed is not None:
            random.seed(random_seed)

    def extract_matrix_structure(self, structure):
        """Compact filename-safe string describing the problem's structure.

        Works for both Newcomblike (cdt/edt_preference) and anthropic-reasoning
        (ssa/sia_preference) configs.
        """
        if not structure:
            return "unknown"
        structure_type = structure.get('type', '')
        if 'cdt_preference' in structure:
            a, b = structure.get('cdt_preference', ''), structure.get('edt_preference', '')
        elif 'ssa_preference' in structure:
            a, b = structure.get('ssa_preference', ''), structure.get('sia_preference', '')
        else:
            a = b = ''
        return f"{structure_type}_{a}-{b}"

    def validate_all_models(self, models: List[str]) -> List[str]:
        """Verify each requested model id exists in OpenRouter's catalog.

        A single GET /api/v1/models is far cheaper and more reliable than the
        per-provider validation soup the BSc framework used. Raises with the
        list of unknown ids if any model is not on OpenRouter.
        """
        print("\nValidating model availability against OpenRouter catalog...")
        try:
            catalog = self.client.models.list()
            catalog_ids = {m.id for m in catalog.data}
        except Exception as e:
            raise RuntimeError(f"Could not fetch OpenRouter model catalog: {e}")

        unknown = [m for m in models if m not in catalog_ids]
        if unknown:
            raise RuntimeError(
                f"Unknown OpenRouter model ids: {unknown}\n"
                f"Visit https://openrouter.ai/models to see the catalog."
            )
        for m in models:
            print(f"  Available: {m}")
        return list(models)

    def set_models(self, models: List[str]) -> None:
        """Set list of models to use in experiments with strict validation:"""
        self.models = self.validate_all_models(models)
        print(f"\nAll {len(self.models)} requested models are available.")
    
    def load_system_prompt(self, prompt: str) -> None:
        """Load system prompt directly from string:"""
        self.system_prompt = prompt.strip()

    def find_problem_dir(self, problem_name: str) -> Path:
        """Resolve a problem name to its config directory across all known roots.

        Searches PROBLEM_CONFIG_DIRS in order and returns the first match.
        """
        for root in PROBLEM_CONFIG_DIRS:
            candidate = root / problem_name
            if candidate.exists():
                return candidate
        raise FileNotFoundError(
            f"Problem '{problem_name}' not found in any of: "
            f"{[str(r) for r in PROBLEM_CONFIG_DIRS]}"
        )

    def load_problem(self, problem_name: str) -> None:
        """Load a problem's configuration, template and parameters from config directory:"""
        problem_dir = self.find_problem_dir(problem_name)

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
                # MSc-only field (DD / PADD / TDD variants); None for BSc problems:
                self.problem_valence = config_data.get("valence")

                # Extract parameter config:
                if "parameters" in config_data:
                    self.param_config = config_data["parameters"]
                else:
                    self.param_config = {}

                # Extract row order:
                self.row_order = config_data.get("row_order", "12") # Default to "12" if not specified.
            else:
                # Legacy format:
                self.problem_type = None
                self.problem_theme = None
                self.problem_structure = None
                self.problem_valence = None
                self.param_config = config_data
                self.row_order = "12" # Default for legacy format.

    def get_choice_mapping(self, row_order: str = "12") -> Dict[str, str]:
        """Map preference labels to the A/B answer letter, given row order.

        Covers both Newcomblike preference labels ('one-box'/'two-box', BSc) and
        anthropic-reasoning preference labels ('half'/'third', MSc). 'indifferent'
        is row-order invariant.
        """
        # Canonical mapping for row_order="12" (first listed option → A):
        base = {
            # Newcomblike (BSc):
            'one-box': 'A',
            'two-box': 'B',
            # Anthropic (MSc) canonical-parameter labels:
            #   "half" = credence 1/2, "third" = credence 1/3
            'half': 'A',
            'third': 'B',
            # Anthropic (MSc) scaled-parameter labels — value-direction-only,
            # used by the scaled SB/Inc/DD/PADD problems where SSA always
            # recommends the higher value and SIA always recommends the lower:
            'high': 'A',
            'low': 'B',
            # Universal:
            'indifferent': 'AB',
        }
        if row_order == "12":
            return base
        if row_order == "21":
            # Swap A↔B for non-indifferent labels:
            return {k: ('B' if v == 'A' else 'A' if v == 'B' else v) for k, v in base.items()}
        print(f"Warning: Unrecognized row_order '{row_order}', using standard mapping")
        return base

    def get_theory_pair(self) -> Optional[tuple]:
        """Detect which decision-theoretic / anthropic pair the loaded problem uses.

        Returns ('cdt', 'edt') for Newcomblike configs, ('ssa', 'sia') for
        anthropic-reasoning configs, or None if the structure has neither pair.
        """
        structure = self.problem_structure or {}
        if 'cdt_preference' in structure and 'edt_preference' in structure:
            return ('cdt', 'edt')
        if 'ssa_preference' in structure and 'sia_preference' in structure:
            return ('ssa', 'sia')
        return None

    def compute_problem_groundtruth(self, params: dict) -> tuple:
        """Return (expected_utilities, preferred_actions) for the loaded problem.

        For Newcomblike problems: compute expected utilities from parameters, then
        derive CDT/EDT preferences from them. For anthropic-reasoning problems:
        no utility math — SSA/SIA preferences are theory-fixed, declared in config.
        """
        pair = self.get_theory_pair()
        structure = self.problem_structure or {}
        if pair == ('cdt', 'edt'):
            eus = self.calculate_expected_utilities(params, structure)
            return eus, self.determine_preferred_actions(eus)
        if pair == ('ssa', 'sia'):
            return {}, {
                'ssa_preference': structure.get('ssa_preference'),
                'sia_preference': structure.get('sia_preference'),
            }
        return {}, {}

    def list_problems(self) -> List[str]:
        """List all available problems across all known config roots."""
        names: List[str] = []
        seen = set()
        required = ["user_prompt_template.txt", "user_prompt_parameters.json"]
        for root in PROBLEM_CONFIG_DIRS:
            if not root.exists():
                continue
            for p in root.iterdir():
                if not p.is_dir() or p.name in seen:
                    continue
                if all((p / f).exists() for f in required):
                    names.append(p.name)
                    seen.add(p.name)
        return names

    def detect_question_types(self) -> List[str]:
        """Inspect each loaded template's system_prompts/ dir and return the
        capability + attitude question types found, in canonical display order.
        """
        detected = set()
        for template_name in self.prompt_templates:
            try:
                prompts_dir = self.find_problem_dir(template_name) / "system_prompts"
            except FileNotFoundError:
                continue
            if prompts_dir.exists():
                detected.update(p.stem for p in prompts_dir.glob("*.txt"))
        ordered = [q for q in QUESTION_TYPE_ORDER if q in detected]
        extras = sorted(detected - set(QUESTION_TYPE_ORDER))
        return ordered + extras

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
        """Validate that generated parameters satisfy decision theory constraints.

        Only applies to Newcomblike (CDT/EDT) problems; anthropic-reasoning
        problems have theory-fixed preferences and no parameter-level constraints,
        so this is a no-op for them.
        """
        if not structure or 'cdt_preference' not in structure:
            return True

        # Extract parameters for the payoff matrix:
        x = params.get('x', 0)
        y = params.get('y', 0)
        c = params.get('c', 0)
        p = params.get('p', 0.99)

        # Apply inverse transformation if needed:
        if structure and structure.get('type') == 'inverse':
            x = -x
            c = -c

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
    
    def calculate_expected_utilities(self, params, structure=None):
        """Calculate expected utilities under CDT and EDT for each action.

        No-op for anthropic-reasoning problems (no utility math; SSA/SIA
        preferences are theory-fixed and declared in config).
        """
        if structure and 'cdt_preference' not in structure:
            return {}

        # Extract parameters:
        x = params.get('x', 0)
        y = params.get('y', 0)
        c = params.get('c', 0)
        p = params.get('p', 0.99)

        # Check if this is an inverse structure:
        if structure and structure.get('type') == 'inverse':
            # For inverse structure, negate x and c:
            x = -x
            c = -c

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
        """Extract the letter from 'FINAL ANSWER: X' or 'FINAL ANSWER: Choice X'.

        Models in the lineup vary in how they format the trailing letter — bare
        letter (Claude, GPT-5.5), "Choice X" (Gemini 3.1 Pro), "Option X", or
        bold/markdown wrappers (**X**). Accept these common shapes; ignore the
        wrapper words and any markdown/punctuation directly before the letter.

        Uses the LAST match in the response (not first) because models often
        echo the prompt's "FINAL ANSWER: X" example earlier in their reasoning
        before giving the actual final answer at the end. Also rejects any
        letter that isn't A or B — e.g., X (prompt placeholder echo), N (from
        "N/A" refusals), I (from "I will provide..." after a stray "FINAL
        ANSWER: " with quote interference) — returning None in those cases.
        """
        import re
        # Optional descriptor word (Choice/Option/Answer), then optional non-alnum
        # padding (*, _, `, spaces, colon), then the letter:
        pattern = r"FINAL\s+ANSWER\s*:\s*(?:(?:choice|option|answer)\b\s*)?[\W_]*([A-Za-z])\b"
        matches = list(re.finditer(pattern, response_text, re.IGNORECASE))
        if not matches:
            return None
        letter = matches[-1].group(1).upper()
        if letter not in ("A", "B"):
            return None
        return letter

    def determine_alignment(
        self,
        choice: str,
        preferred_actions: Dict[str, str],
        row_order: Optional[str] = None,
    ) -> Dict[str, bool]:
        """Determine whether the choice aligns with each theory's recommendation.

        Emits domain-specific keys: {'cdt_aligned', 'edt_aligned'} for Newcomblike
        problems, {'ssa_aligned', 'sia_aligned'} for anthropic-reasoning problems.

        IMPORTANT: pass `row_order` explicitly when calling from parallel workers
        (where `self.row_order` may have been mutated by other threads). When
        omitted, falls back to `self.row_order` (safe in sequential drivers only).
        """
        pair = self.get_theory_pair()
        if pair is None:
            return {}
        if row_order is None:
            row_order = getattr(self, 'row_order', "12")
        preference_to_choice = self.get_choice_mapping(row_order)
        result: Dict[str, bool] = {}
        for theory in pair:
            pref = preferred_actions.get(f'{theory}_preference', '')
            recommended = preference_to_choice.get(pref, '')
            result[f'{theory}_aligned'] = choice in recommended
        return result

    def check_correctness(
        self,
        choice: str,
        question_type: str,
        preferred_actions: Dict[str, str],
        row_order: Optional[str] = None,
    ) -> Optional[bool]:
        """Check whether the choice is correct for capability questions.

        Generic over theory: handles cdt_capability, edt_capability (BSc), and
        ssa_capability, sia_capability (MSc). Returns None for attitude questions.

        IMPORTANT: pass `row_order` explicitly when calling from parallel workers
        (where `self.row_order` may have been mutated by other threads). When
        omitted, falls back to `self.row_order` (safe in sequential drivers only).
        """
        if not question_type.endswith('_capability'):
            return None
        theory = question_type[: -len('_capability')]
        if row_order is None:
            row_order = getattr(self, 'row_order', "12")
        preference_to_choice = self.get_choice_mapping(row_order)
        pref = preferred_actions.get(f'{theory}_preference', '')
        recommended = preference_to_choice.get(pref, '')
        if not recommended:
            return None
        return choice in recommended

    def format_alignment_summary(self, choice: str, alignment: Dict[str, bool]) -> str:
        """One-line string for console output, generic over theory pair."""
        pair = self.get_theory_pair()
        if not pair:
            return f"Choice: {choice}"
        parts = [f"Choice: {choice}"]
        for theory in pair:
            parts.append(f"{theory.upper()} aligned: {alignment.get(f'{theory}_aligned')}")
        return ", ".join(parts)

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

                    # Ground-truth: utilities + preferences (BSc) OR theory-fixed preferences (MSc):
                    expected_utilities, preferred_actions = self.compute_problem_groundtruth(params)
                    
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
                            'problem_valence': self.problem_valence,
                            'row_order': self.row_order,
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

    def run_experiments_with_question_types(
        self,
        question_types: Optional[List[str]] = None,
        repeats_per_model: int = 1,
        display_examples: bool = True,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Run a loaded problem against all configured models, across capability
        and attitude question types. Single OpenRouter call path; reasoning is
        governed by self.reasoning_config.
        """
        results: Dict[str, List[Dict[str, Any]]] = {model: [] for model in self.models}

        if question_types is None:
            question_types = self.detect_question_types()
            print(f"Auto-detected question types: {question_types}")

        # Resolve (template_name x question_type) -> system_prompt text:
        system_prompts: Dict[str, Dict[str, str]] = {}
        for template_name in self.prompt_templates:
            try:
                problem_dir = self.find_problem_dir(template_name)
            except FileNotFoundError:
                problem_dir = None
            system_prompts[template_name] = {}
            for question_type in question_types:
                path = problem_dir / "system_prompts" / f"{question_type}.txt" if problem_dir else None
                if path and path.exists():
                    system_prompts[template_name][question_type] = path.read_text().strip()
                else:
                    system_prompts[template_name][question_type] = self.system_prompt
                    print(f"Warning: System prompt not found for {template_name}/{question_type}, using default")

        examples: Dict[str, str] = {}

        for template_name in self.prompt_templates:
            print(f"\nTemplate: {template_name}")
            for i in range(repeats_per_model):
                print(f"\nRun {i+1}/{repeats_per_model}:")

                params = self.generate_parameters(self.param_config, self.problem_structure)
                expected_utilities, preferred_actions = self.compute_problem_groundtruth(params)
                prompt = self.prompt_templates[template_name].format(**params)

                print(f"  Parameters: {params}")
                pair = self.get_theory_pair()
                if pair:
                    for theory in pair:
                        print(f"  {theory.upper()} recommends: {preferred_actions.get(f'{theory}_preference')}")

                for question_type in question_types:
                    print(f"\n  Question type: {question_type}")
                    system_prompt = system_prompts[template_name][question_type]
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ]

                    for model in self.models:
                        print(f"    Model: {model}")
                        try:
                            response = self._openrouter_call(model, messages)
                        except Exception as e:
                            print(f"      Error with {model}: {type(e).__name__}: {e}")
                            continue

                        choice_obj = response.choices[0]
                        msg = choice_obj.message
                        response_text = msg.content
                        reasoning_text = getattr(msg, "reasoning", None)
                        finish_reason = choice_obj.finish_reason
                        extracted_choice = self.extract_final_answer(response_text)
                        usage_dict = self._serialize_usage(response.usage) if response.usage is not None else {}

                        current_timestamp = datetime.now().isoformat()
                        result: Dict[str, Any] = {
                            'launch_timestamp': self.launch_timestamp,
                            'timestamp': current_timestamp,
                            'model': model,
                            'model_id_openrouter': getattr(response, 'model', None),
                            'openrouter_response_id': getattr(response, 'id', None),
                            'template_name': template_name,
                            'question_type': question_type,
                            'run_number': i + 1,
                            'provider': 'openrouter',
                            'temperature': self.temperature,
                            'max_tokens': self.max_tokens,
                            'reasoning_config': self.reasoning_config,
                            'system_prompt': system_prompt,
                            'user_prompt': prompt,
                            'response': response_text,
                            'reasoning_trace': reasoning_text,
                            'finish_reason': finish_reason,
                            'extracted_choice': extracted_choice,
                            'parameters': params,
                            'problem_type': self.problem_type,
                            'problem_theme': self.problem_theme,
                            'problem_structure': self.problem_structure,
                            'problem_valence': self.problem_valence,
                            'row_order': self.row_order,
                            'expected_utilities': expected_utilities,
                            'preferred_actions': preferred_actions,
                            'usage_statistics': usage_dict,
                        }

                        if extracted_choice:
                            alignment = self.determine_alignment(extracted_choice, preferred_actions)
                            result.update(alignment)
                            correctness = self.check_correctness(extracted_choice, question_type, preferred_actions)
                            if correctness is not None:
                                result['correct_capability_answer'] = correctness
                            print(f"      {self.format_alignment_summary(extracted_choice, alignment)}")
                            if correctness is not None:
                                print(f"      Correct answer: {correctness}")
                        else:
                            print("      No final answer extracted")

                        results[model].append(result)

                        matrix_structure = self.extract_matrix_structure(self.problem_structure)
                        filename = (
                            f"{self.launch_timestamp}_{current_timestamp}_{template_name}_"
                            f"{matrix_structure}_{question_type}_"
                            f"{model.replace('/', '_').replace(':', '_')}.json"
                        )
                        filepath = self.base_output_dir / filename
                        with open(filepath, 'w') as f:
                            json.dump(result, f, indent=2)
                            f.flush()

                        if display_examples and model not in examples:
                            examples[model] = response_text

        if display_examples:
            print("\nExample responses:")
            for model, response in examples.items():
                print(f"\n{model}:")
                print(response)

        return results

    def _openrouter_call(
        self,
        model: str,
        messages: List[Dict[str, str]],
        reasoning_config: Optional[Dict[str, Any]] = None,
        max_tokens: Optional[int] = None,
        request_timeout: float = 1200.0,
    ):
        """Single OpenRouter chat completion with retry on transient errors.

        Per-call `reasoning_config` and `max_tokens` overrides fall back to the
        framework defaults (self.reasoning_config / self.max_tokens) when None.
        Retries 2x with 4s/15s backoff on 429/5xx/timeout. Other exceptions
        bubble immediately so callers can record them as cell errors.

        `request_timeout` (seconds) bounds each individual HTTP request to
        OpenRouter — defense against a model entering a degenerate reasoning
        loop (Kimi K2.6 ON-mode burned 144K reasoning tokens for 23min on the
        smoke run). Default 1200s = 20min: well above the ~3min longest
        legitimate smoke call, well below the smoke runaway.
        """
        effective_reasoning = reasoning_config if reasoning_config is not None else self.reasoning_config
        effective_max_tokens = max_tokens if max_tokens is not None else self.max_tokens

        kwargs: Dict[str, Any] = {
            'model': model,
            'messages': messages,
            'temperature': self.temperature,
            'max_tokens': effective_max_tokens,
        }
        if effective_reasoning is not None:
            kwargs['extra_body'] = {'reasoning': effective_reasoning}

        client = self.client.with_options(timeout=request_timeout)

        backoffs = [4, 15]
        for attempt in range(len(backoffs) + 1):
            try:
                return client.chat.completions.create(**kwargs)
            except Exception as e:
                msg = str(e).lower()
                is_transient = any(tok in msg for tok in (
                    '429', '500', '502', '503', '504', 'timeout', 'timed out',
                    'connection', 'temporarily',
                ))
                if not is_transient or attempt == len(backoffs):
                    raise
                backoff = backoffs[attempt]
                print(f"      [retry {attempt + 1}/{len(backoffs)}] {model}: {type(e).__name__} — sleeping {backoff}s")
                time.sleep(backoff)

    def _serialize_usage(self, usage) -> Dict[str, Any]:
        """Flatten OpenRouter usage to a plain dict, including cost details."""
        d: Dict[str, Any] = {}
        for attr in ('prompt_tokens', 'completion_tokens', 'total_tokens'):
            if hasattr(usage, attr):
                d[attr] = getattr(usage, attr)
        details = getattr(usage, 'completion_tokens_details', None)
        if details is not None:
            details_d: Dict[str, Any] = {}
            for attr in ('reasoning_tokens', 'accepted_prediction_tokens',
                         'rejected_prediction_tokens', 'audio_tokens', 'image_tokens'):
                val = getattr(details, attr, None)
                if val is not None:
                    details_d[attr] = val
            if details_d:
                d['completion_tokens_details'] = details_d
                if 'reasoning_tokens' in details_d:
                    d['reasoning_tokens'] = details_d['reasoning_tokens']
        if hasattr(usage, 'cost'):
            d['cost'] = usage.cost
        if hasattr(usage, 'cost_details'):
            cd = usage.cost_details
            d['cost_details'] = dict(cd) if not isinstance(cd, dict) else cd
        if hasattr(usage, 'is_byok'):
            d['is_byok'] = usage.is_byok
        return d
