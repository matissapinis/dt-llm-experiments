# ./src/parse_results.py

import json
import re
import os
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

def extract_final_answer(response_text: str) -> Optional[str]:
    """Extract the 'FINAL ANSWER: X' from a response."""
    pattern = r"FINAL ANSWER:\s*([A-Za-z])"
    match = re.search(pattern, response_text)
    if match:
        return match.group(1).upper() # Normalize to uppercase.
    return None

def determine_alignment(choice: str, preferred_actions: Dict[str, str]) -> Dict[str, bool]:
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

def determine_theory_choices(expected_utilities: Dict[str, float]) -> Dict[str, str]:
    """Determine which choices each decision theory recommends."""
    eu_cdt_a1 = expected_utilities.get('eu_cdt_one_box', 0)
    eu_cdt_a2 = expected_utilities.get('eu_cdt_two_box', 0)
    eu_edt_a1 = expected_utilities.get('eu_edt_one_box', 0)
    eu_edt_a2 = expected_utilities.get('eu_edt_two_box', 0)
    
    # Small epsilon for floating point comparison:
    epsilon = 0.001
    
    # Determine CDT choice:
    if abs(eu_cdt_a1 - eu_cdt_a2) < epsilon:
        cdt_choice = "AB" # Indifferent.
    elif eu_cdt_a1 > eu_cdt_a2:
        cdt_choice = "A"  # One-box.
    else:
        cdt_choice = "B"  # Two-box.
    
    # Determine EDT choice:
    if abs(eu_edt_a1 - eu_edt_a2) < epsilon:
        edt_choice = "AB" # Indifferent.
    elif eu_edt_a1 > eu_edt_a2:
        edt_choice = "A"  # One-box.
    else:
        edt_choice = "B"  # Two-box.
    
    return {
        'cdt_recommended_choice': cdt_choice,
        'edt_recommended_choice': edt_choice
    }

def create_calculation_formulas(params: Dict[str, Any]) -> Dict[str, str]:
    """Create human-readable formula strings for expected utilities."""
    # Extract parameters:
    x = params.get('x', 0)
    y = params.get('y', 0)
    c = params.get('c', 0)
    p = params.get('p', params.get('p', 0.99))

    # Derive payoff values for two-boxing:
    z = x + c
    w = y + c
    
    # CDT formulas (causal expected utility):
    cdt_one_box_formula = f"0.5 * {x} + 0.5 * {y} = {0.5 * x + 0.5 * y}"
    cdt_two_box_formula = f"0.5 * {z} + 0.5 * {w} = {0.5 * z + 0.5 * w}"
    
    # EDT formulas (evidential expected utility):
    edt_one_box_formula = f"{p} * {x} + (1 - {p}) * {y} = {p * x + (1 - p) * y}"
    edt_two_box_formula = f"(1 - {p}) * {z} + {p} * {w} = {(1 - p) * z + p * w}"
    
    return {
        'cdt_one_box_formula': cdt_one_box_formula,
        'cdt_two_box_formula': cdt_two_box_formula,
        'edt_one_box_formula': edt_one_box_formula,
        'edt_two_box_formula': edt_two_box_formula
    }

def check_correctness(choice: str, question_type: str, preferred_actions: Dict[str, str]) -> Optional[bool]:
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

def parse_experiment_results(results_dir: str = "experiment_results", output_dir: str = "parsed_results") -> None:
    """Parse all experiment result files, adding analysis data."""
    # Create output directory if it doesn't exist:
    Path(output_dir).mkdir(exist_ok=True)
    
    # Find all JSON files in results directory:
    result_files = list(Path(results_dir).glob("*.json"))
    
    if not result_files:
        print(f"No result files found in {results_dir}")
        return
    
    print(f"Found {len(result_files)} result files to parse.")
    
    # Counters for analytics:
    total_parsed = 0
    total_with_final_answer = 0
    question_type_counts = {}
    alignment_counts = {"cdt_only": 0, "edt_only": 0, "both": 0, "neither": 0}
    correctness_counts = {"correct": 0, "incorrect": 0, "not_applicable": 0}
    preserved_annotations = 0
    
    # Process each file:
    for result_file in result_files:
        try:
            # Load the original result:
            with open(result_file, "r") as f:
                result = json.load(f)
            
            # Check if an already-parsed version exists with manual annotation:
            output_file = Path(output_dir) / result_file.name
            existing_annotation = None
            annotation_metadata = {}
            
            if output_file.exists():
                try:
                    with open(output_file, "r") as f:
                        existing_result = json.load(f)
                    
                    # Check if this was manually annotated:
                    if existing_result.get('manually_annotated', False):
                        existing_annotation = existing_result.get('extracted_choice')
                        # Preserve all annotation metadata:
                        annotation_metadata = {
                            'manually_annotated': existing_result.get('manually_annotated'),
                            'annotation_timestamp': existing_result.get('annotation_timestamp'),
                            'choice_not_applicable': existing_result.get('choice_not_applicable', False),
                            'cdt_aligned': existing_result.get('cdt_aligned'),
                            'edt_aligned': existing_result.get('edt_aligned'),
                            'correct_capability_answer': existing_result.get('correct_capability_answer')
                        }
                        preserved_annotations += 1
                except Exception as e:
                    print(f"Warning: Error reading existing parsed file {output_file.name}: {e}")
            
            # Extract basic information:
            response_text = result.get('response', '')
            question_type = result.get('question_type', 'unknown')
            preferred_actions = result.get('preferred_actions', {})
            parameters = result.get('parameters', {})
            expected_utilities = result.get('expected_utilities', {})
            
            # Track question types:
            if question_type not in question_type_counts:
                question_type_counts[question_type] = 0
            question_type_counts[question_type] += 1
            
            # Add recommended choices:
            if expected_utilities:
                theory_choices = determine_theory_choices(expected_utilities)
                result.update(theory_choices)
            
            # Add calculation formulas:
            if parameters:
                formulas = create_calculation_formulas(parameters)
                result['calculation_formulas'] = formulas
            
            # If there's an existing manual annotation, preserve it:
            if existing_annotation:
                result['extracted_choice'] = existing_annotation
                # Restore all annotation metadata:
                for key, value in annotation_metadata.items():
                    if value is not None:  # Only copy non-None values
                        result[key] = value
            else:
                # Otherwise, extract final answer from the response:
                final_answer = extract_final_answer(response_text)
                result['extracted_choice'] = final_answer
                
                if final_answer:
                    total_with_final_answer += 1
                    
                    # Determine alignment with CDT and EDT:
                    alignment = determine_alignment(final_answer, preferred_actions)
                    result['cdt_aligned'] = alignment['cdt_aligned']
                    result['edt_aligned'] = alignment['edt_aligned']
                    
                    # Track alignment stats:
                    if alignment['cdt_aligned'] and alignment['edt_aligned']:
                        alignment_counts["both"] += 1
                    elif alignment['cdt_aligned']:
                        alignment_counts["cdt_only"] += 1
                    elif alignment['edt_aligned']:
                        alignment_counts["edt_only"] += 1
                    else:
                        alignment_counts["neither"] += 1
                    
                    # Check correctness for capability questions:
                    correctness = check_correctness(final_answer, question_type, preferred_actions)
                    if correctness is not None:
                        result['correct_capability_answer'] = correctness
                        if correctness:
                            correctness_counts["correct"] += 1
                        else:
                            correctness_counts["incorrect"] += 1
                    else:
                        correctness_counts["not_applicable"] += 1
            
            # Save the parsed result:
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)
            
            total_parsed += 1
            
        except Exception as e:
            print(f"Error processing {result_file.name}: {e}")
    
    # Print summary
    print(f"\nParsing complete! {total_parsed} files processed.")
    if preserved_annotations > 0:
        print(f"Preserved manual annotations for {preserved_annotations} files.")
    print(f"Files with extractable final answers: {total_with_final_answer} ({total_with_final_answer/total_parsed*100:.1f}%)")
    
    print("\nQuestion Type Distribution:")
    for qtype, count in question_type_counts.items():
        print(f"  {qtype}: {count} files")
    
    print("\nDecision Theory Alignment:")
    for alignment, count in alignment_counts.items():
        if total_with_final_answer > 0:
            print(f"  {alignment}: {count} ({count/total_with_final_answer*100:.1f}%)")
    
    print("\nCapability Question Correctness:")
    capability_questions = correctness_counts["correct"] + correctness_counts["incorrect"]
    if capability_questions > 0:
        print(f"  Correct: {correctness_counts['correct']} ({correctness_counts['correct']/capability_questions*100:.1f}%)")
        print(f"  Incorrect: {correctness_counts['incorrect']} ({correctness_counts['incorrect']/capability_questions*100:.1f}%)")
    print(f"  Not applicable: {correctness_counts['not_applicable']} (attitude questions)")

if __name__ == "__main__":
    parse_experiment_results()