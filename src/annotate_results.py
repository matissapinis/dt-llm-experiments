# ./src/annotate_results.py

"""
1. List files with missing choices:
python3 src/annotate_results.py --list

2. Review a response (to help decide the choice):
python3 src/annotate_results.py --show 2025-05-02T19:36:02.515840_2025-05-02T19:36:34.460361_newcomb_money_v2_standard_two-box-one-box_normative_attitude_openai_chatgpt-4o-latest.json

3. Update a file with your manual choice:
python3 src/annotate_results.py 2025-05-02T19:36:02.515840_2025-05-02T19:36:34.460361_newcomb_money_v2_standard_two-box-one-box_normative_attitude_openai_chatgpt-4o-latest.json A

4. Mark a response as not applicable (misunderstanding, ambiguous, refusal, abrupt completion etc.):
python3 src/annotate_results.py 2025-05-02T19:49:19.550295_2025-05-02T19:50:20.303547_newcomb_money_v2_standard_two-box-one-box_cdt_capability_openai_chatgpt-4o-latest.json N/A 

5. Interactive batch annotation (process all files needing annotation):
python3 src/annotate_results.py --annotate-all

This script allows:
- Finding files where automatic extraction failed.
- Reviewing the full response text.
- Assigning choices manually.
- Marking responses as "N/A" when neither A nor B applies.
- Updating associated analysis (alignment, correctness) with manual annotation flag and timestamp.
- Interactive batch annotation to quickly process all files potentially needing annotation.
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

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
    cdt_one_box_formula = f"EU_CDT(A) = 0.5 * {x} + 0.5 * {y} = {0.5 * x + 0.5 * y}"
    cdt_two_box_formula = f"EU_CDT(B) = 0.5 * {z} + 0.5 * {w} = {0.5 * z + 0.5 * w}"
    
    # EDT formulas (evidential expected utility):
    edt_one_box_formula = f"EU_EDT(A) = {p} * {x} + (1 - {p}) * {y} = {p * x + (1 - p) * y}"
    edt_two_box_formula = f"EU_EDT(B) = (1 - {p}) * {z} + {p} * {w} = {(1 - p) * z + p * w}"
    
    return {
        'cdt_one_box_formula': cdt_one_box_formula,
        'cdt_two_box_formula': cdt_two_box_formula,
        'edt_one_box_formula': edt_one_box_formula,
        'edt_two_box_formula': edt_two_box_formula
    }

def list_missing_choices(parsed_dir: str = "parsed_results") -> List[str]:
    """List all files missing an extracted choice."""
    missing_files = []
    
    # Find all JSON files in parsed directory:
    parsed_files = list(Path(parsed_dir).glob("*.json"))
    
    for file_path in parsed_files:
        try:
            with open(file_path, "r") as f:
                result = json.load(f)
                
            if result.get('extracted_choice') is None:
                missing_files.append(file_path.name)
        except Exception as e:
            print(f"Error processing {file_path.name}: {e}")
    
    return missing_files

def update_file_with_choice(filename: str, choice: str, parsed_dir: str = "parsed_results") -> None:
    """Update a specific file with a manually assigned choice."""
    file_path = Path(parsed_dir) / filename
    
    if not file_path.exists():
        print(f"Error: File {filename} not found in {parsed_dir}")
        return
    
    try:
        # Load the file:
        with open(file_path, "r") as f:
            result = json.load(f)
        
        # Update the choice:
        choice = choice.upper() # Normalize to uppercase.
        result['extracted_choice'] = choice
        
        # Add manual annotation tracking:
        annotation_timestamp = datetime.now().isoformat()
        result['manually_annotated'] = True
        result['annotation_timestamp'] = annotation_timestamp
        
        # Update alignment and correctness only if it's a valid A/B choice (not N/A):
        preferred_actions = result.get('preferred_actions', {})
        question_type = result.get('question_type', 'unknown')
        
        if choice == "N/A":
            # For N/A choices, set special flags:
            result['choice_not_applicable'] = True
            
            # Remove any existing alignment or correctness values:
            if 'cdt_aligned' in result:
                del result['cdt_aligned']
            if 'edt_aligned' in result:
                del result['edt_aligned']
            if 'correct_capability_answer' in result:
                del result['correct_capability_answer']
                
            print(f"Updated {filename} with choice N/A (Not Applicable)")
            print(f"  Manually annotated at: {annotation_timestamp}")
            print("  Alignment and correctness evaluations not applicable")
        else:
            # For A or B choices, determine alignment with CDT and EDT:
            result['choice_not_applicable'] = False
            
            if preferred_actions:
                # Map preferences to choices:
                preference_to_choice = {
                    'one-box': 'A',
                    'two-box': 'B',
                    'indifferent': 'AB' # Both A and B are acceptable for indifferent.
                }
                
                cdt_preference = preferred_actions.get('cdt_preference', '')
                edt_preference = preferred_actions.get('edt_preference', '')
                
                cdt_recommended = preference_to_choice.get(cdt_preference, '')
                edt_recommended = preference_to_choice.get(edt_preference, '')
                
                result['cdt_aligned'] = choice in cdt_recommended
                result['edt_aligned'] = choice in edt_recommended
                
                # For capability questions, check correctness:
                if question_type == 'cdt_capability':
                    result['correct_capability_answer'] = choice in cdt_recommended
                elif question_type == 'edt_capability':
                    result['correct_capability_answer'] = choice in edt_recommended
            
            print(f"Updated {filename} with choice {choice}")
            print(f"  Manually annotated at: {annotation_timestamp}")
            if preferred_actions:
                print(f"  CDT aligned: {result.get('cdt_aligned', 'N/A')}")
                print(f"  EDT aligned: {result.get('edt_aligned', 'N/A')}")
                if 'correct_capability_answer' in result:
                    print(f"  Correct capability answer: {result['correct_capability_answer']}")
        
        # Save the updated file:
        with open(file_path, "w") as f:
            json.dump(result, f, indent=2)
        
    except Exception as e:
        print(f"Error updating {filename}: {e}")

def show_response(filename: str, parsed_dir: str = "parsed_results") -> None:
    """Show comprehensive information for a specific file to aid in manual review."""
    file_path = Path(parsed_dir) / filename
    
    if not file_path.exists():
        print(f"Error: File {filename} not found in {parsed_dir}")
        return
    
    try:
        # Load the file:
        with open(file_path, "r") as f:
            result = json.load(f)
        
        # Get model and question type info for the header:
        model = result.get('model', 'Unknown model')
        question_type = result.get('question_type', 'Unknown question type')
        template_name = result.get('template_name', 'Unknown template')
        
        # Print header:
        print("\n" + "=" * 80)
        print(f"REVIEW: {template_name} | {question_type} | {model}")
        print("=" * 80)
        
        # Print system prompt:
        print("\nSYSTEM PROMPT:")
        print("-" * 80)
        print(result.get('system_prompt', 'No system prompt found in file'))
        
        # Print user prompt:
        print("\nUSER PROMPT:")
        print("-" * 80)
        print(result.get('user_prompt', 'No user prompt found in file'))
        
        # Print model response:
        print("\nMODEL RESPONSE:")
        print("-" * 80)
        print(result.get('response', 'No response found in file'))
        
        # Print reasoning content if available:
        if 'reasoning' in result and result['reasoning']:
            print("\nMODEL REASONING:")
            print("-" * 80)
            print(result['reasoning'])
            
            # Print token usage info if available:
            if 'usage_statistics' in result and result['usage_statistics']:
                usage = result['usage_statistics']
                print("\nTOKEN USAGE:")
                print("-" * 80)
                for key, value in usage.items():
                    if value is not None: # Only print non-None values.
                        print(f"{key}: {value}")
        
        # Print decision theory recommendations:
        print("\nDECISION THEORY ANALYSIS:")
        print("-" * 80)
        
        # Get preferred actions:
        preferred_actions = result.get('preferred_actions', {})
        cdt_preference = preferred_actions.get('cdt_preference', 'unknown')
        edt_preference = preferred_actions.get('edt_preference', 'unknown')
        
        # Map preferences to choices:
        preference_to_choice = {
            'one-box': 'A (one-box)',
            'two-box': 'B (two-box)',
            'indifferent': 'Either A or B (indifferent)'
        }
        
        cdt_recommended = preference_to_choice.get(cdt_preference, 'unknown')
        edt_recommended = preference_to_choice.get(edt_preference, 'unknown')
        
        print(f"CDT recommends: {cdt_recommended}")
        print(f"EDT recommends: {edt_recommended}")
        
        # Extract existing choice if available:
        extracted_choice = result.get('extracted_choice', None)
        if extracted_choice:
            if extracted_choice == "N/A":
                print(f"Extracted choice: N/A (Not Applicable)")
            else:
                print(f"Extracted choice: {extracted_choice}")
            
            # Show annotation information if available:
            if result.get('manually_annotated', False):
                print(f"Manually annotated at: {result.get('annotation_timestamp', 'unknown')}")
            
            # Show N/A status if applicable:
            if result.get('choice_not_applicable', False):
                print("Status: Not applicable for alignment or correctness evaluation")
        else:
            print("No choice extracted (requires manual annotation)")
            print("Annotation options: A, B, or N/A (for ambiguous/refusal cases)")
        
        # Print EU formulas:
        print("\nEXPECTED UTILITY FORMULAS:")
        print("-" * 80)
        
        # Get parameters:
        params = result.get('parameters', {})
        if params:
            # Calculate formulas:
            formulas = create_calculation_formulas(params)
            
            # Print formulas:
            print(f"CDT one-box:  {formulas['cdt_one_box_formula']}")
            print(f"CDT two-box:  {formulas['cdt_two_box_formula']}")
            print(f"EDT one-box:  {formulas['edt_one_box_formula']}")
            print(f"EDT two-box:  {formulas['edt_two_box_formula']}")
            
            # Print prediction accuracy:
            print(f"\nPrediction accuracy: {params.get('p', 0):.2f}")
        else:
            print("No parameters found in file, cannot calculate formulas")
        
        print("\n" + "=" * 80)
        print("END OF REVIEW")
        print("=" * 80)
        
    except Exception as e:
        print(f"Error reading {filename}: {e}")

def annotate_all_files(parsed_dir: str = "parsed_results") -> None:
    """Interactive batch annotation for all files missing extracted choices."""
    # Get files needing annotation:
    missing_files = list_missing_choices(parsed_dir)
    
    if not missing_files:
        print("No files found that need annotation.")
        return
    
    total_files = len(missing_files)
    processed = 0
    skipped = 0
    annotated = 0
    
    print(f"\nStarting batch annotation of {total_files} files...")
    print("For each file, you can:")
    print("  - Enter 'A' or 'B' to annotate with that choice")
    print("  - Enter 'N/A' for responses that cannot be classified")
    print("  - Enter 'S' or 'skip' to skip this file for now")
    print("  - Enter 'Q' or 'quit' to exit batch annotation\n")
    
    try:
        for i, filename in enumerate(missing_files, 1):
            print(f"\nFile {i} of {total_files}: {filename}")
            
            # Show the file contents:
            show_response(filename, parsed_dir)
            
            # Prompt for action:
            while True:
                choice = input("\nEnter annotation (A/B/N/A), skip (S), or quit (Q): ").strip().upper()
                
                if choice in ['A', 'B', 'N/A']:
                    update_file_with_choice(filename, choice, parsed_dir)
                    annotated += 1
                    processed += 1
                    break
                elif choice in ['S', 'SKIP']:
                    print(f"Skipped: {filename}")
                    skipped += 1
                    processed += 1
                    break
                elif choice in ['Q', 'QUIT']:
                    print("\nBatch annotation terminated early.")
                    print(f"Progress: {processed}/{total_files} files processed")
                    print(f"  - {annotated} files annotated")
                    print(f"  - {skipped} files skipped")
                    print(f"  - {total_files - processed} files remaining")
                    return
                else:
                    print("Invalid choice. Please enter A, B, N/A, S (skip), or Q (quit).")
        
        print("\nBatch annotation complete!")
        print(f"All {total_files} files processed:")
        print(f"  - {annotated} files annotated")
        print(f"  - {skipped} files skipped")
        
    except KeyboardInterrupt:
        # Handle process interrupt signal gracefully:
        print("\n\nBatch annotation interrupted.")
        print(f"Progress: {processed}/{total_files} files processed")
        print(f"  - {annotated} files annotated")
        print(f"  - {skipped} files skipped")
        print(f"  - {total_files - processed} files remaining")

def main():
    if len(sys.argv) == 1 or sys.argv[1] == "--list":
        # List all files missing an extracted choice:
        missing_files = list_missing_choices()
        if missing_files:
            print(f"Found {len(missing_files)} files missing extracted choices:")
            for i, filename in enumerate(missing_files, 1):
                print(f"{i}. {filename}")
        else:
            print("No files missing extracted choices found.")
        return
    
    if sys.argv[1] == "--show" and len(sys.argv) >= 3:
        # Show comprehensive information for a specific file:
        show_response(sys.argv[2])
        return
    
    if sys.argv[1] == "--annotate-all":
        # Interactive batch annotation mode:
        annotate_all_files()
        return
    
    if len(sys.argv) == 3:
        # Update a specific file with a manually assigned choice:
        filename = sys.argv[1]
        choice = sys.argv[2]
        update_file_with_choice(filename, choice)
        return
    
    # Print usage if arguments don't match expected patterns:
    print("Usage:")
    print("  python annotate_results.py --list                # List files missing extracted choices")
    print("  python annotate_results.py --show FILENAME       # Show response for manual review")
    print("  python annotate_results.py --annotate-all        # Interactive batch annotation mode")
    print("  python annotate_results.py FILENAME CHOICE       # Update file with manual choice (A/B/N/A)")

if __name__ == "__main__":
    main()