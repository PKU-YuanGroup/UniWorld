import json
import os
import argparse
from collections import defaultdict

def calculate_wiscore(consistency, realism, aesthetic_quality):
    """Calculates the WiScore based on given components."""
    return (0.7 * consistency + 0.2 * realism + 0.1 * aesthetic_quality) / 2

# Define expected prompt ID ranges at a global level for easy access
EXPECTED_PROMPT_RANGES = {
    "culture": range(1, 401),
    "space-time": range(401, 701), # Covers both TIME (401-567) and SPACE (568-700)
    "science": range(701, 1001), # Covers BIOLOGY (701-800), PHYSICS (801-900), CHEMISTRY (901-1000)
    "all": range(1, 1001) # Full range for combined evaluation
}

def process_jsonl_file_segment(file_path, category_arg=None):
    """
    Processes a segment of a JSONL file, collecting scores and present prompt_ids.
    Performs prompt_id validation if a specific category_arg is provided for a single file.
    Returns collected data or None if critical errors or missing prompt_ids (for single-file validation).
    """
    segment_scores = defaultdict(list)
    segment_present_prompt_ids = set()
    
    if not os.path.exists(file_path):
        print(f"Error: File '{file_path}' not found.")
        return None

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                try:
                    data = json.loads(line)
                    
                    prompt_id = data.get('prompt_id')
                    if prompt_id is None:
                        print(f"Warning: File '{file_path}', Line {line_num}: Missing 'prompt_id'. Skipping this line.")
                        continue
                    
                    if not isinstance(prompt_id, int):
                        print(f"Warning: File '{file_path}', Line {line_num}: 'prompt_id' is not an integer. Skipping this line.")
                        continue

                    segment_present_prompt_ids.add(prompt_id)
                    
                    consistency = data.get('consistency')
                    realism = data.get('realism')
                    aesthetic_quality = data.get('aesthetic_quality')

                    if not all(isinstance(val, (int, float)) for val in [consistency, realism, aesthetic_quality]):
                        print(f"Warning: File '{file_path}', Line {line_num}: One or more score values are not numeric. Skipping this line for category calculation.")
                        continue
                    
                    wiscore = calculate_wiscore(consistency, realism, aesthetic_quality)

                    # Determine category based on prompt_id
                    if 1 <= prompt_id <= 400:
                        segment_scores['CULTURE'].append(wiscore)
                    elif 401 <= prompt_id <= 567:
                        segment_scores['TIME'].append(wiscore)
                    elif 568 <= prompt_id <= 700:
                        segment_scores['SPACE'].append(wiscore)
                    elif 701 <= prompt_id <= 800:
                        segment_scores['BIOLOGY'].append(wiscore)
                    elif 801 <= prompt_id <= 900:
                        segment_scores['PHYSICS'].append(wiscore)
                    elif 901 <= prompt_id <= 1000:
                        segment_scores['CHEMISTRY'].append(wiscore)
                    else:
                        print(f"Warning: File '{file_path}', Line {line_num}: prompt_id {prompt_id} is outside defined categories. Skipping this line.")
                        continue

                except json.JSONDecodeError:
                    print(f"Warning: File '{file_path}', Line {line_num}: Invalid JSON format. Skipping this line.")
                except KeyError as e:
                    print(f"Warning: File '{file_path}', Line {line_num}: Missing expected key '{e}'. Skipping this line.")
    except Exception as e:
        print(f"Error reading file '{file_path}': {e}")
        return None
    
    # --- Single-file prompt_id validation logic ---
    if category_arg and category_arg != 'all' and category_arg in EXPECTED_PROMPT_RANGES:
        expected_ids_for_this_category = set(EXPECTED_PROMPT_RANGES[category_arg])
        missing_ids_in_segment = expected_ids_for_this_category - segment_present_prompt_ids

        if missing_ids_in_segment:
            print(f"Error: File '{file_path}': When evaluating as '--category {category_arg}', "
                  f"missing the following prompt_ids: {sorted(list(missing_ids_in_segment))}")
            return None # Return None if required prompt_ids are missing for a specific category file
    
    return {
        'scores': segment_scores,
        'present_prompt_ids': segment_present_prompt_ids,
        'file_path': file_path
    }

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate JSONL files for model performance, categorizing scores by prompt_id."
    )
    parser.add_argument(
        'files',
        metavar='FILE',
        nargs='+',  # Accepts one or more file paths
        help="Path(s) to the JSONL file(s) to be evaluated (e.g., cultural_common_sense_ModelName_scores.jsonl)"
    )
    parser.add_argument(
        '--category',
        type=str,
        choices=['culture', 'space-time', 'science', 'all'],
        default='all',
        help="Specify the category of the JSONL file(s) for specific prompt_id validation. Choose from 'culture', 'space-time', 'science', or 'all' (default). If evaluating a single category file, use the corresponding category."
    )
    
    args = parser.parse_args()
    
    all_raw_results = []
    
    # Process each file to collect raw scores and prompt IDs
    for file_path in args.files:
        print(f"\n--- Processing file: {file_path} ---")
        # Pass the category argument to process_jsonl_file_segment
        # This enables single-file validation logic
        results = process_jsonl_file_segment(file_path, args.category if len(args.files) == 1 else None)
        if results:
            all_raw_results.append(results)
        else:
            print(f"Could not process '{file_path}'. Please check previous warnings/errors.")

    if not all_raw_results:
        print("No valid data processed from any of the provided files. Exiting.")
        return # Exit if no files were successfully processed

    # Aggregate data across all successful files
    aggregated_scores = defaultdict(list)
    combined_present_prompt_ids = set()
    final_file_reports = {} # To store calculated averages/counts per file for individual display

    for file_data in all_raw_results:
        file_path = file_data['file_path']
        combined_present_prompt_ids.update(file_data['present_prompt_ids'])
        
        # Calculate scores for this individual file (for individual file report)
        current_file_avg_scores = {}
        current_file_num_samples = {}
        detected_categories_in_file = []

        for category, scores_list in file_data['scores'].items():
            aggregated_scores[category].extend(scores_list) # Aggregate for overall score later
            
            if scores_list: # Only add to individual file report if samples exist
                current_file_avg_scores[category] = sum(scores_list) / len(scores_list)
                current_file_num_samples[category] = len(scores_list)
                detected_categories_in_file.append(category)

        final_file_reports[file_path] = {
            'average': current_file_avg_scores,
            'num_processed_samples': current_file_num_samples,
            'detected_categories': detected_categories_in_file
        }

    # --- Step 1: Validate Prompt IDs for 'all' category scenario ---
    # This check happens only when --category all is explicitly chosen or is the default for multiple files.
    # Single-file specific category validation happens inside process_jsonl_file_segment.
    if args.category == 'all':
        expected_prompt_ids_for_all = set(EXPECTED_PROMPT_RANGES['all'])
        missing_prompt_ids_in_combined = expected_prompt_ids_for_all - combined_present_prompt_ids

        if missing_prompt_ids_in_combined:
            print(f"\nError: When '--category all' is specified, the combined files are missing the following prompt_ids:")
            print(f"Missing IDs: {sorted(list(missing_prompt_ids_in_combined))}")
            print("\nAborting overall evaluation due to incomplete data.")
            return # Exit if combined prompt IDs are missing when 'all' is expected

    # --- Step 2: Display individual file reports ---
    print("\n" + "="*50)
    print("                 Individual File Reports")
    print("="*50 + "\n")

    ordered_categories = ['CULTURE', 'TIME', 'SPACE', 'BIOLOGY', 'PHYSICS', 'CHEMISTRY']

    for file_path, file_data in final_file_reports.items():
        print(f"--- Evaluation Results for File: {file_path} ---")
        
        categories_to_print = sorted([cat for cat in ordered_categories if cat in file_data['detected_categories']],
                                     key=lambda x: ordered_categories.index(x))

        if not categories_to_print:
            print("  No scores found for any defined categories in this file.")
        else:
            for category in categories_to_print:
                avg_score = file_data['average'].get(category, 0)
                sample_count = file_data['num_processed_samples'].get(category, 0)
                print(f"  Category: {category}")
                print(f"    Average WiScore: {avg_score:.2f}")
                print(f"    Number of samples: {sample_count}\n")
        print("-" * (len(file_path) + 30) + "\n")

    # --- Step 3: Calculate and Display Overall Summary (if applicable) ---
    print("\n" + "="*50)
    print("                 Overall Evaluation Summary")
    print("="*50 + "\n")

    # Calculate overall averages from aggregated scores
    overall_avg_scores = {
        category: sum(scores) / len(scores) if len(scores) > 0 else 0
        for category, scores in aggregated_scores.items()
    }
    overall_num_samples = {
        category: len(scores)
        for category, scores in aggregated_scores.items()
    }

    # Print overall category scores (only for categories that have samples)
    overall_categories_to_print = sorted([cat for cat in ordered_categories if overall_num_samples.get(cat, 0) > 0],
                                          key=lambda x: ordered_categories.index(x))
    
    if not overall_categories_to_print and args.category != 'all':
        print("No valid scores found for any categories in the aggregated data.")
    else:
        print("Aggregated Category Scores:")
        for category in overall_categories_to_print:
            print(f"  Category: {category}")
            print(f"    Average WiScore: {overall_avg_scores.get(category, 0):.2f}")
            print(f"    Number of samples: {overall_num_samples.get(category, 0)}\n")

    # Calculate and print Overall WiScore if '--category all' was specified and all categories have samples
    all_categories_have_overall_samples = all(overall_num_samples.get(cat, 0) > 0 for cat in ordered_categories)
    
    if args.category == 'all' and all_categories_have_overall_samples:
        cultural_score = overall_avg_scores.get('CULTURE', 0)
        time_score = overall_avg_scores.get('TIME', 0)
        space_score = overall_avg_scores.get('SPACE', 0)
        biology_score = overall_avg_scores.get('BIOLOGY', 0)
        physics_score = overall_avg_scores.get('PHYSICS', 0)
        chemistry_score = overall_avg_scores.get('CHEMISTRY', 0)

        overall_wiscore = (0.4 * cultural_score + 0.167 * time_score + 0.133 * space_score +
                           0.1 * biology_score + 0.1 * physics_score + 0.1 * chemistry_score)
        
        print("\n--- Overall WiScore Across All Categories ---")
        print(f"Overall WiScore: {overall_wiscore:.2f}")
        print("Cultural\tTime\tSpace\tBiology\tPhysics\tChemistry\tOverall")
        print(f"{cultural_score:.2f}\t\t{time_score:.2f}\t{space_score:.2f}\t{biology_score:.2f}\t{physics_score:.2f}\t{chemistry_score:.2f}\t\t{overall_wiscore:.2f}")
    elif args.category == 'all' and not all_categories_have_overall_samples:
        print("\nOverall WiScore cannot be calculated: Not all categories have samples in the aggregated data when '--category all' is specified.")
    else:
        print(f"\nOverall WiScore calculation skipped. To calculate overall score, use '--category all' and provide files covering all prompt IDs.")


if __name__ == "__main__":
    main()