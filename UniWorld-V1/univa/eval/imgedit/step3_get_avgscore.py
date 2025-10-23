import json
import argparse
from collections import defaultdict

# Code1: Calculate average score for each key
def extract_scores_and_average(entry: str) -> float:
    lines = entry.splitlines()
    scores = []
    for line in lines:
        parts = line.strip().split(': ')
        if len(parts) == 2 and parts[1].isdigit():
            scores.append(int(parts[1]))
    if scores:
        return round(sum(scores) / len(scores), 2)
    return None

def compute_averages(input_dict):
    result = {}
    for key, value in input_dict.items():
        avg = extract_scores_and_average(value)
        if avg is not None:
            result[key] = avg
    return result

# Code2: Compute edit type averages
def compute_edit_type_averages(score_dict, meta_dict):
    edit_type_scores = defaultdict(list)

    for key, score in score_dict.items():
        meta = meta_dict.get(key, {})
        edit_type = meta.get("edit_type")
        if edit_type is not None:
            edit_type_scores[edit_type].append(score)

    averaged_by_type = {
        etype: round(sum(scores) / len(scores), 2)
        for etype, scores in edit_type_scores.items() if scores
    }
    return averaged_by_type

def main():
    parser = argparse.ArgumentParser(description="Calculate averages based on the input json")
    parser.add_argument('--input', type=str, required=True, help='Path of input json (with scores)')
    parser.add_argument('--meta_json', type=str, required=True, help='Path of meta json (with edit_type info)')
    parser.add_argument('--output_json', type=str, required=True, help='Path of output json (averaged scores by edit_type)')

    args = parser.parse_args()

    # Step 1: Load the input JSON and calculate average scores
    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    averaged_data = compute_averages(data)

    # Step 2: Load the meta JSON and calculate edit type averages
    with open(args.meta_json, 'r', encoding='utf-8') as f:
        meta_data = json.load(f)

    averaged_result = compute_edit_type_averages(averaged_data, meta_data)

    # Step 3: Save the result into output JSON
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(averaged_result, f, indent=2)

if __name__ == '__main__':
    main()
