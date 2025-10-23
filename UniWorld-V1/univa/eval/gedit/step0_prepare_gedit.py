import json
import os
import math
import argparse
from datasets import Dataset, load_dataset

# Dataset info structure:
# - task_type: string - Type of the task
# - key: string - Unique identifier for the sample
# - instruction: string - Task instruction/prompt
# - instruction_language: string - Language of the instruction
# - input_image: Image - Original input image
# - input_image_raw: Image - Raw/unprocessed input image
# - Intersection_exist: bool - Whether intersection exists

def calculate_dimensions(target_area, ratio):
    width = math.sqrt(target_area * ratio)
    height = width / ratio
    
    width = round(width / 32) * 32
    height = round(height / 32) * 32
    
    new_area = width * height
    if new_area < target_area:
        width += 32
        new_area = width * height
    elif new_area > target_area:
        width -= 32
        new_area = width * height
    
    return width, height, new_area

def main(args):
    # Load dataset
    dataset = load_dataset("stepfun-ai/GEdit-Bench")

    # Dictionary to store instruction and image paths
    instruction_image_paths = {}

    for item in dataset['train']:
        task_type = item['task_type']
        key = item['key']
        instruction = item['instruction']
        instruction_language = item['instruction_language']
        input_image = item['input_image']
        input_image_raw = item['input_image_raw']
        intersection_exist = item['Intersection_exist']

        target_width, target_height, new_area = calculate_dimensions(512 * 512, input_image_raw.width / input_image_raw.height)
        resize_input_image = input_image_raw.resize((target_width, target_height))

        save_path_fullset_source_image = os.path.join(args.save_path, f"fullset/{task_type}/{instruction_language}/{key}_SRCIMG.png")
        save_path_fullset = os.path.join(args.save_path, f"fullset/{task_type}/{instruction_language}/{key}.png")

        relative_path = f"fullset/{task_type}/{instruction_language}/{key}.png"

        # Create directories if they don't exist
        os.makedirs(os.path.dirname(save_path_fullset_source_image), exist_ok=True)
        os.makedirs(os.path.dirname(save_path_fullset), exist_ok=True)

        # Save the images
        input_image.save(save_path_fullset_source_image)
        resize_input_image.save(save_path_fullset)

        # Store instruction and corresponding image path in the dictionary
        instruction_image_paths[key] = {
            'prompt': instruction,
            'id': relative_path,
            'edit_type':  task_type,
        }

    # Save the dictionary to a JSON file
    with open(args.json_file_path, 'w') as json_file:
        json.dump(instruction_image_paths, json_file, indent=4)

    print(f"Instruction and image paths saved to {args.json_file_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process and save dataset images and instructions.")
    parser.add_argument("--save_path", type=str, required=True, help="Directory to save processed images.")
    parser.add_argument("--json_file_path", type=str, required=True, help="Path to save the JSON file with instruction-image mappings.")

    args = parser.parse_args()

    main(args)
