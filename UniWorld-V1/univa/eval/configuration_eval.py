from dataclasses import dataclass
from typing import Optional, List

@dataclass
class EvalConfig:
    pretrained_lvlm_name_or_path: str
    pretrained_denoiser_name_or_path: str
    pretrained_siglip_name_or_path: str

    ocr_enhancer: bool = False
    joint_with_t5: bool = False
    only_use_t5: bool = False

    seed: int = 42
    allow_tf32: bool = False

    output_dir: str = "./output"

    num_images_per_prompt: int = 1
    num_inference_steps: int = 32
    guidance_scale: float = 3.5  # Used in Flux
    num_samples_per_prompt: int = 1
    height: int = 1024
    width: int = 1024
    min_pixels: int = 448*448
    max_pixels: int = 448*448
    anyres: str = 'any_11ratio'
    padding_side: str = 'right'


    local_rank: int = 0
    world_size: int = 1

    # genai
    genai_prompt_path: str = "univa/eval/genai/eval_prompts/genai527/genai_image.json"

    # geneval
    n_samples: int = 4
    geneval_prompt_path: str = "univa/eval/geneval/evaluation_metadata.jsonl"
    resized_height: int = 1024
    resized_width: int = 1024

    # dpgbench
    dpgbench_prompt_path: str = "univa/eval/dpgbench/dpgbench_prompts.json"

    # wise
    wise_prompt_path: str = "univa/eval/wise/data"

    # imgedit
    imgedit_prompt_path: str = "univa/eval/imgedit/basic_edit.json"
    imgedit_image_dir: str = "/mnt/data/lb/Remake/imgedit_bench_eval_images"

    # gedit
    gedit_prompt_path: str = "univa/eval/gedit/gedit_edit.json"
    gedit_image_dir: str = "/mnt/data/lb/Remake/gedit_bench_eval_images"
