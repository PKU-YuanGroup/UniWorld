from datasets import load_from_disk
from diffusers import QwenImageEditPlusPipeline, QwenImageEditPipeline, QwenImageTransformer2DModel
import ray
import torch
from typing import Optional
import argparse
import os
from tqdm import tqdm

def load_pipeline(pretrained_name_or_path: str, lora_path: Optional[str] = None):
    pipeline = QwenImageEditPlusPipeline.from_pretrained(
        pretrained_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    pipeline.to("cuda")

    if lora_path:
        pipeline.load_lora_weights(
            lora_path,
            weight_name="adapter_model.safetensors",
            adapter_name="lora",
        )
        pipeline.set_adapters(["lora"], adapter_weights=[1])
        print("Lora path provided")
    else:
        print("No lora path provided, using origin model")
    return pipeline


@ray.remote(num_gpus=1)
def sample(sliced_data, pretrained_name_or_path, lora_path, output_dir, seed):
    pipeline = load_pipeline(pretrained_name_or_path, lora_path)
    for item in tqdm(sliced_data):
        if item["instruction_language"] != "en":
            continue
        key = item["key"]
        prompt = item["instruction"]
        task_type = item["task_type"]
        input_image = item["input_image_raw"].convert("RGB")
        image_output_dir = os.path.join(output_dir, task_type)

        if os.path.exists(os.path.join(image_output_dir, f"{key}.png")):
            continue

        os.makedirs(image_output_dir, exist_ok=True)

        width, height = input_image.size

        generator = torch.Generator(device="cuda").manual_seed(seed)

        output_image = pipeline(
            prompt=prompt,
            true_cfg_scale=4.0,
            guidance_scale=1.0,
            image=input_image,
            negative_prompt=" ",
            num_inference_steps=28,
            generator=generator,
        ).images[0]

        output_image.save(f"{image_output_dir}/{key}.png")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_name_or_path", type=str, default="/mnt/data/checkpoints/")
    parser.add_argument("--gedit_bench_path", type=str, default="/mnt/data/datasets/GEdit-Bench")
    parser.add_argument("--lora_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/no_name",
        help="path to save the output images",
    )
    args = parser.parse_args()
    pretrained_name_or_path = args.pretrained_name_or_path
    lora_path = args.lora_path
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    ray.init()
    os.makedirs(args.output_dir, exist_ok=True)
    dataset = load_from_disk(args.gedit_bench_path)
    gpu_count = int(ray.available_resources().get("GPU", 1))
    print(f"GPU count: {gpu_count}")
    ray.get(
        [
            sample.remote(
                dataset.select(range(i, len(dataset), gpu_count)),
                pretrained_name_or_path,
                lora_path,
                args.output_dir,
                seed,
            )
            for i in range(gpu_count)
        ]
    )

if __name__ == "__main__":
    main()
