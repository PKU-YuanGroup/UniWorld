import argparse
import os
import json
import torch
import ray
from diffusers import FluxKontextPipeline
from diffusers.utils import load_image
from tqdm import tqdm

@ray.remote(num_gpus=1)
def process_slice(slice_items, pretrained_name_or_path, lora_path, output_dir, root_path, seed):
    pipe = FluxKontextPipeline.from_pretrained(
        pretrained_name_or_path,
        torch_dtype=torch.bfloat16,
    )
    pipe.to("cuda")
    
    if lora_path:
        print("Load lora", lora_path)
        pipe.load_lora_weights(
            lora_path,
            weight_name="adapter_model.safetensors",
            adapter_name="lora",
        )
        pipe.set_adapters(["lora"], adapter_weights=[1])

    for key, item in tqdm(slice_items):
        try:
            relative_image_path = item["id"]
            prompt = item["prompt"]
            absolute_image_path = os.path.normpath(os.path.join(root_path, relative_image_path))
            output_filename = f"{key}.jpg"
            output_filepath = os.path.join(output_dir, output_filename)
            if os.path.exists(output_filepath):
                continue

            input_image = load_image(absolute_image_path)
            generator = torch.Generator(device="cuda").manual_seed(seed)
            output_image = pipe(
                num_inference_steps=28,
                image=input_image,
                prompt=prompt,
                generator=generator,
            ).images[0]

            output_image.save(output_filepath)
        except Exception as e:
            continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="output_images")
    parser.add_argument("--pretrained_name_or_path", type=str, required=True)
    parser.add_argument("--root_path", type=str, required=True)
    parser.add_argument("--lora_path", type=str, required=False, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    root_path = os.path.normpath(args.root_path)
    ray.init()
    gpu_count = int(ray.available_resources().get("GPU", 1))

    def load_json(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            exit(1)

    ds = load_json(args.input_path)
    all_items = list(ds.items())

    slices = [all_items[i::gpu_count] for i in range(gpu_count)]
    ray.get([
        process_slice.remote(
            slices[i],
            args.pretrained_name_or_path,
            args.lora_path,
            args.output_dir,
            root_path,
            args.seed,
        ) for i in range(gpu_count)
    ])
   