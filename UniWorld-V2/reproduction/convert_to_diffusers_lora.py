import torch
from safetensors.torch import load_file, save_file

base_dir = "lora/"

state_dict = load_file(f"{base_dir}/adapter_model.safetensors")

new_state_dict = {}

for key, value in state_dict.items():
    new_key = key.replace("base_model.model", "transformer")
    new_state_dict[new_key] = value


save_file(new_state_dict, f"{base_dir}/adapter_model_converted.safetensors")
