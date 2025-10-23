import sys

sys.path.append(".")
from univa.models.qwen2vl.configuration_univa_qwen2vl import UnivaQwen2VLConfig
from univa.models.configuration_univa_denoise_tower import UnivaDenoiseTowerConfig
from univa.models.qwen2vl.modeling_univa_qwen2vl import (
    UnivaQwen2VLForConditionalGeneration,
)
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from diffusers import SD3Transformer2DModel, FluxTransformer2DModel
import torch

model_type = 'flux'
qwenvl_base = "Qwen2-VL-7B-Instruct"
origin_qwenvl_ckpt_path = f"/mnt/data/checkpoints/Qwen/{qwenvl_base}"
if model_type == 'sd3':
    origin_sd3_ckpt_path = "/mnt/data/checkpoints/stabilityai/stable-diffusion-3.5-large"
    save_sd3_ckpt_path = f"/mnt/data/checkpoints/UniVA/UniVA-{qwenvl_base}-stable-diffusion-3.5-large-fp32"
elif model_type == 'flux':
    origin_flux_ckpt_path = "/mnt/data/checkpoints/black-forest-labs/FLUX.1-dev"
    save_flux_ckpt_path = f"/mnt/data/checkpoints/UniVA/UniVA-{qwenvl_base}-FLUX.1-dev-fp32"

processor = AutoProcessor.from_pretrained(origin_qwenvl_ckpt_path)

config = UnivaQwen2VLConfig.from_pretrained(origin_qwenvl_ckpt_path)

if model_type == 'sd3':
    config.denoise_tower = UnivaDenoiseTowerConfig(
        denoiser_type="sd3",
        denoise_projector_type="mlp2x_gelu",
        input_hidden_size=config.hidden_size,
        output_hidden_size=4096,
        denoiser_config=f"{origin_sd3_ckpt_path}/transformer/config.json",
    )
elif model_type == 'flux':
    config.denoise_tower = UnivaDenoiseTowerConfig(
        denoiser_type="flux",
        denoise_projector_type="mlp2x_gelu",
        input_hidden_size=config.hidden_size,
        output_hidden_size=4096,
        denoiser_config=f"{origin_flux_ckpt_path}/transformer/config.json",
    )
print(config)


#######################################################################################
model = UnivaQwen2VLForConditionalGeneration._from_config(
    config, 
    torch_dtype=torch.float32,
    )
print(model.dtype, model.device)

qwenvl = Qwen2VLForConditionalGeneration.from_pretrained(
    origin_qwenvl_ckpt_path,
    torch_dtype=torch.float32,
)
print(qwenvl.dtype, qwenvl.device)
missing_key, unexpected_key = model.load_state_dict(qwenvl.state_dict(), strict=False)
assert all(['denoise_tower' in miss_key for miss_key in missing_key])
assert len(unexpected_key) == 0
#######################################################################################





#######################################################################################
if model_type == 'sd3':
    sd3 = SD3Transformer2DModel.from_pretrained(
        f"{origin_sd3_ckpt_path}",
        subfolder="transformer",
        torch_dtype=torch.float32,
    )
    print(sd3.dtype, sd3.device)
    model.denoise_tower.denoiser = sd3
    model.save_pretrained(save_sd3_ckpt_path)
elif model_type == 'flux':
    flux = FluxTransformer2DModel.from_pretrained(
        f"{origin_flux_ckpt_path}",
        subfolder="transformer",
        torch_dtype=torch.float32,
    )
    print(flux.dtype, flux.device)
    model.denoise_tower.denoiser = flux
    model.save_pretrained(save_flux_ckpt_path)
#######################################################################################

if model_type == 'sd3':
    processor.save_pretrained(save_sd3_ckpt_path)
elif model_type == 'flux':
    processor.save_pretrained(save_flux_ckpt_path)