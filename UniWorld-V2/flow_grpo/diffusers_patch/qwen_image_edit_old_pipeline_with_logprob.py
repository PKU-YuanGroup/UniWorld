from typing import Any, Dict, List, Optional, Union
import torch
import numpy as np
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit import (
    retrieve_timesteps,
)
from diffusers.pipelines.qwenimage.pipeline_qwenimage_edit_plus import (
    calculate_shift,
    calculate_dimensions,
)

from diffusers.image_processor import PipelineImageInput
from flow_grpo.diffusers_patch.solver import run_sampling


CONDITION_IMAGE_SIZE = 384 * 384
VAE_IMAGE_SIZE = 1024 * 1024

def _get_qwen_prompt_embeds(
    self,
    prompt: Union[str, List[str]] = None,
    image: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    max_seq_len: int = 1024,
):
    device = device or self._execution_device
    dtype = dtype or self.text_encoder.dtype
    prompt = [prompt] if isinstance(prompt, str) else prompt
    template = self.prompt_template_encode
    drop_idx = self.prompt_template_encode_start_idx
    txt = [template.format(e) for e in prompt]

    model_inputs = self.processor(
        text=txt,
        images=image,
        padding=True,
        return_tensors="pt",
    ).to(device)

    outputs = self.text_encoder(
        input_ids=model_inputs.input_ids,
        attention_mask=model_inputs.attention_mask,
        # pixel_values=model_inputs.pixel_values,
        # image_grid_thw=model_inputs.image_grid_thw,
        output_hidden_states=True,
    )

    hidden_states = outputs.hidden_states[-1]
    split_hidden_states = self._extract_masked_hidden(hidden_states, model_inputs.attention_mask)
    split_hidden_states = [e[drop_idx:] for e in split_hidden_states]
    attn_mask_list = [torch.ones(e.size(0), dtype=torch.long, device=e.device) for e in split_hidden_states]
    # max_seq_len = max([e.size(0) for e in split_hidden_states])
    prompt_embeds = torch.stack([
        torch.cat([
            u[:max_seq_len] if u.size(0) > max_seq_len else u,
            u.new_zeros(max(0, max_seq_len - u.size(0)), u.size(1))
        ])
        for u in split_hidden_states
    ])
    encoder_attention_mask = torch.stack([
        torch.cat([
            u[:max_seq_len] if u.size(0) > max_seq_len else u,
            u.new_zeros(max(0, max_seq_len - u.size(0)))
        ])
        for u in attn_mask_list
    ])
    prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
    return prompt_embeds, encoder_attention_mask

def encode_prompt(
    self,
    prompt: Union[str, List[str]],
    image: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None,
    num_images_per_prompt: int = 1,
    prompt_embeds: Optional[torch.Tensor] = None,
    prompt_embeds_mask: Optional[torch.Tensor] = None,
    max_sequence_length: int = 1024,
):
    device = device or self._execution_device
    prompt = [prompt] if isinstance(prompt, str) else prompt
    batch_size = len(prompt) if prompt_embeds is None else prompt_embeds.shape[0]
    if prompt_embeds is None:
        prompt_embeds, prompt_embeds_mask = _get_qwen_prompt_embeds(self, prompt, image, device, max_seq_len=max_sequence_length)
    _, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
    prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
    prompt_embeds_mask = prompt_embeds_mask.repeat(1, num_images_per_prompt, 1)
    prompt_embeds_mask = prompt_embeds_mask.view(batch_size * num_images_per_prompt, seq_len)
    return prompt_embeds, prompt_embeds_mask

@torch.no_grad()
def pipeline_with_logprob(
    self,
    image: Optional[PipelineImageInput] = None,
    prompt: Union[str, List[str]] = None,
    negative_prompt: Union[str, List[str]] = None,
    height: Optional[int] = None,
    width: Optional[int] = None,
    num_inference_steps: int = 28,
    true_cfg_scale: float = 4.0,
    guidance_scale: Optional[float] = None,
    num_images_per_prompt: Optional[int] = 1,
    generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
    latents: Optional[torch.FloatTensor] = None,
    prompt_embeds: Optional[torch.FloatTensor] = None,
    prompt_embeds_mask: Optional[torch.Tensor] = None,
    negative_prompt_embeds: Optional[torch.FloatTensor] = None,
    negative_prompt_embeds_mask: Optional[torch.Tensor] = None,
    output_type: Optional[str] = "pil",
    joint_attention_kwargs: Optional[Dict[str, Any]] = None,
    callback_on_step_end_tensor_inputs: List[str] = ["latents"],
    max_sequence_length: int = 256,
    noise_level: float = 0.7,
    deterministic: bool = False,
    max_area: Optional[int] = None,
    solver: str = "flow",
):
    max_area = VAE_IMAGE_SIZE if max_area is None else max_area
    image_size = image[0].size if isinstance(image, list) else image.size
    calculated_width, calculated_height = calculate_dimensions(max_area, image_size[0] / image_size[1])
    height = height or calculated_height
    width = width or calculated_width

    multiple_of = self.vae_scale_factor * 2
    width = width // multiple_of * multiple_of
    height = height // multiple_of * multiple_of

    self.check_inputs(
        prompt,
        height,
        width,
        negative_prompt=negative_prompt,
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=negative_prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        negative_prompt_embeds_mask=negative_prompt_embeds_mask,
        callback_on_step_end_tensor_inputs=callback_on_step_end_tensor_inputs,
        max_sequence_length=max_sequence_length,
    )

    self._guidance_scale = guidance_scale
    self._joint_attention_kwargs = joint_attention_kwargs
    self._current_timestep = None
    self._interrupt = False

    # Define call parameters
    if prompt is not None and isinstance(prompt, str):
        batch_size = 1
    elif prompt is not None and isinstance(prompt, list):
        batch_size = len(prompt)
    else:
        batch_size = prompt_embeds.shape[0]

    device = self._execution_device

    if image is not None and not (isinstance(image, torch.Tensor) and image.size(1) == self.latent_channels):
            image = self.image_processor.resize(image, calculated_height, calculated_width)
            prompt_image = image
            image = self.image_processor.preprocess(image, calculated_height, calculated_width)
            image = image.unsqueeze(2)

    has_neg_prompt = negative_prompt is not None or (
        negative_prompt_embeds is not None and negative_prompt_embeds_mask is not None
    )

    do_true_cfg = true_cfg_scale > 1 and has_neg_prompt

    prompt_embeds, prompt_embeds_mask = encode_prompt(
		self,
        prompt=prompt,
        prompt_embeds=prompt_embeds,
        prompt_embeds_mask=prompt_embeds_mask,
        device=device,
        num_images_per_prompt=num_images_per_prompt,
        max_sequence_length=max_sequence_length,
    )
    if do_true_cfg:
        negative_prompt_embeds, negative_prompt_embeds_mask = encode_prompt(
			self,
            prompt=negative_prompt,
            prompt_embeds=negative_prompt_embeds,
            prompt_embeds_mask=negative_prompt_embeds_mask,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
    
    # Preprocess image
    num_channels_latents = self.transformer.config.in_channels // 4
    latents, image_latents = self.prepare_latents(
        image,
        batch_size * num_images_per_prompt,
        num_channels_latents,
        height,
        width,
        prompt_embeds.dtype,
        device,
        generator,
        latents,
    )
    img_shapes = [
            [
                (1, height // self.vae_scale_factor // 2, width // self.vae_scale_factor // 2),
                (1, calculated_height // self.vae_scale_factor // 2, calculated_width // self.vae_scale_factor // 2),
            ]
        ] * batch_size

    # Prepare timesteps
    sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
    if (
        hasattr(self.scheduler.config, "use_flow_sigmas")
        and self.scheduler.config.use_flow_sigmas
    ):
        sigmas = None
    image_seq_len = latents.shape[1]
    mu = calculate_shift(
        image_seq_len,
        self.scheduler.config.get("base_image_seq_len", 256),
        self.scheduler.config.get("max_image_seq_len", 4096),
        self.scheduler.config.get("base_shift", 0.5),
        self.scheduler.config.get("max_shift", 1.15),
    )
    timesteps, num_inference_steps = retrieve_timesteps(
        self.scheduler,
        num_inference_steps,
        device,
        sigmas=sigmas,
        mu=mu,
    )
    self._num_timesteps = len(timesteps)

    # handle guidance
    if self.transformer.config.guidance_embeds and guidance_scale is None:
        raise ValueError("guidance_scale is required for guidance-distilled model.")
    elif self.transformer.config.guidance_embeds:
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0])
    elif not self.transformer.config.guidance_embeds and guidance_scale is not None:
        guidance = None
    elif not self.transformer.config.guidance_embeds and guidance_scale is None:
        guidance = None
    
    self._attention_kwargs = {}

    sigmas = self.scheduler.sigmas.float()

    txt_seq_lens = [max_sequence_length] * batch_size
    negative_txt_seq_lens = [max_sequence_length] * batch_size

    def v_pred_fn(z, sigma):
        latent_model_input = z
        if image_latents is not None:
            latent_model_input = torch.cat([z, image_latents], dim=1)

        timesteps = torch.full(
            [latent_model_input.shape[0]], sigma, device=z.device, dtype=torch.float32
        )
        noise_pred = self.transformer(
            hidden_states=latent_model_input,
            timestep=timesteps,
            guidance=guidance,
            encoder_hidden_states_mask=prompt_embeds_mask,
            encoder_hidden_states=prompt_embeds,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            attention_kwargs=self.attention_kwargs,
            return_dict=False,
        )[0]
        noise_pred = noise_pred[:, : latents.size(1)]
        
        if do_true_cfg:
            neg_noise_pred = self.transformer(
                hidden_states=latent_model_input,
                timestep=timesteps,
                guidance=guidance,
                encoder_hidden_states_mask=negative_prompt_embeds_mask,
                encoder_hidden_states=negative_prompt_embeds,
                img_shapes=img_shapes,
                txt_seq_lens=negative_txt_seq_lens,
                attention_kwargs=self.attention_kwargs,
                return_dict=False,
            )[0]
            neg_noise_pred = neg_noise_pred[:, : latents.size(1)]

            comb_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
            cond_norm = torch.norm(noise_pred, dim=-1, keepdim=True)
            noise_norm = torch.norm(comb_pred, dim=-1, keepdim=True)
            noise_pred = comb_pred * (cond_norm / noise_norm)

        return noise_pred

    # 6. Prepare image embeddings
    all_latents = [latents]
    all_log_probs = []

    # 7. Denoising loop
    latents, all_latents, all_log_probs = run_sampling(
        v_pred_fn, latents, sigmas, solver, deterministic, noise_level
    )

    latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
    latents = latents.to(self.vae.dtype)
    latents_mean = (
        torch.tensor(self.vae.config.latents_mean)
        .view(1, self.vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
        latents.device, latents.dtype
    )
    latents = latents / latents_std + latents_mean
    image = self.vae.decode(latents, return_dict=False)[0][:, :, 0]
    image = self.image_processor.postprocess(image, output_type=output_type)

    # Offload all models
    self.maybe_free_model_hooks()

    return (
        image,
        all_latents,
        image_latents,
        img_shapes,
        txt_seq_lens,
        prompt_embeds,
        prompt_embeds_mask,
        all_log_probs,
    )
