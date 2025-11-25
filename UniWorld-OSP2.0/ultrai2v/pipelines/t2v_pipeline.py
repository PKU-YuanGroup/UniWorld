import torch
import ftfy
import html
import re

from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines import DiffusionPipeline
from ultrai2v.utils.constant import NEGATIVE_PROMOPT

class T2VInferencePipeline(DiffusionPipeline):

    # model_cpu_offload_seq = "text_encoder->predictor->vae"

    def __init__(
        self, 
        vae,
        tokenizer,
        text_encoder, 
        predictor, 
        scheduler
    ):
        super().__init__()
        self.register_modules(
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            predictor=predictor,
            scheduler=scheduler
        )
        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 8

    def prompt_preprocess(self, prompt):

        def basic_clean(text):
            text = ftfy.fix_text(text)
            text = html.unescape(html.unescape(text))
            return text.strip()

        def whitespace_clean(text):
            text = re.sub(r"\s+", " ", text)
            text = text.strip()

            return text

        return whitespace_clean(basic_clean(prompt))

    def _get_prompt_embeds(
        self,
        prompt=None,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        prompt = [self.prompt_preprocess(u) for u in prompt]
        batch_size = len(prompt)
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device))

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        prompt_embeds = prompt_embeds.view(batch_size, seq_len, -1)

        return prompt_embeds.to(dtype)


    def encode_texts(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_prompt_embeds(
                prompt=prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_prompt_embeds(
                prompt=negative_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def prepare_latents(self, shape, generator, device, dtype, latents=None):
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        if hasattr(self.scheduler, "init_noise_sigma"):
            # scale the initial noise by the standard deviation required by the scheduler
            latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents, value_range=(-1, 1), normalize=True, **kwargs):
        video = self.vae.decode(latents, **kwargs)  # [b, c, t, h, w]
        if normalize:
            low, high = value_range
            video.clamp_(min=low, max=high)
            video.sub_(low).div_(max(high - low, 1e-5))
        # [b, c, t, h, w] --> [b, t, h, w, c]
        video = video.mul(255).add_(0.5).clamp_(0, 255).permute(0, 2, 3, 4, 1).to("cpu", torch.uint8)
        return video

    @torch.inference_mode()
    def __call__(
        self,
        prompt,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        num_frames=49,
        height=480,
        width=832,
        seed=None,
        max_sequence_length=512,
        device="cuda:0",
    ):
        if negative_prompt is None or negative_prompt == "":
            negative_prompt = NEGATIVE_PROMOPT
        
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        do_classifier_free_guidance = self.scheduler.do_classifier_free_guidance
        prompt_embeds, negative_prompt_embeds = self.encode_texts(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
            dtype=self.text_encoder.dtype,
        )

        shape = (
            batch_size,
            self.predictor.in_dim,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        latents = self.prepare_latents(shape, generator=generator, device=device, dtype=prompt_embeds.dtype)

        model_kwargs = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
        }

        latents = self.scheduler.sample(model=self.predictor, latents=latents, **model_kwargs)

        latents = latents.to(self.vae.dtype)
        video = self.decode_latents(latents)
        return video

pipeline = {
    't2v': T2VInferencePipeline
}