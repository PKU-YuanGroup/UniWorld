import torch
from torchvision.transforms import Compose

from diffusers.utils.torch_utils import randn_tensor
from .t2v_pipeline import T2VInferencePipeline
from ultrai2v.utils.constant import NEGATIVE_PROMOPT
from ultrai2v.data.utils.transforms import CenterCropResizeVideo, ToTensorAfterResize, AENorm

class FlashI2VInferencePipeline(T2VInferencePipeline):

    def prepare_transform(self, height, width):
        return Compose(
            [
                CenterCropResizeVideo((height, width), interpolation_mode='bicubic', align_corners=False, antialias=True),
                ToTensorAfterResize(),
                AENorm()
            ]
        )

    def prepare_start_frame_latents(self, image, transform):
        image = [transform(i.unsqueeze(0)) for i in image]
        image = torch.cat(image) # B [1 C H W] -> B C H W
        image = image.unsqueeze(2) # B C H W -> B C 1 H W
        image = image.to(dtype=self.vae.dtype, device=self.vae.device)
        image_latents = self.vae.encode(image).to(torch.float32)
        return image_latents

    @torch.inference_mode()
    def __call__(
        self,
        prompt,
        conditional_image,
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
            self.predictor.model.in_dim,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        # Caused by latent shifting, initial precision of latents must be fp32
        latents = self.prepare_latents(shape, generator=generator, device=device, dtype=torch.float32)

        transform = self.prepare_transform(height, width)
        start_frame_latents = self.prepare_start_frame_latents(conditional_image, transform)

        model_kwargs = {
            "prompt_embeds": prompt_embeds,
            "negative_prompt_embeds": negative_prompt_embeds,
            "start_frame_latents": start_frame_latents,
            "fourier_features": None,
            "start_frame_latents_proj": None,
        }

        latents = self.scheduler.sample(model=self.predictor, latents=latents, **model_kwargs)

        latents = latents.to(self.vae.dtype)
        video = self.decode_latents(latents)
        return video

pipeline = {
    'flashi2v': FlashI2VInferencePipeline
}