
import logging

import math
import torch
from tqdm.auto import tqdm


class FlowMatchingScheduler:

    def __init__(
        self,
        num_inference_steps=None,
        guidance_scale=0.0,
        use_dynamic_shifting=False,
        use_logitnorm_time_sampling=False,
        shift=1.0,
        **kwargs
    ):
        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.do_classifier_free_guidance = guidance_scale > 1.0

        self.use_dynamic_shifting = use_dynamic_shifting
        self.shift = shift
        self.use_logitnorm_time_sampling = use_logitnorm_time_sampling

        if self.use_dynamic_shifting:
            logging.info(f'In {__class__.__name__}, use dynamic shifting.')
            self.base_image_seq = kwargs.pop("base_image_seq", 256)
            self.max_image_seq = kwargs.pop("max_image_seq", 4096)
            self.base_shift = kwargs.pop("base_shift", 0.5)
            self.max_shift = kwargs.pop("max_shift", 1.15)
            self.shift_k = (self.max_shift - self.base_shift) / (self.max_image_seq - self.base_image_seq)
            self.shift_b = self.base_shift - self.shift_k * self.base_image_seq

        if self.use_logitnorm_time_sampling:
            logging.info(f'In {__class__.__name__}, use logitnorm time sampling.')
            self.logit_mean = kwargs.pop("logit_mean", 0.0)
            self.logit_std = kwargs.pop("logit_std", 1.0)
        
        self._sigma_min = 0.0
        self._sigma_max = 1.0

    @property
    def sigma_min(self):
        return self._sigma_min

    @property
    def sigma_max(self):
        return self._sigma_max


    def interpolation(
        self,
        latents,
        prior_dist,
        sigmas,
    ):
        latents_dtype = latents.dtype
        latents = latents.float()
        prior_dist = prior_dist.float()
        sigmas = sigmas.float()

        interpolated_latents = sigmas * prior_dist + (1.0 - sigmas) * latents

        interpolated_latents = interpolated_latents.to(latents_dtype)

        return interpolated_latents

    def training_losses(
        self,
        model_output: torch.Tensor,
        latents: torch.Tensor,
        prior_dist: torch.Tensor,
        mask: torch.Tensor = None,
        sigmas: torch.Tensor = None,
        **kwargs
    ):
        if mask is not None:
            raise ValueError("mask is not supported currently.")

        target = prior_dist - latents
        loss = torch.nn.functional.mse_loss(model_output.float(), target.float(), reduction='mean')
        loss *= self._training_weight(sigmas)
        return [loss]

    def sample(
        self,
        model,
        latents,
        **model_kwargs,
    ):
        num_inference_steps = self.num_inference_steps
        sigmas = self._set_sigmas(training=False)
        print(f"sigmas: {sigmas}")
        timesteps = sigmas.clone() * 1000

        # for loop denoising to get clean latents
        with tqdm(total=num_inference_steps, desc="Sampling") as progress_bar:
            for i in range(num_inference_steps):
                latent_model_input_cond, latent_model_input_uncond = self.get_latent_model_input(latents, **model_kwargs)
                t = timesteps[i]
                timestep = t.expand(latent_model_input_cond.shape[0]).to(latents.device)

                with torch.autocast("cuda", dtype=latents.dtype):
                    noise_pred = model(
                        latent_model_input_cond,
                        timestep,
                        model_kwargs.get('prompt_embeds'),
                        **model_kwargs
                    )
    
                if self.do_classifier_free_guidance:
                    with torch.autocast("cuda", dtype=latents.dtype):
                        noise_uncond = model(
                            latent_model_input_uncond,
                            timestep,
                            model_kwargs.get('negative_prompt_embeds'),
                            **model_kwargs
                        )
                    noise_pred = noise_uncond + self.guidance_scale * (noise_pred - noise_uncond)
                # compute the previous noisy sample x_t -> x_t-1
                latents = self._step(noise_pred, latents, sigmas[i + 1] - sigmas[i])
                progress_bar.update()
        
        return latents

    def get_latent_model_input(self, latents, **kwargs):
        latent_model_input_cond = latent_model_input_uncond = latents
        return latent_model_input_cond, latent_model_input_uncond

    def q_sample(
        self, 
        latents,
        sigmas=None,
        prior_dist=None,
        **kwargs
    ):
        b, c, t, h, w = latents.shape
        if prior_dist is None:
            prior_dist = torch.randn_like(latents, dtype=latents.dtype, device=latents.device)
        image_seq_len = h * w // 4 if self.use_dynamic_shifting else None
        sigmas = self._set_sigmas(sigmas=sigmas, batch_size=b, image_seq_len=image_seq_len, device=latents.device)
        timesteps = sigmas.clone() * 1000
        while sigmas.ndim < latents.ndim:
            sigmas = sigmas.unsqueeze(-1)
        # print(f"rank = {torch.distributed.get_rank()}, sigmas = {sigmas.squeeze().cpu().numpy()}")
        interpolated_latents = self.interpolation(latents, prior_dist, sigmas)

        return dict(x_t=interpolated_latents, prior_dist=prior_dist, sigmas=sigmas, timesteps=timesteps)

    def _set_sigmas(self, sigmas=None, batch_size=None, image_seq_len=None, training=True, device=torch.device("cpu")):
        if training:
            if sigmas is None:
                if self.use_logitnorm_time_sampling:
                    sigmas = torch.normal(mean=self.logit_mean, std=self.logit_std, size=(batch_size,), device=device)
                    sigmas = torch.nn.functional.sigmoid(sigmas)
                else:
                    sigmas = torch.rand((batch_size,), device=device)
        else:
            sigmas = torch.linspace(self.sigma_max, self.sigma_min, self.num_inference_steps + 1, device=device)

        if self.use_dynamic_shifting:
            if image_seq_len is None:
                raise ValueError("you have to pass `image_seq_len` when `use_dynamic_shifting` is set to be `True`")
            shift = image_seq_len * self.shift_k + self.shift_b
            shift = math.exp(shift)
        else:
            shift = self.shift
        sigmas = shift * sigmas / (1 + (shift - 1) * sigmas)

        return sigmas


    def _step(self, model_output, latents, delta_t):
        latents_dtype = latents.dtype
        latents = latents.to(torch.float32)
        model_output = model_output.to(torch.float32)
        delta_t = delta_t.to(torch.float32)
        next_latents = latents + model_output * delta_t
        return next_latents.to(latents_dtype)

    def _training_weight(self, sigmas):
        return 1.0

flow_scheduler = {
    "flow_matching": FlowMatchingScheduler,
}