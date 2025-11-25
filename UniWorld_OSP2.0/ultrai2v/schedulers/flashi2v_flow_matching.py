
import torch
from tqdm import tqdm

from .flow_matching import FlowMatchingScheduler

class FlashI2VFlowMatchingScheduler(FlowMatchingScheduler):

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

        assert model_kwargs.get("start_frame_latents", None) is not None, "When use flashi2v, start_frame_latents must be specified!"
        assert latents.dtype == torch.float32, "Caused by latent shifting, initial precision of latents must be fp32!"

        # for loop denoising to get clean latents
        with tqdm(total=num_inference_steps, desc="Sampling") as progress_bar:
            for i in range(num_inference_steps):
                latent_model_input_cond, latent_model_input_uncond = self.get_latent_model_input(latents, **model_kwargs)
                t = timesteps[i]
                timestep = t.expand(latent_model_input_cond.shape[0]).to(latents.device)

                # with torch.autocast("cuda", dtype=latents.dtype):
                # flashi2v model use autocast internally
                output = model(
                    latent_model_input_cond,
                    timestep,
                    model_kwargs.get("prompt_embeds"),
                    **model_kwargs
                )
                noise_pred = output.pop("model_output")
                model_kwargs.update(output)
    
                if self.do_classifier_free_guidance:
                    # with torch.autocast("cuda", dtype=latents.dtype):
                    # flashi2v model use autocast internally
                    output = model(
                        latent_model_input_uncond,
                        timestep,
                        model_kwargs.get('negative_prompt_embeds'),
                        **model_kwargs
                    )
                    noise_uncond = output.pop("model_output")
                    noise_pred = noise_uncond + self.guidance_scale * (noise_pred - noise_uncond)
                # compute the previous noisy sample x_t -> x_t-1
                latents = self._step(noise_pred, latents, sigmas[i + 1] - sigmas[i])
                progress_bar.update()
        
        return latents

    def interpolation(
        self,
        latents,
        prior_dist,
        sigmas,
        start_frame_latents
    ):
        latents_dtype = latents.dtype
        latents = latents.float()
        prior_dist = prior_dist.float()
        sigmas = sigmas.float()
        start_frame_latents = start_frame_latents.float()

        interpolated_latents = sigmas * prior_dist + (1.0 - sigmas) * latents - start_frame_latents

        interpolated_latents = interpolated_latents.to(latents_dtype)

        return interpolated_latents


    def q_sample(
        self, 
        latents,
        start_frame_latents,
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
        interpolated_latents = self.interpolation(latents, prior_dist, sigmas, start_frame_latents)

        return dict(x_t=interpolated_latents, prior_dist=prior_dist, sigmas=sigmas, timesteps=timesteps)


flow_scheduler = {
    "flashi2v_flow_matching": FlashI2VFlowMatchingScheduler,
}