from diffusers.models.autoencoders import autoencoder_kl, vae
import torch

class CompiledAutoencoderKL(autoencoder_kl.AutoencoderKL):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.compile
    def encode(self, *args, **kwargs):
        return super().encode(*args, **kwargs)
    
autoencoder_kl.AutoencoderKL = CompiledAutoencoderKL


class CompiledEncoder(vae.Encoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.compile
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
vae.Encoder = CompiledEncoder