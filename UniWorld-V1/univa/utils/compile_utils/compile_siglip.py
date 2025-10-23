from transformers.models.siglip import modeling_siglip
import torch

class CompiledSiglipVisionModel(modeling_siglip.SiglipVisionModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.compile
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
modeling_siglip.SiglipVisionModel = CompiledSiglipVisionModel