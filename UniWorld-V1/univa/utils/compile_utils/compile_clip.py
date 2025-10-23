from transformers.models.clip import modeling_clip
import torch

class CompiledCLIPTextModel(modeling_clip.CLIPTextModel):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.compile
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
modeling_clip.CLIPTextModel = CompiledCLIPTextModel