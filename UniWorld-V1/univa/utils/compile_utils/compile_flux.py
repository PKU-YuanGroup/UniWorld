from diffusers.models.transformers import transformer_flux
import torch

class CompiledFluxTransformerBlock(transformer_flux.FluxTransformerBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.compile
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
class CompiledFluxSingleTransformerBlock(transformer_flux.FluxSingleTransformerBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.compile
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
class CompiledFluxPosEmbed(transformer_flux.FluxPosEmbed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.compile
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
class CompiledCombinedTimestepTextProjEmbeddings(transformer_flux.CombinedTimestepTextProjEmbeddings):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.compile
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
class CompiledAdaLayerNormContinuous(transformer_flux.AdaLayerNormContinuous):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.compile
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
transformer_flux.FluxTransformerBlock = CompiledFluxTransformerBlock
transformer_flux.FluxSingleTransformerBlock = CompiledFluxSingleTransformerBlock
transformer_flux.FluxPosEmbed = CompiledFluxPosEmbed
transformer_flux.CombinedTimestepTextProjEmbeddings = CompiledCombinedTimestepTextProjEmbeddings
transformer_flux.AdaLayerNormContinuous = CompiledAdaLayerNormContinuous