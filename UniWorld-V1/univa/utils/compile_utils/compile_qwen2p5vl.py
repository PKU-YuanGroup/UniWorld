from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl
import torch

class CompiledQwen2_5_VLVisionBlock(modeling_qwen2_5_vl.Qwen2_5_VLVisionBlock):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.compile
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
class CompiledQwen2_5_VLPatchMerger(modeling_qwen2_5_vl.Qwen2_5_VLPatchMerger):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.compile
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
class CompiledQwen2_5_VLDecoderLayer(modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.compile
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
class CompiledQwen2_5_VLRotaryEmbedding(modeling_qwen2_5_vl.Qwen2_5_VLRotaryEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.compile
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
class CompiledQwen2_5_VisionRotaryEmbedding(modeling_qwen2_5_vl.Qwen2_5_VisionRotaryEmbedding):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.compile
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
class CompiledQwen2_5_VisionPatchEmbed(modeling_qwen2_5_vl.Qwen2_5_VisionPatchEmbed):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @torch.compile
    def forward(self, *args, **kwargs):
        return super().forward(*args, **kwargs)
    
modeling_qwen2_5_vl.Qwen2_5_VLVisionBlock = CompiledQwen2_5_VLVisionBlock
modeling_qwen2_5_vl.Qwen2_5_VLPatchMerger = CompiledQwen2_5_VLPatchMerger
modeling_qwen2_5_vl.Qwen2_5_VLDecoderLayer = CompiledQwen2_5_VLDecoderLayer
modeling_qwen2_5_vl.Qwen2_5_VLRotaryEmbedding = CompiledQwen2_5_VLRotaryEmbedding
modeling_qwen2_5_vl.Qwen2_5_VisionRotaryEmbedding = CompiledQwen2_5_VisionRotaryEmbedding
modeling_qwen2_5_vl.Qwen2_5_VisionPatchEmbed = CompiledQwen2_5_VisionPatchEmbed