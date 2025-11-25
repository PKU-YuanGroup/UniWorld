import math
import torch
import torch.nn as nn
from torch.distributed.tensor import Shard, Replicate
from diffusers.configuration_utils import register_to_config
from .want2v import (
    sinusoidal_embedding_1d,
    rope_params,
    WanModel,
    WanAttentionBlock,
    WanSelfAttention,
    WanT2VCrossAttention,
    WanLayerNorm,
    WanRMSNorm,
)

from ultrai2v.distributed.redistribution import Redistribution

class LearnableProj(nn.Module):

    def __init__(
        self,
        in_dim=16,
        dim=2048,
        patch_size=(1, 2, 2),
        out_dim=16,
        conv3x3x3_proj=False,
    ):
        
        super().__init__()
        
        self.in_dim = in_dim
        self.dim = dim
        self.patch_size = patch_size
        self.out_dim = out_dim        
        self.conv3x3x3_proj = conv3x3x3_proj

        proj_in_dim = proj_out_dim = self.in_dim
        print(f"Use {'3 x 3 x 3' if self.conv3x3x3_proj else '1 x 3 x 3'} conv as a learnable proj!")
        if self.conv3x3x3_proj:
            self.proj = nn.Sequential(
                nn.Conv3d(
                    proj_in_dim,
                    proj_out_dim * 4,
                    kernel_size=(3, 3, 3),
                    stride=(1, 1, 1),
                    padding=(1, 1, 1),
                ),
                nn.SiLU(),
                nn.Conv3d(
                    proj_out_dim * 4, 
                    proj_out_dim,
                    kernel_size=(3, 3, 3),
                    stride=(1, 1, 1),
                    padding=(1, 1, 1)
                )
            )
        else:
            self.proj = nn.Sequential(
                nn.Conv3d(
                    proj_in_dim,
                    proj_out_dim * 4,
                    kernel_size=(1, 3, 3),
                    stride=(1, 1, 1),
                    padding=(0, 1, 1)
                ),
                nn.SiLU(),
                nn.Conv3d(
                    proj_out_dim * 4, 
                    proj_out_dim,
                    kernel_size=(1, 3, 3),
                    stride=(1, 1, 1),
                    padding=(0, 1, 1)
                )
            )

    def init_weights(self):
        nn.init.xavier_uniform_(self.proj[0].weight.flatten(1))
        nn.init.zeros_(self.proj[0].bias)
        nn.init.zeros_(self.proj[-1].weight)
        nn.init.zeros_(self.proj[-1].bias)

    def reset_parameters(self):
        self.init_weights()

    def forward(self, x):
        return self.proj(x)


class EncoderAdapter(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, p_drop=0.1):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self._init_weights() # Call the initialization method

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
                nn.init.xavier_uniform_(m.weight, gain=1.0)    
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def reset_parameters(self):
        print(f"{__class__.__name__} reset parameters!")
        for n, m in self.named_modules():
            if n == "":
                continue
            if hasattr(m, "reset_parameters"):
                # print(f"{n} -> reset_parameters")
                m.reset_parameters()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x

    @property
    def dtype(self):
        """Required for compatibility with diffusers pipeline.to()"""
        return next(self.parameters()).dtype
    


class FlashI2VModel(WanModel):

    ignore_for_config = [
        "patch_size",
        "cross_attn_norm",
        "qk_norm",
        "text_dim",
        "window_size",
    ]

    @register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        eps=1e-6,
        conv3x3x3_proj=False,
        **kwargs,
    ):
        super().__init__(
            patch_size=patch_size,
            text_len=text_len,
            in_dim=in_dim,
            dim=dim,
            ffn_dim=ffn_dim,
            freq_dim=freq_dim,
            text_dim=text_dim,
            out_dim=out_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            window_size=window_size,
            qk_norm=qk_norm,
            cross_attn_norm=cross_attn_norm,
            eps=eps,
            **kwargs
        )

        self.fourier_embedding = nn.Sequential(
            nn.Conv3d(
                in_channels=self.in_dim,
                out_channels=self.dim,
                kernel_size=self.patch_size,
                stride=self.patch_size
            ),
            nn.Conv3d(
                in_channels=self.dim,
                out_channels=self.dim,
                kernel_size=(1, 1, 1),
                stride=(1, 1, 1)
            )
        )

        self.learnable_proj = LearnableProj(
            in_dim=self.in_dim,
            dim=self.dim,
            patch_size=self.patch_size,
            out_dim=self.out_dim,
            conv3x3x3_proj=conv3x3x3_proj,
        )

        self.flashi2v_init_weights()

    # we use t2v model as initial weights, so set ignore_predictor as True
    def flashi2v_init_weights(self):

        nn.init.xavier_uniform_(self.fourier_embedding[0].weight.flatten(1))
        nn.init.zeros_(self.fourier_embedding[0].bias)
        nn.init.zeros_(self.fourier_embedding[-1].weight)
        nn.init.zeros_(self.fourier_embedding[-1].bias)

        self.learnable_proj.init_weights()

    def reset_parameters(self):
        print(f"{__class__.__name__} reset parameters!")
        super().reset_parameters()
        self.flashi2v_init_weights()

    def forward(
        self,
        x, # [B C T H W]
        t, # [B]
        context, # [B N C]
        fourier_features, # [B C T H W]
        **kwargs,
    ):
        # params
        device = self.patch_embedding.weight.device

        # maybe we use meta device for init, so rope freqs should init before forward
        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        if self.freqs is None:
            self.freqs = torch.cat(
                [
                    rope_params(1024, self.rope_d - 4 * (self.rope_d // 6)),
                    rope_params(1024, 2 * (self.rope_d // 6)),
                    rope_params(1024, 2 * (self.rope_d // 6)),
                ],
                dim=1,
            ).to(device)

        # embeddings
        x = self.patch_embedding(x)
        fourier_features = self.fourier_embedding(fourier_features)
        x = x + fourier_features

        # time embeddings
        # if not is_npu_available():
        #     with torch.autocast("cuda", dtype=torch.float32):
        #         e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        #         e0 = self.time_projection(e).unflatten(1, (6, self.dim))
        #         assert e.dtype == torch.float32 and e0.dtype == torch.float32
        #     e0 = e0.to(x.dtype)
        # else:
        e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).float())
        e0 = self.time_projection(e).unflatten(1, (6, self.dim))

        x, grid_sizes = self.patchify(x)
        seq_lens = torch.tensor(math.prod(grid_sizes), dtype=torch.long, device=device).repeat(x.size(0))
        grid_size_for_rope = torch.tensor(grid_sizes, dtype=torch.long, device=device).unsqueeze(0).repeat(x.size(0), 1)
        
        # maybe we need cp
        x = self.cp_input_layer(x)
        context = self.cp_input_layer(context)

        # context
        context_lens = None

        # print(self.text_embedding, context.shape, '!!!')
        context = self.text_embedding(context)
        # arguments
        args = [x, e0, seq_lens, grid_size_for_rope, self.freqs, context, context_lens]

        for block in self.blocks:
            # if self.gradient_checkpointing and self.training:
            if self.gradient_checkpointing:
                x = torch.utils.checkpoint.checkpoint(block, *args, use_reentrant=False)
            else:
                x = block(*args)
            args[0] = x
        # head
        x = self.head(x, e)

        x = self.cp_output_layer(x)

        # unpatchify
        x = self.unpatchify(x, *grid_sizes)
        return x.float()

models = {
    "flashi2v": FlashI2VModel
}

models_main_block = {
    "flashi2v": WanAttentionBlock
}

models_blocks_to_float = {
    "flashi2v": [WanLayerNorm, WanRMSNorm]
}

models_blocks_to_output_float = {
    "flashi2v": [LearnableProj]
}

cp_plans = {
    "flashi2v": {
        FlashI2VModel:{
            "cp_input_layer": Redistribution(
                original_layouts=(Replicate(),),
                target_layouts=(Shard(1),), # split on sequence dim, (B, N, C) -> (B, N / cp_size, C)
            ),
            "cp_output_layer": Redistribution(
                original_layouts=(Shard(1),),
                target_layouts=(Replicate(),), # gather on sequence dim, (B, N / cp_size, C) -> (B, N, C)
            ),
        },
        WanSelfAttention: {
            "cp_self_attn_before_attention_layer": Redistribution(
                original_layouts=(Shard(1),), 
                target_layouts=(Shard(2),), # all to all, (B, N / cp_size, H, D) -> (B, N, H / cp_size, D)
            ),
            "cp_self_attn_after_attention_layer": Redistribution(
                original_layouts=(Shard(2),),
                target_layouts=(Shard(1),), # all to all, (B, N, H / cp_size, D) -> (B, N / cp_size, H, D)
            ),
        },
        WanT2VCrossAttention: {
            "cp_cross_attn_before_attention_layer": Redistribution(
                original_layouts=(Shard(1),), 
                target_layouts=(Shard(2),), # all to all, (B, N / cp_size, H, D) -> (B, N, H / cp_size, D)
            ),
            "cp_cross_attn_after_attention_layer": Redistribution(
                original_layouts=(Shard(2),),
                target_layouts=(Shard(1),), # all to all, (B, N, H / cp_size, D) -> (B, N / cp_size, H, D)
            ),
        }
    }
}

if __name__ == "__main__":
    device = "cuda:0"
    dtype = torch.bfloat16
    model = FlashI2VModel().to(device=device, dtype=dtype)
    model.set_gradient_checkpointing(True)
    x = torch.randn(2, 16, 21, 60, 104, device=device, dtype=dtype)
    x = torch.cat([x, x[:, :, 0:1].repeat(1, 1, x.shape[2], 1, 1)], dim=1)
    t = torch.randint(0, 1000, (2,), device=device)
    context = torch.randn(2, 512, 4096, device=device, dtype=dtype)
    with torch.autocast("cuda", dtype=dtype):
        y = model(x, t, context)
    print(y.shape)