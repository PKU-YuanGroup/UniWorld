# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.nn as nn
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from einops import rearrange

from torch.distributed.tensor import Shard, Replicate
from ultrai2v.utils.utils import is_npu_available

from ultrai2v.distributed.redistribution import Redistribution


from .attention import flash_attention, attention


T5_CONTEXT_TOKEN_NUMBER = 512

def sinusoidal_embedding_1d(dim, position):
    # preprocess
    assert dim % 2 == 0
    half = dim // 2
    position = position.type(torch.float64)

    # calculation
    sinusoid = torch.outer(
        position, torch.pow(10000, -torch.arange(half).to(position).div(half))
    )
    x = torch.cat([torch.cos(sinusoid), torch.sin(sinusoid)], dim=1)
    return x


@torch.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000):
    assert dim % 2 == 0
    freqs = torch.outer(
        torch.arange(max_seq_len),
        1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim)),
    )
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs


@torch.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    if is_npu_available(): freqs = freqs.to(torch.complex64)
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(
            x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2)
        )
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).float()


class WanRMSNorm(nn.Module):

    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return self._norm(x.float()).type_as(x) * self.weight

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)


class WanLayerNorm(nn.LayerNorm):

    def __init__(self, dim, eps=1e-6, elementwise_affine=False):
        super().__init__(dim, elementwise_affine=elementwise_affine, eps=eps)

    def forward(self, x):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
        """
        return super().forward(x.float()).type_as(x)


class WanSelfAttention(nn.Module):

    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

        # cp dummy layers
        self.cp_self_attn_before_attention_layer = nn.Identity()
        self.cp_self_attn_after_attention_layer = nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        
        # maybe we need cp
        q = self.cp_self_attn_before_attention_layer(q)
        k = self.cp_self_attn_before_attention_layer(k)
        v = self.cp_self_attn_before_attention_layer(v)

        attention_func = flash_attention if not is_npu_available() else attention

        x = attention_func(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
        )

        # maybe we need cp
        x = self.cp_self_attn_after_attention_layer(x)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cp_cross_attn_before_attention_layer = nn.Identity()
        self.cp_cross_attn_after_attention_layer = nn.Identity()

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # maybe we need cp
        q = self.cp_cross_attn_before_attention_layer(q)
        k = self.cp_cross_attn_before_attention_layer(k)
        v = self.cp_cross_attn_before_attention_layer(v)

        # compute attention
        attention_func = flash_attention if not is_npu_available() else attention
        x = attention_func(q, k, v, k_lens=context_lens)

        # maybe we need cp
        x = self.cp_cross_attn_after_attention_layer(x)
        
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x



class WanAttentionBlock(nn.Module):

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = WanSelfAttention(dim, num_heads, window_size, qk_norm, eps)
        self.norm3 = (
            WanLayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )
        self.cross_attn = WanT2VCrossAttention(dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(
            nn.Linear(dim, ffn_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(ffn_dim, dim),
        )

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            e(Tensor): Shape [B, 6, C]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        e = (self.modulation + e).chunk(6, dim=1)
        # self-attention
        y = self.self_attn(
            self.norm1(x) * (1 + e[1]) + e[0], seq_lens, grid_sizes, freqs
        )
        x = x + y * e[2]
        # cross-attention & ffn function
        x = x + self.cross_attn(self.norm3(x), context, context_lens)
        y = self.ffn(self.norm2(x) * (1 + e[4]) + e[3])
        x = x + y * e[5]
        return x


class Head(nn.Module):

    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, C]
        """
        e = (self.modulation + e.unsqueeze(1)).chunk(2, dim=1)
        x = self.head(self.norm(x) * (1 + e[1]) + e[0])
        return x



class WanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video.
    """

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
        model_type="t2v",
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
        **kwargs,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video) or 'flf2v' (first-last-frame-to-video) or 'vace'
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in ["t2v", "i2v"]
        self.model_type = model_type

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        self.gradient_checkpointing = False

        # embeddings
        self.patch_embedding = nn.Conv3d(
            in_dim, dim, kernel_size=patch_size, stride=patch_size
        )
        self.text_embedding = nn.Sequential(
            nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim)
        )

        self.time_embedding = nn.Sequential(
            nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim)
        )
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))

        # blocks
        self.blocks = nn.ModuleList(
            [
                WanAttentionBlock(
                    dim,
                    ffn_dim,
                    num_heads,
                    window_size,
                    qk_norm,
                    cross_attn_norm,
                    eps,
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = Head(dim, out_dim, patch_size, eps)

        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        self.rope_d = dim // num_heads
        self.freqs = None

        # cp dummy layers
        self.cp_input_layer = nn.Identity()
        self.cp_output_layer = nn.Identity()

        # initialize weights
        self.init_weights()

    def set_gradient_checkpointing(self, enabled = False):
        self.gradient_checkpointing = enabled 

    def reset_parameters(self):
        print(f"{__class__.__name__} reset parameters!")
        self.init_weights()

    def forward(
        self,
        x, # [B C T H W]
        t, # [B]
        context, # [B N C]
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
        context = self.text_embedding(context)
        # arguments
        args = [x, e0, seq_lens, grid_size_for_rope, self.freqs, context, context_lens]

        for block in self.blocks:
            if self.gradient_checkpointing and self.training:
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

    def patchify(self, embs):
        # get f, h, w from b c f h w
        grid_sizes = embs.shape[2:]

        # b c f h w  -> b (f h w) c
        patch_out = rearrange(embs, "b c f h w -> b (f h w) c")

        return patch_out, grid_sizes

    def unpatchify(self, embs, frames, height, width):
        # b (f h w) (x y z c) -> b c (f x) (h y) (w z)
        patch_out = rearrange(
            embs,
            "b (f h w) (x y z c) -> b c (f x) (h y) (w z)",
            f=frames,
            h=height,
            w=width,
            x=self.patch_size[0],
            y=self.patch_size[1],
            z=self.patch_size[2],
        )
        return patch_out


    # def init_weights(self):
    #     r"""
    #     Initialize model parameters using Xavier initialization.
    #     """

    #     # basic init
    #     for m in self.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.xavier_uniform_(m.weight)
    #             if m.bias is not None:
    #                 nn.init.zeros_(m.bias)

    #     # init embeddings
    #     nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
    #     for m in self.text_embedding.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, std=0.02)
    #     for m in self.time_embedding.modules():
    #         if isinstance(m, nn.Linear):
    #             nn.init.normal_(m.weight, std=0.02)

    #     # init output layer
    #     nn.init.zeros_(self.head.head.weight)

    def init_weights(self):
        for n, m in self.named_modules():
            if n == "":
                continue
            if hasattr(m, "reset_parameters"):
                # print(f"{n} -> reset_parameters")
                m.reset_parameters()


models = {
    "wan_t2v": WanModel
}

models_main_block = {
    "wan_t2v": WanAttentionBlock
}

models_blocks_to_float = {
    "wan_t2v": [WanLayerNorm, WanRMSNorm]
}

models_blocks_to_output_float = {
    "wan_t2v": None
}

cp_plans = {
    "wan_t2v": {
        WanModel:{
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
    model = WanModel().to(device=device, dtype=dtype)
    model.set_gradient_checkpointing(True)
    x = torch.randn(2, 16, 21, 60, 104, device=device, dtype=dtype)
    t = torch.randint(0, 1000, (2,), device=device)
    context = torch.randn(2, 512, 4096, device=device, dtype=dtype)
    with torch.autocast("cuda", dtype=dtype):
        y = model(x, t, context)
    print(y.shape)