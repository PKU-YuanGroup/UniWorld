from univa.models.configuration_univa_denoise_tower import UnivaDenoiseTowerConfig
from transformers.modeling_utils import PreTrainedModel

from typing import Any, Dict, Optional, Tuple, Union
import torch
from torch import nn
import numpy as np
from diffusers import FluxTransformer2DModel, SD3Transformer2DModel
from diffusers.utils import is_torch_version
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from torch.nn.utils.rnn import pad_sequence


class UnivaDenoiseTower(PreTrainedModel):
    config_class = UnivaDenoiseTowerConfig
    base_model_prefix = "model"

    def __init__(self, config: UnivaDenoiseTowerConfig):
        super().__init__(config)
        self.config = config
        if config.denoiser_type == "flux":
            self.denoiser = FluxTransformer2DModel.from_config(config.denoiser_config)
        elif config.denoiser_type == "sd3":
            self.denoiser = SD3Transformer2DModel.from_config(config.denoiser_config)
        else:
            raise ValueError(f"Unknown denoiser type: {config.denoiser_type}")

        self._init_denoise_projector()
        self._init_vae_projector()
        self._init_siglip_projector()

    def _init_denoise_projector(self):
        """Initialize the denoise_projector for multi model input."""
        if self.config.denoise_projector_type == "mlp2x_gelu":
            self.denoise_projector = nn.Sequential(
                nn.Linear(
                    self.config.input_hidden_size,
                    self.config.output_hidden_size * 3,
                ),
                nn.SiLU(),
                nn.Linear(
                    self.config.output_hidden_size * 3, self.config.output_hidden_size
                ),
            )
        else:
            raise ValueError(
                f"Unknown denoise_projector_type: {self.config.denoise_projector_type}"
            )

    def _init_vae_projector(self):
        """Initialize the denoise_projector for multi model input."""
        if self.config.vae_projector_type == "mlp2x_gelu":
            self.vae_projector = nn.Sequential(
                nn.Linear(
                    self.config.vae_input_hidden_size,
                    # 2 * self.config.output_hidden_size,
                    3072,  # HARDCODE, x_embedder from flux
                ),
                nn.SiLU(),
                nn.Linear(
                    # 2 * self.config.output_hidden_size, 
                    3072,  # HARDCODE, x_embedder from flux
                    self.config.output_hidden_size
                ),
            )
        # elif self.config.vae_projector_type == "linear":
        # self.vae_projector = nn.Sequential(
        #     nn.Linear(
        #         self.config.vae_input_hidden_size,
        #         self.config.output_hidden_size,
        #     ),
        # )
        else:
            raise ValueError(
                f"Unknown vae_projector_type: {self.config.vae_projector_type}"
            )

    def _init_siglip_projector(self):
        """Initialize the denoise_projector for multi model input."""
        self.siglip_projector = nn.Sequential(
            nn.Linear(
                1152,  # HARDCODE, out from siglip
                4096 * 3,  # HARDCODE
            ),
            nn.SiLU(),
            nn.Linear(
                4096 * 3,  # HARDCODE
                4096,  # HARDCODE, context_embedder from flux
            ),
        )

    def _init_convnext_projector(self):
        """Initialize the denoise_projector for multi model input."""
        self.convnext_projector = nn.Sequential(
            nn.Linear(
                1152,  # HARDCODE, out from convnext
                4096 * 3,  # HARDCODE
            ),
            nn.SiLU(),
            nn.Linear(
                4096 * 3,  # HARDCODE
                4096,  # HARDCODE, context_embedder from flux
            ),
        )
        
    @staticmethod
    def _insert_image_feats(
            encoder_h, img_feats, img_pos, 
            output_hidden_size, vae_projector
            ):
        """
        encoder_h: Tensor[B, L, D]
        img_feats: list of B lists: 第 i 个元素是一个 list，长度 = len(img_pos[i])，
                其内第 k 项是一个 Tensor[Nik, D]
        img_pos:   list of B lists: 第 i 个元素是个位置列表 [p_i0, p_i1, ...]
                len(img_pos[i]) == len(img_feats[i])
        returns:   Tensor[B, L + Nmax, D]，在各自位置插入完后，按最长插入数 pad 右侧
        """
        B, L, D = encoder_h.shape
        device = encoder_h.device

        # —— 1. 每个样本先把多组 feats concat 成一条“插入流”，同时 expand positions
        flat_feats = []
        flat_pos   = []
        for feats_list, pos_list in zip(img_feats, img_pos):
            assert len(feats_list) == len(pos_list)
            # feats_list = [Tensor[N0,D], Tensor[N1,D], ...]
            # pos_list   = [p0,      p1,       ...]
            # concat 所有要插入的 tokens
            if len(feats_list) == 0:
                # 没有插入
                concat_f = torch.empty(0, output_hidden_size, device=device)
                pos_expanded = torch.empty(0, dtype=torch.long, device=device)
            else:
                concat_f = torch.cat(feats_list, dim=0)    # [Ni_total, D]
                concat_f = vae_projector(concat_f)
                # 对应位置也 expand 成同样长度
                # eg. feats_list[0].shape[0] 个 p0， feats_list[1].shape[0] 个 p1，…
                # ATTENTION p-1
                pos_expanded = torch.cat([
                    torch.full((f.shape[0],), p-1, dtype=torch.long, device=device)
                    for f, p in zip(feats_list, pos_list)
                ], dim=0)                                   # [Ni_total]
            flat_feats.append(concat_f)
            flat_pos.append(pos_expanded)

        # —— 2. pad 到同一个长度 Nmax
        padded_feats = pad_sequence(flat_feats, batch_first=True)    # [B, Nmax, D]
        pos_pad = pad_sequence(flat_pos, batch_first=True, padding_value=L)

        # —— 3. 准备所有 token 的“排序键”（sort‐key）
        # 原 token j 的 key = 2*j
        orig_key = (torch.arange(L, device=device) * 2).unsqueeze(0).expand(B, -1)       # [B, L]
        # 插入 token 的 key = 2*pos + 1
        ins_key  = pos_pad * 2 + 1                                                      # [B, Nmax]

        # —— 4. 拼接、一次性排序 + gather
        all_keys   = torch.cat([orig_key,    ins_key],    dim=1)                        # [B, L+Nmax]
        all_feats  = torch.cat([encoder_h, padded_feats], dim=1)                        # [B, L+Nmax, D]
        sort_idx   = all_keys.argsort(dim=1)                                            # [B, L+Nmax]
        new_seq    = all_feats.gather(1, sort_idx.unsqueeze(-1).expand(-1, -1, D))      # [B, L+Nmax, D]

        return new_seq

    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        pooled_projections: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        # if encoder_hidden_states is not None:
        #     encoder_hidden_states = self.denoise_projector(encoder_hidden_states)
        if self.config.denoiser_type == "flux":
            prefix_prompt_embeds = kwargs.pop("prefix_prompt_embeds", None)
            
            if encoder_hidden_states is not None:
                if prefix_prompt_embeds is not None:
                    encoder_hidden_states = torch.concat(
                        [encoder_hidden_states, prefix_prompt_embeds], dim=1
                    )
            else:
                assert prefix_prompt_embeds is not None
                encoder_hidden_states = prefix_prompt_embeds
            txt_ids = torch.zeros(encoder_hidden_states.shape[1], 3).to(
                hidden_states.device, dtype=hidden_states.dtype
            )

            joint_attention_kwargs = kwargs.pop('joint_attention_kwargs', None)
            # if joint_attention_kwargs is not None and 'attention_mask' in joint_attention_kwargs:
            #     attention_mask = joint_attention_kwargs['attention_mask']
            # else:
            #     attention_mask = torch.full(
            #         (hidden_states.shape[0], 1, hidden_states.shape[1]), 
            #         True, dtype=torch.bool, device=hidden_states.device
            #         )
                
            enc_attention_mask = kwargs.pop('enc_attention_mask', None)
            # if enc_attention_mask is None:
            #     enc_attention_mask = torch.full(
            #         (encoder_hidden_states.shape[0], 1, encoder_hidden_states.shape[1]), 
            #         True, dtype=torch.bool, device=encoder_hidden_states.device
            #         )
            # else:
            #     enc_attention_mask = enc_attention_mask.unsqueeze(1)
                    
            # attention_mask = torch.concat([enc_attention_mask, attention_mask], dim=-1)
            # attention_mask = attention_mask.unsqueeze(1)

            # joint_attention_kwargs['attention_mask'] = attention_mask
            # kwargs['joint_attention_kwargs'] = joint_attention_kwargs

            # print(f'hidden_states.shape, {hidden_states.shape}, encoder_hidden_states.shape, {encoder_hidden_states.shape}')
            # return self.fixed_flux_forward(
            return self.denoiser(
                hidden_states=hidden_states,
                timestep=timestep, # Note: timestep is in [0, 1]. It has been scaled by 1000 in the training script.
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                txt_ids=txt_ids,
                **kwargs,
            )[0]

        elif self.config.denoiser_type == "sd3":
            prefix_prompt_embeds = kwargs.pop("prefix_prompt_embeds", None)
            if prefix_prompt_embeds is not None:
                encoder_hidden_states = torch.concat(
                    [prefix_prompt_embeds, encoder_hidden_states], dim=1
                )

            return self.denoiser(
                hidden_states=hidden_states,
                timestep=timestep,
                encoder_hidden_states=encoder_hidden_states,
                pooled_projections=pooled_projections,
                **kwargs,
            )[0]



    def fixed_flux_forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor = None,
        pooled_projections: torch.Tensor = None,
        timestep: torch.LongTensor = None,
        img_ids: torch.Tensor = None,
        txt_ids: torch.Tensor = None,
        guidance: torch.Tensor = None,
        joint_attention_kwargs: Optional[Dict[str, Any]] = None,
        controlnet_block_samples=None,
        controlnet_single_block_samples=None,
        return_dict: bool = True,
        controlnet_blocks_repeat: bool = False,
    ) -> Union[torch.FloatTensor, Transformer2DModelOutput]:
        """
        The [`FluxTransformer2DModel`] forward method.

        Args:
            hidden_states (`torch.FloatTensor` of shape `(batch size, channel, height, width)`):
                Input `hidden_states`.
            encoder_hidden_states (`torch.FloatTensor` of shape `(batch size, sequence_len, embed_dims)`):
                Conditional embeddings (embeddings computed from the input conditions such as prompts) to use.
            pooled_projections (`torch.FloatTensor` of shape `(batch_size, projection_dim)`): Embeddings projected
                from the embeddings of input conditions.
            timestep ( `torch.LongTensor`):
                Used to indicate denoising step.
            block_controlnet_hidden_states: (`list` of `torch.Tensor`):
                A list of tensors that if specified are added to the residuals of transformer blocks.
            joint_attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~models.transformer_2d.Transformer2DModelOutput`] instead of a plain
                tuple.

        Returns:
            If `return_dict` is True, an [`~models.transformer_2d.Transformer2DModelOutput`] is returned, otherwise a
            `tuple` where the first element is the sample tensor.
        """

        hidden_states = self.denoiser.x_embedder(hidden_states)

        timestep = timestep.to(hidden_states.dtype) * 1000
        if guidance is not None:
            guidance = guidance.to(hidden_states.dtype) * 1000
        else:
            guidance = None

        temb = (
            self.denoiser.time_text_embed(timestep, pooled_projections)
            if guidance is None
            else self.denoiser.time_text_embed(timestep, guidance, pooled_projections)
        )
        encoder_hidden_states = self.denoiser.context_embedder(encoder_hidden_states)

        if txt_ids.ndim == 3:
            txt_ids = txt_ids[0]
        if img_ids.ndim == 3:
            img_ids = img_ids[0]

        ids = torch.cat((txt_ids, img_ids), dim=0)
        image_rotary_emb = self.denoiser.pos_embed(ids)
        if joint_attention_kwargs is not None and "ip_adapter_image_embeds" in joint_attention_kwargs:
            ip_adapter_image_embeds = joint_attention_kwargs.pop("ip_adapter_image_embeds")
            ip_hidden_states = self.denoiser.encoder_hid_proj(ip_adapter_image_embeds)
            joint_attention_kwargs.update({"ip_hidden_states": ip_hidden_states})

        for index_block, block in enumerate(self.denoiser.transformer_blocks):
            if torch.is_grad_enabled() and self.denoiser.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                encoder_hidden_states, hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    encoder_hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs,  # add this line
                    **ckpt_kwargs,
                )

            else:
                encoder_hidden_states, hidden_states = block(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_block_samples is not None:
                interval_control = len(self.denoiser.transformer_blocks) / len(controlnet_block_samples)
                interval_control = int(np.ceil(interval_control))
                # For Xlabs ControlNet.
                if controlnet_blocks_repeat:
                    hidden_states = (
                        hidden_states + controlnet_block_samples[index_block % len(controlnet_block_samples)]
                    )
                else:
                    hidden_states = hidden_states + controlnet_block_samples[index_block // interval_control]
        hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)

        for index_block, block in enumerate(self.denoiser.single_transformer_blocks):
            if torch.is_grad_enabled() and self.denoiser.gradient_checkpointing:

                def create_custom_forward(module, return_dict=None):
                    def custom_forward(*inputs):
                        if return_dict is not None:
                            return module(*inputs, return_dict=return_dict)
                        else:
                            return module(*inputs)

                    return custom_forward

                ckpt_kwargs: Dict[str, Any] = {"use_reentrant": False} if is_torch_version(">=", "1.11.0") else {}
                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(block),
                    hidden_states,
                    temb,
                    image_rotary_emb,
                    joint_attention_kwargs, 
                    **ckpt_kwargs,
                )

            else:
                hidden_states = block(
                    hidden_states=hidden_states,
                    temb=temb,
                    image_rotary_emb=image_rotary_emb,
                    joint_attention_kwargs=joint_attention_kwargs,
                )

            # controlnet residual
            if controlnet_single_block_samples is not None:
                interval_control = len(self.denoiser.single_transformer_blocks) / len(controlnet_single_block_samples)
                interval_control = int(np.ceil(interval_control))
                hidden_states[:, encoder_hidden_states.shape[1] :, ...] = (
                    hidden_states[:, encoder_hidden_states.shape[1] :, ...]
                    + controlnet_single_block_samples[index_block // interval_control]
                )

        hidden_states = hidden_states[:, encoder_hidden_states.shape[1] :, ...]

        hidden_states = self.denoiser.norm_out(hidden_states, temb)
        output = self.denoiser.proj_out(hidden_states)


        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)


