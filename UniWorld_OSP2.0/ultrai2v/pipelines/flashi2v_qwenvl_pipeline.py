import torch
from torchvision.transforms import Compose

from diffusers.utils.torch_utils import randn_tensor
from .t2v_pipeline import T2VInferencePipeline
from ultrai2v.utils.constant import NEGATIVE_PROMOPT
from ultrai2v.data.utils.transforms import CenterCropResizeVideo, ToTensorAfterResize, AENorm

from transformers import  AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
from PIL import Image
import os

class FlashI2VInferencePipeline(T2VInferencePipeline):

    def __init__(
        self, 
        vae,
        tokenizer,
        text_encoder, 
        vlm_encoder,
        vlm_encoder_adapter,
        predictor, 
        scheduler
    ):
        super().__init__(
            vae,
            tokenizer,
            text_encoder, 
            predictor, 
            scheduler
        )
        self.register_modules(
            vae=vae,
            tokenizer=tokenizer,
            text_encoder=text_encoder,
            vlm_encoder=vlm_encoder,
            vlm_encoder_adapter=vlm_encoder_adapter,
            predictor=predictor,
            scheduler=scheduler
        )
        self.vae_scale_factor_temporal = 4
        self.vae_scale_factor_spatial = 8
        self.processor = AutoProcessor.from_pretrained("/work/share/projects/gyy/zhubin_FlashI2V/checkpoints/Qwen2.5-VL-7B-Instruct")


    def prepare_transform(self, height, width):
        return Compose(
            [
                CenterCropResizeVideo((height, width), interpolation_mode='bicubic', align_corners=False, antialias=True),
                ToTensorAfterResize(),
                AENorm()
            ]
        )

    def prepare_start_frame_latents(self, image, transform):
        image = [transform(i.unsqueeze(0)) for i in image]
        image = torch.cat(image) # B [1 C H W] -> B C H W
        image = image.unsqueeze(2) # B C H W -> B C 1 H W
        image = image.to(dtype=self.vae.dtype, device=self.vae.device)
        image_latents = self.vae.encode(image).to(torch.float32)
        return image_latents

    def _get_vlm_embeds(
        self,
        prompt=None,
        img_path=None,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        assert type(prompt) is str 
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": img_path,
                    },
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        # Preparation for inference
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        # print(messages)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            # padding=True,
            padding="max_length", # 启用填充到最大长度
            max_length=1024,       # 指定最大长度为 1024
            # truncation=True,
            return_tensors="pt",
        )
        inputs = { key:value.to(device) for (key, value) in inputs.items()}
        embeds = self.vlm_encoder(**inputs, return_dict=False, use_cache=False,)[0]
        embeds_new = self.vlm_encoder_adapter(embeds)
        return embeds_new.to(dtype)
        # prompt = [self.prompt_preprocess(u) for u in prompt]
        # batch_size = len(prompt)
        # text_inputs = self.tokenizer(
        #     prompt,
        #     padding="max_length",
        #     max_length=max_sequence_length,
        #     truncation=True,
        #     add_special_tokens=True,
        #     return_attention_mask=True,
        #     return_tensors="pt",
        # )
        # text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask

        # prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device))

        # # duplicate text embeddings for each generation per prompt, using mps friendly method
        # _, seq_len, _ = prompt_embeds.shape
        # prompt_embeds = prompt_embeds.repeat(1, 1, 1)
        # prompt_embeds = prompt_embeds.view(batch_size, seq_len, -1)

        # return prompt_embeds.to(dtype)
    def resize_and_save_image(self, input_path: str,  new_width: int = 416, new_height: int = 240):
        """
        读取图像文件，将其调整为指定尺寸，并保存为新的图像文件。

        Args:
            input_path: 原始图像文件的路径。
            output_path: 保存新图像文件的路径。
            new_width: 调整后的目标宽度 (默认为 416)。
            new_height: 调整后的目标高度 (默认为 240)。
        """
        try:

            output_path = input_path.split('.')[0]+'_resize.jpg'
            # 1. 读取图像
            img = Image.open(input_path)
            print(f"成功读取图像: {input_path}")
            print(f"原始尺寸: 宽={img.width}, 高={img.height}")
            # 2. 调整图像大小 (Resize)
            # Image.LANCZOS 是高质量的重采样滤波器，通常用于缩小图像。
            resized_img = img.resize((new_width, new_height), Image.Resampling.LANCZOS)
            print(f"图像已调整大小为: 宽={new_width}, 高={new_height}")

            # 3. 保存新图像
            resized_img.save(output_path)
            return output_path

        except FileNotFoundError:
            print(f"错误：找不到文件 {input_path}")
        except Exception as e:
            print(f"处理图像时发生错误: {e}")

    def encode_text_with_image(
        self,
        prompt,
        img_path=None,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        assert img_path is not None
        if prompt_embeds is None:
            vlm_embeds = self._get_vlm_embeds(
                prompt=prompt,
                img_path=img_path,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_vlm_embeds = self._get_vlm_embeds(
                prompt=negative_prompt,
                img_path=img_path,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return vlm_embeds, negative_vlm_embeds


    @torch.inference_mode()
    def __call__(
        self,
        prompt,
        conditional_image,
        img_path=None,
        negative_prompt=None,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        num_frames=49,
        height=480,
        width=832,
        seed=None,
        max_sequence_length=512,
        device="cuda:0",
    ):

        if negative_prompt is None or negative_prompt == "":
            negative_prompt = NEGATIVE_PROMOPT
        
        # 2. Define call parameters
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        # 3. Encode input prompt
        do_classifier_free_guidance = self.scheduler.do_classifier_free_guidance
        
        if type(prompt) is list:
            assert len(prompt)==1
            prompt = prompt[0]
            img_path = img_path[0]
            resize_img_path = img_path.split('.')[0] + '_resize.jpg'
            if not os.path.exists(resize_img_path):
                resize_img_path = self.resize_and_save_image(img_path)
            img_path = resize_img_path
        # import ipdb; ipdb.set_trace()
        # vlm_embeddings
        vlm_embeds, negative_vlm_embeds = self.encode_text_with_image(
            prompt=prompt,
            img_path=img_path,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
            # dtype=self.text_encoder.dtype,
            dtype=self.vlm_encoder.dtype,
        )

        shape = (
            batch_size,
            self.predictor.model.in_dim,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        # Caused by latent shifting, initial precision of latents must be fp32
        latents = self.prepare_latents(shape, generator=generator, device=device, dtype=torch.float32)

        transform = self.prepare_transform(height, width)
        start_frame_latents = self.prepare_start_frame_latents(conditional_image, transform)

        model_kwargs = {
            # "prompt_embeds": prompt_embeds,
            # "negative_prompt_embeds": negative_prompt_embeds,
            "prompt_embeds": vlm_embeds,
            "negative_prompt_embeds": negative_vlm_embeds,
            "start_frame_latents": start_frame_latents,
            "fourier_features": None,
            "start_frame_latents_proj": None,
        }

        latents = self.scheduler.sample(model=self.predictor, latents=latents, **model_kwargs)

        latents = latents.to(self.vae.dtype)
        video = self.decode_latents(latents)
        return video

pipeline = {
    'flashi2v_qwenvl': FlashI2VInferencePipeline
}