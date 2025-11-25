# Copyright (c) 2024 Huawei Technologies Co., Ltd.


import os
import torch    
import random
from concurrent.futures import ThreadPoolExecutor
import copy

from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer
from ultrai2v.utils.constant import VIDEO, PROMPT, PROMPT_IDS, PROMPT_MASK, START_FRAME, NAME_INDEX, VLM_INPUTS
from ultrai2v.data.utils.utils import LMDBReader
from ultrai2v.data.utils.image_reader import is_image_file
from ultrai2v.data.utils.video_reader import is_video_file
from ultrai2v.data.datasets.t2v_dataset import WanT2VDataset, T2VRandomDataset, T2VEvalDataset
from ultrai2v.data.utils.wan_utils import WanVideoProcessor, WanImageProcessor

# FIXME  Qwenvl
from transformers import  AutoTokenizer, AutoProcessor # Qwen2_5_VLTextModel,
from qwen_vl_utils import process_vision_info
from PIL import Image
from torchvision.transforms import functional as F

FlashI2VOutputData = {
    PROMPT_IDS: None,
    PROMPT_MASK: None,
    START_FRAME: None,
    VIDEO: None,
    VLM_INPUTS: None,
}

I2VEvalOutputData = {
    PROMPT: None,
    START_FRAME: None,
    NAME_INDEX: None,
}

class FlashI2VDataset(WanT2VDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.vlm_processor = AutoProcessor.from_pretrained("/work/share/projects/gyy/zhubin_FlashI2V/checkpoints/Qwen2.5-VL-7B-Instruct")

    
    def save_tensor_as_image(self, first_frame, filename="output_frame.png"):
        # 3,h,w
        # import ipdb; ipdb.set_trace()
        first_frame = first_frame.permute(1, 2, 0).numpy().astype(np.uint8) # (3,h,w)->(h,w,3)
        # 4. 使用 PIL 保存图像
        try:
            image = Image.fromarray(first_frame)
            image.save(filename)
            # print(f"✅ 图像保存成功！文件路径: {os.path.abspath(filename)}")
        except Exception as e:
            print(f"❌ 图像保存失败: {e}")


    def save_tensor_as_image_0_255(self, first_frame, filename="output_frame_resized.png", target_size=(240, 416)):
        """
        将 (3, H, W) 形状的 Tensor (值域为 0-255) 保存为图像，并在保存前进行 Resize 和 CenterCrop。
        
        Args:
            first_frame (torch.Tensor): 输入图像 Tensor，形状为 (3, H, W)，值域为 [0, 255]。
            filename (str): 输出文件名。
            target_size (tuple): 目标输出图像的 (高度, 宽度)。
        """
        
        # --- 步骤 1: 准备浮点输入供几何变换使用 ---
        
        # F.resize 和 F.center_crop 需要浮点输入。
        # 如果输入是 uint8，我们将其转换为 float32。
        if first_frame.dtype == torch.uint8:
            frame_float = first_frame.float()
        else:
            # 假设已经是 float32 或 float64
            frame_float = first_frame
            
        # --- 步骤 2: 图像处理 (Resize/Crop) ---
        
        # Resize：调整大小到目标尺寸（使用双三次插值）
        resized_frame = F.resize(
            frame_float, 
            size=target_size,
            interpolation=F.InterpolationMode.BICUBIC 
        )
        
        # Center Crop：中心裁剪
        cropped_frame = F.center_crop(resized_frame, output_size=target_size)
        
        # --- 步骤 3: 转换为 NumPy 数组并调整通道顺序 ---
        
        # 3a. 钳位 (Clamp) 确保值在 0 到 255 之间，并转换为 uint8
        # 由于插值操作可能产生略微超出 [0, 255] 的值，钳位是安全做法。
        # 使用 .round() 四舍五入到最近的整数。
        final_frame_tensor = cropped_frame.clamp(0, 255).round().to(torch.uint8)

        # 3b. (3, H, W) -> (H, W, 3) 并转换为 NumPy
        final_frame_np = final_frame_tensor.permute(1, 2, 0).cpu().numpy()
        
        # --- 步骤 4: 使用 PIL 保存图像 ---
        try:
            image = Image.fromarray(final_frame_np)
            image.save(filename)
            # print(f"✅ 图像保存成功！文件路径: {os.path.abspath(filename)}")
        except Exception as e:
            print(f"❌ 图像保存失败: {e}")


    def getitem(self, index):
        # init output data
        examples = copy.deepcopy(FlashI2VOutputData)
        meta_info = self.dataset_reader.getitem(index)
        text = meta_info["cap"]
        video_path = meta_info["path"]

        drop_text = False
        rand_num = random.random()
        if rand_num < self.text_drop_ratio:
            drop_text = True

        examples[PROMPT_IDS], examples[PROMPT_MASK] = self.get_text_data(text, drop=drop_text)
        
        first_frame_3hw, orig_video = self.get_video_data(video_path, meta_info) # first_frame_3hw是没有经过transforms的
        examples[VIDEO] = orig_video
        examples[START_FRAME] = orig_video[:, 0:1, :, :].clone()

        # Qwenvl
        start_frame_path = video_path.replace('.mp4', '_1f.jpg')
        # self.save_tensor_as_image(first_frame_3hw, filename=start_frame_path)
        self.save_tensor_as_image_0_255(first_frame_3hw, filename=start_frame_path)
        # ==============
        # 编码、截断并填充到 512 长度
        if drop_text:
            text = ''
        encoded_inputs = self.vlm_processor.tokenizer(
            text,
            padding="max_length",  # 填充到最大长度
            max_length=512, # 目标长度
            truncation=True,       # 启用截断
            return_tensors="pt"    # 返回 PyTorch Tensor
        )
        input_ids = encoded_inputs['input_ids'].squeeze()[:512]
        # 将截断后的 ID 序列解码回人类可读的文本
        text = self.vlm_processor.tokenizer.decode(
            input_ids,
            skip_special_tokens=True # 移除 [CLS], [SEP], [PAD] 等特殊 token
        )
        # print(text)
        # ==============
        vlm_messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "image": start_frame_path,
                            },
                            {"type": "text", "text": text},
                        ],
                    }
                ]
        # Preparation for inference
        vlm_text = self.vlm_processor.apply_chat_template(
            vlm_messages, tokenize=False, add_generation_prompt=True
        )
        vlm_image_inputs, vlm_video_inputs = process_vision_info(vlm_messages)
        vlm_inputs = self.vlm_processor(
            text=[vlm_text],
            images=vlm_image_inputs,
            videos=vlm_video_inputs,
            padding="max_length",  # 填充到最大长度
            max_length=1024, # 目标长度
            return_tensors="pt",
        )
        examples[VLM_INPUTS] = vlm_inputs

        return examples

    def get_text_data(self, text, drop=False):
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)][0]
        if drop:
            text = ""
        prompt_input_ids, prompt_mask = self.text_processor(text)
        return prompt_input_ids, prompt_mask
        

class I2VRandomDataset(T2VRandomDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def getitem(self, index):
        # init output data
        examples = copy.deepcopy(FlashI2VOutputData)
        text = ""
        examples[PROMPT_IDS], examples[PROMPT_MASK] = self.get_text_data(text)
        orig_video = torch.randn(3, self.sample_num_frames, self.sample_height, self.sample_width)
        examples[VIDEO] = orig_video
        examples[START_FRAME] = orig_video[:, 0:1, :, :].clone()
        return examples

class I2VEvalDataset(T2VEvalDataset):
    def __init__(
        self,
        metafile_or_dir_path,
        sample_height=480,
        sample_width=832,
        sample_num_frames=49,
        train_fps=16,
        num_samples_per_prompt=1,
        **kwargs,
    ):
        super().__init__(
            metafile_or_dir_path,
            sample_height=sample_height,
            sample_width=sample_width,
            sample_num_frames=sample_num_frames,
            train_fps=train_fps,
            num_samples_per_prompt=num_samples_per_prompt,
            **kwargs,
        )

        self.is_image = is_image_file(self.dataset_reader.getitem(0)["path"])
        self.is_video = is_video_file(self.dataset_reader.getitem(0)["path"])

        if self.is_video:
            print(f"Using video mode, sample_height: {self.sample_height}, sample_width: {self.sample_width}, sample_num_frames: {self.sample_num_frames}")
            self.visual_processor = WanVideoProcessor(
                video_layout_type='TCHW',
                sample_height=self.sample_height,
                sample_width=self.sample_width,
                sample_num_frames=self.sample_num_frames,
                train_fps=self.train_fps,
                force_cut_video_from_start=True,
            )
        elif self.is_image:
            print(f"Using image mode, sample_height: {self.sample_height}, sample_width: {self.sample_width}")
            self.visual_processor = WanImageProcessor(
                image_layout_type='CHW',
                sample_height=self.sample_height,
                sample_width=self.sample_width,
            )
        else:
            raise ValueError("Must specify either video or image")

    def getitem(self, index):
        video_index = index // self.num_samples_per_prompt
        local_index = index % self.num_samples_per_prompt
        examples = copy.deepcopy(I2VEvalOutputData)
        meta_info = self.dataset_reader.getitem(video_index)
        text = meta_info["cap"]
        item_path = meta_info["path"]
        examples[PROMPT] = self.get_text_data(text)
        examples[START_FRAME] = self.get_visual_data(item_path, meta_info)
        examples[NAME_INDEX] = f"video_{video_index:06d}_{local_index:06d}"
        return examples
    
    def get_visual_data(self, path, meta_info):
        visual = self.visual_processor(path, meta_info, need_processing=False)
        if self.is_video:
            visual = visual[0] # only use the start frame
        return visual


dataset = {
    'flashi2v': FlashI2VDataset,
    'i2v_random': I2VRandomDataset,
    'i2v_eval': I2VEvalDataset,
}
 