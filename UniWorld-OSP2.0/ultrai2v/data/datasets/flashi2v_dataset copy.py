# Copyright (c) 2024 Huawei Technologies Co., Ltd.


import os
import torch    
import random
from concurrent.futures import ThreadPoolExecutor
import copy

from torch.utils.data import Dataset
import numpy as np
from transformers import AutoTokenizer
from ultrai2v.utils.constant import VIDEO, PROMPT, PROMPT_IDS, PROMPT_MASK, START_FRAME, NAME_INDEX
from ultrai2v.data.utils.utils import LMDBReader
from ultrai2v.data.utils.image_reader import is_image_file
from ultrai2v.data.utils.video_reader import is_video_file
from ultrai2v.data.datasets.t2v_dataset import WanT2VDataset, T2VRandomDataset, T2VEvalDataset
from ultrai2v.data.utils.wan_utils import WanVideoProcessor, WanImageProcessor

FlashI2VOutputData = {
    PROMPT_IDS: None,
    PROMPT_MASK: None,
    START_FRAME: None,
    VIDEO: None,
}

I2VEvalOutputData = {
    PROMPT: None,
    START_FRAME: None,
    NAME_INDEX: None,
}

class FlashI2VDataset(WanT2VDataset):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
            
        orig_video = self.get_video_data(video_path, meta_info)
        examples[VIDEO] = orig_video
        examples[START_FRAME] = orig_video[:, 0:1, :, :].clone()

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