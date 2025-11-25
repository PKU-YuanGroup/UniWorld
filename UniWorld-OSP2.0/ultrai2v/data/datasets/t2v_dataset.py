# Copyright (c) 2024 Huawei Technologies Co., Ltd.


import os
import torch    
import random
from concurrent.futures import ThreadPoolExecutor
import copy

import numpy as np
from transformers import AutoTokenizer
from ultrai2v.utils.constant import VIDEO, PROMPT, PROMPT_IDS, PROMPT_MASK, NAME_INDEX
from ultrai2v.data.utils.utils import LMDBReader
from ultrai2v.data.utils.wan_utils import WanTextProcessor, WanVideoProcessor
from ultrai2v.data.datasets.base_dataset import BaseDataset

T2VOutputData = {
    PROMPT_IDS: None,
    PROMPT_MASK: None,
    VIDEO: None,
}

T2VEvalOutputData = {
    PROMPT: None,
    NAME_INDEX: None,
}

class WanT2VDataset(BaseDataset):

    def __init__(
        self,
        metafile_or_dir_path,
        text_tokenizer_path,
        sample_height=480,
        sample_width=832,
        sample_num_frames=49,
        train_fps=16,
        sample_stride=None,
        text_drop_ratio=0.1,
        text_max_length=512,
        return_prompt_mask=True,
        **kwargs,
    ):
        self.dataset_reader = LMDBReader(metafile_or_dir_path)
        self.data_length = len(self.dataset_reader)
        print(f'Build WanT2VDataset, data length: {self.data_length}...')
        
        self.sample_height = sample_height
        self.sample_width = sample_width
        self.sample_num_frames = sample_num_frames

        self.train_fps = train_fps
        self.sample_stride = sample_stride

        self.sample_mode = None
        if self.train_fps is not None:
            print(f"Using train_fps mode, train_fps: {self.train_fps}")
            self.sample_mode = "train_fps"
        elif self.sample_stride is not None:
            print(f"Using sample_stride mode, sample_stride: {self.sample_stride}")
            self.sample_mode = "sample_stride"
        else:
            raise ValueError("Must specify either train_fps or sample_stride")

        self.text_drop_ratio = text_drop_ratio
        self.text_max_length = text_max_length
        self.return_prompt_mask = return_prompt_mask
        self.text_processor = WanTextProcessor(
            tokenizer=AutoTokenizer.from_pretrained(text_tokenizer_path),
            model_max_length=self.text_max_length,
            return_prompt_mask=self.return_prompt_mask,
        )

        self.video_processor = WanVideoProcessor(
            sample_height=self.sample_height,
            sample_width=self.sample_width,
            sample_num_frames=self.sample_num_frames,
            train_fps=self.train_fps,
            sample_stride=self.sample_stride,
        )

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.timeout = kwargs.get("timeout", 600) 


    def __getitem__(self, index):
        try:
            future = self.executor.submit(self.getitem, index)
            data = future.result(timeout=self.timeout) 
            return data
        except Exception as e:
            print(f"the error is {e}")
            return self.__getitem__(np.random.randint(0, self.__len__() - 1))
        # return self.getitem(index)

    def __len__(self):
        return self.data_length

    def getitem(self, index):
        # init output data
        examples = copy.deepcopy(T2VOutputData)
        meta_info = self.dataset_reader.getitem(index)
        text = meta_info["cap"]
        video_path = meta_info["path"]
        examples[PROMPT_IDS], examples[PROMPT_MASK] = self.get_text_data(text)
        examples[VIDEO] = self.get_video_data(video_path, meta_info)
        return examples


    def get_video_data(self, video_path, meta_info):
        first_frame_3hw, video = self.video_processor(video_path, meta_info)
        return first_frame_3hw, video

    
    def get_text_data(self, text):
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)][0]
        if random.random() < self.text_drop_ratio:
            text = ""
        prompt_input_ids, prompt_mask = self.text_processor(text)
        return prompt_input_ids, prompt_mask
        

class T2VRandomDataset(BaseDataset):
    def __init__(
        self,
        text_tokenizer_path,
        sample_height=480,
        sample_width=832,
        sample_num_frames=49,
        text_max_length=512,
        return_prompt_mask=True,
        **kwargs,
    ):
        self.sample_height = sample_height
        self.sample_width = sample_width
        self.sample_num_frames = sample_num_frames

        self.text_max_length = text_max_length
        self.return_prompt_mask = return_prompt_mask
        self.text_processor = WanTextProcessor(
            tokenizer=AutoTokenizer.from_pretrained(text_tokenizer_path),
            model_max_length=self.text_max_length,
            return_prompt_mask=self.return_prompt_mask,
        )

    def __len__(self):
        return 1000000

    def __getitem__(self, index):
        return self.getitem(index)

    def getitem(self, index):
        # init output data
        examples = copy.deepcopy(T2VOutputData)
        text = ""
        examples[PROMPT_IDS], examples[PROMPT_MASK] = self.get_text_data(text)
        examples[VIDEO] = torch.randn(3, self.sample_num_frames, self.sample_height, self.sample_width)
        return examples

    def get_text_data(self, text):
        prompt_input_ids, prompt_mask = self.text_processor(text)
        return prompt_input_ids, prompt_mask


class T2VEvalDataset(BaseDataset):
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
        self.dataset_reader = LMDBReader(metafile_or_dir_path)
        self.num_samples_per_prompt = num_samples_per_prompt
        self.data_length = len(self.dataset_reader) * self.num_samples_per_prompt
        print(f'Build T2VEvalDataset, data length: {self.data_length}...')

        self.sample_height = sample_height
        self.sample_width = sample_width
        self.sample_num_frames = sample_num_frames
        self.train_fps = train_fps

        self.executor = ThreadPoolExecutor(max_workers=1)
        self.timeout = kwargs.get("timeout", 60) 

    def __getitem__(self, index):
        # try:
        #     future = self.executor.submit(self.getitem, index)
        #     data = future.result(timeout=self.timeout) 
        #     return data
        # except Exception as e:
        #     print(f"the error is {e}")
        #     return self.__getitem__(np.random.randint(0, self.__len__() - 1))
        return self.getitem(index)

    def __len__(self):
        return self.data_length

    def getitem(self, index):
        video_index = index // self.num_samples_per_prompt
        local_index = index % self.num_samples_per_prompt
        examples = copy.deepcopy(T2VEvalOutputData)
        meta_info = self.dataset_reader.getitem(video_index)
        text = meta_info["cap"]
        examples[PROMPT] = self.get_text_data(text)
        examples[NAME_INDEX] = f"video_{video_index:06d}_{local_index:06d}"
        return examples

    def get_text_data(self, text):
        if not isinstance(text, list):
            text = [text]
        text = [random.choice(text)][0]
        return text

dataset = {
    'wan_t2v': WanT2VDataset,
    't2v_random': T2VRandomDataset,
    't2v_eval': T2VEvalDataset,
}