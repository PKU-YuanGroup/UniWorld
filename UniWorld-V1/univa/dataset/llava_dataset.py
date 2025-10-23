from typing import Any, Callable, Optional, List

import torch
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import os
from PIL import Image
from univa.utils.prompter import Prompter
import numpy as np
from einops import rearrange
import random
from univa.utils.constant import SPACIAL_TOKEN, GENERATE_TOKEN

class LlavaDataset(Dataset):
    def __init__(
        self,
        dataset_type: str,
        data_txt: str,
        tokenizer: PreTrainedTokenizer,
        prompter: Prompter,
        image_processor: Callable,
        processor: Callable = None,
        min_pixels: int = 384*384, 
        max_pixels: int = 384*384, 
        image_token_length: int = 729,
        only_generated_task: bool = False,
        drop_prompt_rate: float = 0.2,
    ):
        assert dataset_type == 'llava'
        with open(data_txt, "r") as f:
            self.datasets = [line.strip() for line in f.readlines()]

        self.data = []
        self._load_data()
        self.tokenizer = tokenizer
        self.prompter = prompter
        self.image_token_length = image_token_length
        self.image_token = SPACIAL_TOKEN[dataset_type]['image_token']
        self.image_begin_token = SPACIAL_TOKEN[dataset_type]['image_begin_token']
        self.image_end_token = SPACIAL_TOKEN[dataset_type]['image_end_token']
        self.generated_image_token = GENERATE_TOKEN
        self.image_processor = image_processor

        self.only_generated_task = only_generated_task  # For denoiser training
        self.drop_prompt_rate = drop_prompt_rate
        if self.drop_prompt_rate > 0:
            assert self.only_generated_task, (
                "Only generated task is supported when drop prompt rate is greater than 0"
            )

        # Add image token if not exists.
        if self.image_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_special_tokens(
                {"additional_special_tokens": [self.image_token]}
            )
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

        self.image_begin_token_id = self.tokenizer.convert_tokens_to_ids(
            self.image_begin_token
        )
        assert isinstance(self.image_begin_token_id, int), (
            f"tokenizer miss image begin token `{self.image_begin_token}`"
        )
        self.image_end_token_id = self.tokenizer.convert_tokens_to_ids(
            self.image_end_token
        )
        assert isinstance(self.image_end_token_id, int), (
            f"tokenizer miss image end token `{self.image_end_token}`"
        )

    def _load_data(self):
        for dataset in self.datasets:
            image_root, json_file = dataset.split(",")

            # Load json file
            with open(json_file, "r") as f:
                data = json.load(f)

            dataset_data = []
            for line in tqdm(data):
                # Ensure `image` is a list
                if isinstance(line["image"], str):
                    line["image"] = [line["image"]]
                assert isinstance(line["image"], list), (
                    "`image` must be a str or a list."
                )

                # Convert image path to absolute path
                line["image"] = [
                    os.path.join(image_root, image_path) for image_path in line["image"]
                ]

                dataset_data.append(line)

            print(f"Load {len(dataset_data)} data from {json_file}.")
            self.data.extend(dataset_data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            data: Any = self.data[idx]

            # Reformat the conversation to the format of prompter
            conversations = []
            prompt = ""
            for item in data["conversations"]:
                if item["from"] == "human":
                    role = self.prompter.user_role
                elif item["from"] == "gpt":
                    role = self.prompter.assistant_role
                else:
                    raise ValueError(f"Unknown role: {item['from']}")
                conversations.append({"from": role, "value": item["value"]})
            assert prompt != ""

            # Make prompt
            drop_condition = False
            if self.only_generated_task:
                if self.drop_prompt_rate < random.random():  # Randomly drop the prompt
                    prompt_list = self.prompter.get_train_prompt(conversations)
                else:
                    drop_condition = True
                    # Drop the prompt
                    prompt_list = [
                        {
                            "from": self.prompter.system_role,
                            "value": "You are a helpful assistant.",
                        },
                        {
                            "from": self.prompter.user_role,
                            "value": "Generate an image.",
                        },
                        {
                            "from": self.prompter.assistant_role,
                            "value": self.generated_image_token,
                        },
                    ]
                    prompt_list = self.prompter.get_train_prompt(prompt_list)
            else:
                prompt_list = self.prompter.get_train_prompt(conversations)
                
            input_ids = []
            labels = []
            has_generated_image = False
            for item in prompt_list:
                item["prompt"] = item["prompt"].replace('<image>', self.image_token)
                if self.generated_image_token in item["prompt"]:  # Check if self.generated_image_token in prompt
                    assert item["from"] == self.prompter.assistant_role, (
                        "Generated image token must be in assistant role"
                    )
                    assert (
                        f"{self.generated_image_token}{self.prompter.eos_token}"
                        in item["prompt"]
                    ), "Generated image token must in end of prompt"

                    # Replace the generated image token with image begin token and without eos token
                    item["prompt"] = item["prompt"].replace(
                        f"{self.generated_image_token}{self.prompter.eos_token}",
                        self.image_begin_token,
                    )
                    has_generated_image = True

                tokenized_item = self.tokenizer(
                    item["prompt"],
                    return_tensors="pt",
                    truncation=False,
                )
                if item["is_labels"]:  # If this prompt is labels
                    labels.append(tokenized_item.input_ids)
                else:
                    labels.append(torch.full_like(tokenized_item.input_ids, -100))
                input_ids.append(tokenized_item.input_ids)

            if (
                self.only_generated_task and not has_generated_image
            ):  # For denoiser training
                raise ValueError(
                    f"Only generated task is not supported. But this prompt not contains generated image token: {prompt_list[0]['prompt']}"
                )

            input_ids = torch.cat(input_ids, dim=1)
            labels = torch.cat(labels, dim=1)

            # Load images
            if has_generated_image:
                if not drop_condition:
                    image_slice = data["image"][:-1]
                else:
                    image_slice = []
            else:
                image_slice = data["image"]
            image_dict = self._load_image(image_slice, image_processor=self.image_processor, image_token_lengths=self.image_token_length)
            image_token_lengths = image_dict['image_token_lengths']
            pixel_values = image_dict['pixel_values']
            image_grid_thw = image_dict['image_grid_thw']


            # Repeat the image token to the length of image_token_length 
            # and record the position of image tokens.
            input_ids, labels, image_position = self._process_image_token(
                input_ids,
                labels=labels,
                image_token_id=self.image_token_id,
                image_begin_token_id=self.image_begin_token_id,
                image_end_token_id=self.image_end_token_id,
                image_token_lengths=image_token_lengths,
            )

            return_data = {
                "input_ids": input_ids,
                "labels": labels,
                "pixel_values": pixel_values,
                "image_position": image_position,
                "image_grid_thw": image_grid_thw, 
                "prompt": [prompt],
            }

            if has_generated_image: # If this item is a generation task
                image = Image.open(data["image"][-1]).convert("RGB")
                image_tensor = torch.tensor(np.array(image)) / 255.0  # scale to 0-1
                image_tensor = rearrange(image_tensor, "h w c -> c h w")
                return_data["generated_image"] = image_tensor

            return return_data
        except Exception as e:
            print(f'Error with {e}')
            return self.__getitem__(random.randint(0, self.__len__()-1))

    @staticmethod
    def _load_image(
        image_slice: List[str],
        max_pixels: int = 448*448,  
        min_pixels: int = 448*448, 
        processor: Callable = None, 
        image_processor: Callable = None, 
        image_token_lengths: int = 729, 
        image_token: str = '<image>', 
    ):
        # images tensor shape is (b, c, h, w)
        images = []
        # Ignore the last image (generated image)
        for image_path in image_slice: # Ignore the last image (generated image)
            image = Image.open(image_path).convert("RGB")
            image = image_processor(
                image, return_tensors="pt"
            ).pixel_values
            images.append(image)
        if len(images) > 0:
            images = torch.cat(images)
        image_token_lengths = len(images) * [image_token_lengths]
        return {'pixel_values': images, 'image_grid_thw': [], 'image_token_lengths': image_token_lengths}
    
    @staticmethod
    def _process_image_token(
        input_ids: torch.Tensor,
        image_token_id: int,
        image_begin_token_id: int,
        image_end_token_id: int,
        image_token_lengths: List[int],
        labels: Optional[torch.Tensor] = None,
    ):
        # Find the indices of the image token
        image_token_indices = (input_ids == image_token_id).nonzero(as_tuple=True)
        image_position = []
        offset = 0
        cur_i = 0
        if isinstance(image_token_lengths, int):
            image_token_lengths = [image_token_lengths] * len(image_token_indices[1])
        for idx in image_token_indices[1]:
            image_token_length = image_token_lengths[cur_i]
            adjusted_idx = idx + offset
            assert input_ids[0, adjusted_idx] == image_token_id

            # Add image begin and end token
            input_ids = torch.cat(
                [
                    input_ids[:, :adjusted_idx],
                    input_ids.new_full(
                        (1, 1), image_begin_token_id
                    ),  # image begin token
                    input_ids.new_full(
                        (1, image_token_length), image_token_id
                    ),  # Repeat the image token to the length of image_token_length
                    input_ids.new_full((1, 1), image_end_token_id),  # image end token
                    input_ids[:, adjusted_idx + 1 :],
                ],
                dim=1,
            )
            if labels is not None:
                labels = torch.cat(
                    [
                        labels[:, :adjusted_idx],
                        labels.new_full(
                            (1, 1), image_begin_token_id
                        ),  # Make begin token as label
                        labels.new_full((1, image_token_length), -100),
                        labels.new_full((1, 1), -100),
                        labels[:, adjusted_idx + 1 :],
                    ],
                    dim=1,
                )

            adjusted_idx += 1  # skip the image begin token
            image_position.append(adjusted_idx.item())
            offset += image_token_length - 1
            offset += 2  # begin and end token

        return input_ids, labels, image_position
