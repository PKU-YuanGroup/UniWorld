# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from collections import defaultdict
import os
import datetime
from concurrent import futures
import time
import json
from absl import app, flags
import logging
from diffusers import FluxKontextPipeline
from transformers.models.clip.modeling_clip import CLIPEncoderLayer
from transformers.models.t5.modeling_t5 import T5Block
import numpy as np
import flow_grpo.rewards
from flow_grpo.stat_tracking import PerPromptStatTracker
from flow_grpo.diffusers_patch.kontext_pipeline_with_logprob import (
    pipeline_with_logprob,
)
from flow_grpo.diffusers_patch.train_dreambooth_lora_flux import encode_prompt
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import wandb
from functools import partial
import tqdm
import tempfile
from PIL import Image
from peft import LoraConfig, get_peft_model, PeftModel
import random
from torch.utils.data import Dataset, DataLoader, Sampler
from flow_grpo.ema import EMAModuleWrapper
from flow_grpo.fsdp2_utils import prepare_fsdp_model
from ml_collections import config_flags
from torch.cuda.amp import GradScaler, autocast as torch_autocast
import time

tqdm = partial(tqdm.tqdm, dynamic_ncols=True)


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "config/base.py", "Training configuration.")

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)


def setup_distributed(rank, lock_rank, world_size):
    os.environ["MASTER_ADDR"] = os.getenv("MASTER_ADDR", "localhost")
    os.environ["MASTER_PORT"] = os.getenv("MASTER_PORT", "12355")
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(lock_rank)


def cleanup_distributed():
    dist.destroy_process_group()


def is_main_process(rank):
    return rank == 0


def set_seed(seed: int, rank: int = 0):
    random.seed(seed + rank)
    np.random.seed(seed + rank)
    torch.manual_seed(seed + rank)
    torch.cuda.manual_seed_all(seed + rank)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PromptImageDataset(Dataset):
    def __init__(self, dataset, resolution=512, split="train"):
        self.dataset = dataset
        self.resolution = resolution
        self.file_path = os.path.join(dataset, f"{split}_metadata.jsonl")
        with open(self.file_path, "r", encoding="utf-8") as f:
            self.metadatas = [json.loads(line) for line in f]
            self.prompts = [item["prompt"] for item in self.metadatas]

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, idx):
        item = {"prompt": self.prompts[idx], "metadata": self.metadatas[idx]}
        # Assuming 'image' in metadata contains a path to the image file
        image_path = self.metadatas[idx]["image"]
        item["prompt_with_image_path"] = f"{self.prompts[idx]}_{image_path}"
        image = Image.open(os.path.join(self.dataset, image_path)).convert("RGB")
        w, h = image.size
        if w != h:
            min_dim = min(w, h)
            left = (w - min_dim) // 2
            top = (h - min_dim) // 2
            right = left + min_dim
            bottom = top + min_dim
            image = image.crop((left, top, right, bottom))
        image = image.resize(
            (self.resolution, self.resolution), Image.Resampling.LANCZOS
        )
        item["image"] = image
        return item

    @staticmethod
    def collate_fn(examples):
        prompts = [example["prompt"] for example in examples]
        metadatas = [example["metadata"] for example in examples]
        images = [example["image"] for example in examples]
        prompt_with_image_paths = [
            example["prompt_with_image_path"] for example in examples
        ]
        return prompts, metadatas, images, prompt_with_image_paths


class DistributedKRepeatSampler(Sampler):
    def __init__(
        self, dataset, batch_size, k, num_replicas, rank, seed=0, banned_prompts=None
    ):
        self.dataset = dataset
        self.batch_size = batch_size
        self.k = k  # k means the number of images per prompt
        self.num_replicas = num_replicas
        self.rank = rank
        self.seed = seed
        self.total_samples = self.num_replicas * self.batch_size
        assert (
            self.total_samples % self.k == 0
        ), f"k can not div n*b, k{k}-num_replicas{num_replicas}-batch_size{batch_size}"
        self.m = self.total_samples // self.k
        self.epoch = 0

        self.banned_prompts = banned_prompts if banned_prompts is not None else set()
        self.last_banned_prompts_len = len(self.banned_prompts)
        self.valid_indices_cache = None

    def get_valid_indices(self):
        start_time = time.time()
        if self.valid_indices_cache is None or self.last_banned_prompts_len != len(self.banned_prompts):
            self.valid_indices_cache = [
                i for i, prompt in enumerate(self.dataset.prompts)
                if prompt not in self.banned_prompts
            ]
            self.last_banned_prompts_len = len(self.banned_prompts)
        print("get valid indices time: ", time.time() - start_time)
        return self.valid_indices_cache

    def __iter__(self):
        while True:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            valid_indices = self.get_valid_indices()
            if len(valid_indices) < self.m:
                # TODO
                raise NotImplementedError()

            indices = torch.tensor(valid_indices)[
                torch.randperm(len(valid_indices), generator=g)[: self.m]
            ].tolist()
            repeated_indices = [idx for idx in indices for _ in range(self.k)]
            shuffled_indices = torch.randperm(
                len(repeated_indices), generator=g
            ).tolist()
            shuffled_samples = [repeated_indices[i] for i in shuffled_indices]

            per_card_samples = []
            for i in range(self.num_replicas):
                start = i * self.batch_size
                end = start + self.batch_size
                per_card_samples.append(shuffled_samples[start:end])
            yield per_card_samples[self.rank]

    def set_epoch(self, epoch):
        self.epoch = epoch


def gather_tensor_to_all(tensor, world_size):
    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)
    return torch.cat(gathered_tensors, dim=0).cpu()


def compute_text_embeddings(
    prompt, text_encoders, tokenizers, max_sequence_length, device
):
    with torch.no_grad():
        prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
            text_encoders, tokenizers, prompt, max_sequence_length
        )
        prompt_embeds = prompt_embeds.to(device)
        pooled_prompt_embeds = pooled_prompt_embeds.to(device)
    return prompt_embeds, pooled_prompt_embeds, text_ids


def return_decay(step, decay_type):
    if decay_type == 0:
        flat = 0
        uprate = 0.0
        uphold = 0.0
    elif decay_type == 1:
        flat = 0
        uprate = 0.001
        uphold = 0.5
    elif decay_type == 2:
        flat = 75
        uprate = 0.0075
        uphold = 0.999
    else:
        assert False

    if step < flat:
        return 0.0
    else:
        decay = (step - flat) * uprate
        return min(decay, uphold)


def calculate_zero_std_ratio(prompts, gathered_rewards):
    prompt_array = np.array(prompts)
    unique_prompts, inverse_indices, counts = np.unique(
        prompt_array, return_inverse=True, return_counts=True
    )
    grouped_rewards = gathered_rewards["avg"][np.argsort(inverse_indices), 0]
    split_indices = np.cumsum(counts)[:-1]
    reward_groups = np.split(grouped_rewards, split_indices)
    prompt_std_devs = np.array([np.std(group) for group in reward_groups])
    zero_std_count = np.count_nonzero(prompt_std_devs == 0)
    zero_std_ratio = zero_std_count / len(prompt_std_devs)
    return zero_std_ratio, prompt_std_devs.mean()


def eval_fn(
    pipeline,
    test_dataloader,
    text_encoders,
    tokenizers,
    config,
    device,
    rank,
    world_size,
    global_step,
    reward_fn,
    executor,
    mixed_precision_dtype,
    ema,
    transformer_trainable_parameters,
):
    if config.train.ema and ema is not None:
        ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

    pipeline.transformer.eval()
    all_rewards = defaultdict(list)

    test_sampler = (
        DistributedSampler(
            test_dataloader.dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        if world_size > 1
        else None
    )
    eval_loader = DataLoader(
        test_dataloader.dataset,
        batch_size=config.sample.test_batch_size,  # This is per-GPU batch size
        sampler=test_sampler,
        collate_fn=test_dataloader.collate_fn,
        num_workers=test_dataloader.num_workers,
    )

    for test_batch in tqdm(
        eval_loader,
        desc="Eval: ",
        disable=not is_main_process(rank),
        position=0,
    ):
        prompts, prompt_metadata, ref_images, prompt_with_image_paths = test_batch
        prompt_embeds, pooled_prompt_embeds, _ = compute_text_embeddings(
            prompts, text_encoders, tokenizers, max_sequence_length=128, device=device
        )
        current_batch_size = len(prompt_embeds)
        with torch_autocast(
            enabled=(config.mixed_precision in ["fp16", "bf16"]),
            dtype=mixed_precision_dtype,
        ):
            with torch.no_grad():
                images, _, _, _, _, _ = pipeline_with_logprob(
                    pipeline,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    image=ref_images,
                    num_inference_steps=config.sample.eval_num_steps,
                    guidance_scale=config.sample.guidance_scale,
                    output_type="pt",
                    height=config.resolution,
                    width=config.resolution,
                    noise_level=config.sample.noise_level,
                    deterministic=True,
                    solver="flow",
                    max_area=config.resolution**2,
                    _auto_resize=False,
                )

        rewards_future = executor.submit(
            reward_fn, images, prompts, prompt_metadata, ref_images, only_strict=False
        )
        time.sleep(0)
        rewards, reward_metadata = rewards_future.result()

        for key, value in rewards.items():
            rewards_tensor = torch.as_tensor(value, device=device).float()
            gathered_value = gather_tensor_to_all(rewards_tensor, world_size)
            all_rewards[key].append(gathered_value.numpy())

    if is_main_process(rank):
        final_rewards = {
            key: np.concatenate(value_list) for key, value_list in all_rewards.items()
        }

        images_to_log = images.cpu()
        prompts_to_log = prompts

        with tempfile.TemporaryDirectory() as tmpdir:
            num_samples_to_log = min(15, len(images_to_log))
            for idx in range(num_samples_to_log):
                image = images_to_log[idx].float()
                pil = Image.fromarray(
                    (image.float().numpy().transpose(1, 2, 0) * 255).astype(np.uint8)
                )
                pil = pil.resize((config.resolution, config.resolution))
                pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

            sampled_prompts_log = [prompts_to_log[i] for i in range(num_samples_to_log)]
            sampled_rewards_log = [
                {k: final_rewards[k][i] for k in final_rewards}
                for i in range(num_samples_to_log)
            ]

            wandb.log(
                {
                    "eval_images": [
                        wandb.Image(
                            os.path.join(tmpdir, f"{idx}.jpg"),
                            caption=f"{prompt:.1000} | "
                            + " | ".join(
                                f"{k}: {v:.2f}" for k, v in reward.items() if v != -10
                            ),
                        )
                        for idx, (prompt, reward) in enumerate(
                            zip(sampled_prompts_log, sampled_rewards_log)
                        )
                    ],
                    **{
                        f"eval_reward_{key}": np.mean(value[value != -10])
                        for key, value in final_rewards.items()
                    },
                },
                step=global_step,
            )

    if config.train.ema and ema is not None:
        ema.copy_temp_to(transformer_trainable_parameters)

    if world_size > 1:
        dist.barrier()


def save_ckpt(
    save_dir,
    transformer_ddp,
    global_step,
    rank,
    ema,
    transformer_trainable_parameters,
    config,
    optimizer,
    scaler,
):
    if is_main_process(rank):
        save_root = os.path.join(save_dir, "checkpoints", f"checkpoint-{global_step}")
        save_root_lora = os.path.join(save_root, "lora")
        os.makedirs(save_root_lora, exist_ok=True)

        model_to_save = transformer_ddp.module

        if config.train.ema and ema is not None:
            ema.copy_ema_to(transformer_trainable_parameters, store_temp=True)

        model_to_save.save_pretrained(save_root_lora)  # For LoRA/PEFT models

        torch.save(optimizer.state_dict(), os.path.join(save_root, "optimizer.pt"))
        if scaler is not None:
            torch.save(scaler.state_dict(), os.path.join(save_root, "scaler.pt"))

        if config.train.ema and ema is not None:
            ema.copy_temp_to(transformer_trainable_parameters)
        logger.info(f"Saved checkpoint to {save_root}")


def main(_):
    config = FLAGS.config

    # --- Distributed Setup ---
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    setup_distributed(rank, local_rank, world_size)
    device = torch.device(f"cuda:{local_rank}")

    unique_id = datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S")
    if not config.run_name:
        config.run_name = unique_id
    else:
        config.run_name += "_" + unique_id

    # --- WandB Init (only on main process) ---
    if is_main_process(rank):
        log_dir = os.path.join(config.logdir, config.run_name)
        os.makedirs(log_dir, exist_ok=True)
        wandb.init(
            project="flow-grpo",
            name=config.run_name,
            config=config.to_dict(),
            dir=log_dir,
        )
    logger.info(f"\n{config}")

    set_seed(config.seed, rank)  # Pass rank for different seeds per process

    # --- Mixed Precision Setup ---
    mixed_precision_dtype = None
    if config.mixed_precision == "fp16":
        mixed_precision_dtype = torch.float16
    elif config.mixed_precision == "bf16":
        mixed_precision_dtype = torch.bfloat16

    enable_amp = mixed_precision_dtype is not None
    scaler = GradScaler(enabled=enable_amp)

    # --- Load pipeline and models ---
    pipeline = FluxKontextPipeline.from_pretrained(config.pretrained.model)
    pipeline.vae.requires_grad_(False)
    pipeline.text_encoder.requires_grad_(False)
    pipeline.text_encoder_2.requires_grad_(False)
    pipeline.transformer.requires_grad_(not config.use_lora)
    text_encoders = [pipeline.text_encoder, pipeline.text_encoder_2]
    tokenizers = [pipeline.tokenizer, pipeline.tokenizer_2]
    pipeline.safety_checker = None
    pipeline.set_progress_bar_config(
        position=1,
        disable=not is_main_process(rank),
        leave=False,
        desc="Timestep",
        dynamic_ncols=True,
    )

    text_encoder_dtype = mixed_precision_dtype if enable_amp else torch.float32

    pipeline.vae.to(device, dtype=torch.float32)  # VAE usually fp32

    prepare_fsdp_model(
        pipeline.text_encoder,
        shard_conditions=[lambda n, m: isinstance(m, (CLIPEncoderLayer,))],
        cpu_offload=False,
        weight_dtype=text_encoder_dtype,
    )

    prepare_fsdp_model(
        pipeline.text_encoder_2,
        shard_conditions=[lambda n, m: isinstance(m, (T5Block,))],
        cpu_offload=False,
        weight_dtype=text_encoder_dtype,
    )

    transformer = pipeline.transformer.to(device)
    # pipeline.transformer.compile(fullgraph=True)
    pipeline.transformer.enable_gradient_checkpointing()

    if config.use_lora:
        target_modules = [
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "attn.to_k",
            "attn.to_out.0",
            "attn.to_q",
            "attn.to_v",
        ]
        transformer_lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            init_lora_weights="gaussian",
            target_modules=target_modules,
        )
        if config.train.lora_path:
            transformer = PeftModel.from_pretrained(transformer, config.train.lora_path)
            transformer.set_adapter("default")
        else:
            transformer = get_peft_model(transformer, transformer_lora_config)
        transformer.add_adapter("old", transformer_lora_config)
        transformer.set_adapter("default")
    transformer_ddp = DDP(
        transformer,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,
    )
    transformer_ddp.module.set_adapter("default")
    transformer_trainable_parameters = list(
        filter(lambda p: p.requires_grad, transformer_ddp.module.parameters())
    )
    transformer_ddp.module.set_adapter("old")
    old_transformer_trainable_parameters = list(
        filter(lambda p: p.requires_grad, transformer_ddp.module.parameters())
    )
    transformer_ddp.module.set_adapter("default")

    if config.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    # --- Optimizer ---
    optimizer_cls = torch.optim.AdamW

    optimizer = optimizer_cls(
        transformer_trainable_parameters,  # Use params from original model for optimizer
        lr=config.train.learning_rate,
        betas=(config.train.adam_beta1, config.train.adam_beta2),
        weight_decay=config.train.adam_weight_decay,
        eps=config.train.adam_epsilon,
    )

    # --- Datasets and Dataloaders ---
    train_dataset = PromptImageDataset(
        config.dataset, config.resolution, "train"
    )
    test_dataset = PromptImageDataset(config.dataset, config.resolution, "test")

    train_sampler = DistributedKRepeatSampler(
        dataset=train_dataset,
        batch_size=config.sample.train_batch_size,  # This is per-GPU batch size
        k=config.sample.num_image_per_prompt,
        num_replicas=world_size,
        rank=rank,
        seed=config.seed,
        banned_prompts=None,  # 初始为空
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=0,
        collate_fn=train_dataset.collate_fn,
        pin_memory=True,
    )

    test_sampler = (
        DistributedSampler(
            test_dataset, num_replicas=world_size, rank=rank, shuffle=False
        )
        if world_size > 1
        else None
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.sample.test_batch_size,  # Per-GPU
        sampler=test_sampler,  # Use distributed sampler for eval
        collate_fn=test_dataset.collate_fn,
        num_workers=0,
        pin_memory=True,
    )

    # --- Prompt Trackering ---
    if config.sample.num_image_per_prompt == 1:
        config.per_prompt_stat_tracking = False
    if config.per_prompt_stat_tracking:
        stat_tracker = PerPromptStatTracker(
            config.sample.global_std,
            config.sample.ban_std_thres,
            config.sample.ban_mean_thres,
        )
    else:
        assert False

    executor = futures.ThreadPoolExecutor(max_workers=8)  # Async reward computation

    # Train!
    samples_per_epoch = (
        config.sample.train_batch_size
        * world_size
        * config.sample.num_batches_per_epoch
    )
    total_train_batch_size = (
        config.train.batch_size * world_size * config.train.gradient_accumulation_steps
    )

    logger.info("***** Running training *****")
    logger.info(f"  Num Epochs = {config.num_epochs}")
    logger.info(f"  Sample batch size per device = {config.sample.train_batch_size}")
    logger.info(f"  Train batch size per device = {config.train.batch_size}")
    logger.info(
        f"  Gradient Accumulation steps = {config.train.gradient_accumulation_steps}"
    )
    logger.info("")
    logger.info(f"  Total number of samples per epoch = {samples_per_epoch}")
    logger.info(
        f"  Total train batch size (w. parallel, distributed & accumulation) = {total_train_batch_size}"
    )
    logger.info(
        f"  Number of gradient updates per inner epoch = {samples_per_epoch // total_train_batch_size}"
    )
    logger.info(f"  Number of inner epochs = {config.train.num_inner_epochs}")

    reward_fn = getattr(flow_grpo.rewards, "multi_score")(
        device, config.reward_fn
    )  # Pass device
    eval_reward_fn = getattr(flow_grpo.rewards, "multi_score")(
        device, config.reward_fn
    )  # Pass device

    # --- Resume from checkpoint ---
    first_epoch = 0
    global_step = 0
    if config.resume_from:
        logger.info(f"Resuming from {config.resume_from}")
        # Assuming checkpoint dir contains lora, optimizer.pt, scaler.pt
        lora_path = os.path.join(config.resume_from, "lora")
        if os.path.exists(lora_path):  # Check if it's a PEFT model save
            transformer_ddp.module.load_adapter(
                lora_path, adapter_name="default", is_trainable=True
            )
            transformer_ddp.module.load_adapter(
                lora_path, adapter_name="old", is_trainable=False
            )
        else:  # Try loading full state dict if it's not a PEFT save structure
            model_ckpt_path = os.path.join(
                config.resume_from, "transformer_model.pt"
            )  # Or specific name
            if os.path.exists(model_ckpt_path):
                transformer_ddp.module.load_state_dict(
                    torch.load(model_ckpt_path, map_location=device)
                )

        opt_path = os.path.join(config.resume_from, "optimizer.pt")
        if os.path.exists(opt_path):
            optimizer.load_state_dict(torch.load(opt_path, map_location=device))

        scaler_path = os.path.join(config.resume_from, "scaler.pt")
        if os.path.exists(scaler_path) and enable_amp:
            scaler.load_state_dict(torch.load(scaler_path, map_location=device))

        # Extract epoch and step from checkpoint name, e.g., "checkpoint-1000" -> global_step = 1000
        try:
            global_step = int(os.path.basename(config.resume_from).split("-")[-1])
            logger.info(
                f"Resumed global_step to {global_step}. Epoch estimation might be needed."
            )
        except ValueError:
            logger.warning(
                f"Could not parse global_step from checkpoint name: {config.resume_from}. Starting global_step from 0."
            )
            global_step = 0

    ema = None
    if config.train.ema:
        ema = EMAModuleWrapper(
            transformer_trainable_parameters,
            decay=0.9,
            update_step_interval=1,
            device=device,
        )

    num_train_timesteps = int(config.sample.num_steps * config.train.timestep_fraction)

    logger.info("***** Running training *****")

    train_iter = iter(train_dataloader)
    optimizer.zero_grad()

    for src_param, tgt_param in zip(
        transformer_trainable_parameters,
        old_transformer_trainable_parameters,
        strict=True,
    ):
        tgt_param.data.copy_(src_param.detach().data)
        assert src_param is not tgt_param

    for epoch in range(first_epoch, config.num_epochs):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        # SAMPLING
        pipeline.transformer.eval()
        samples_data_list = []

        for i in tqdm(
            range(config.sample.num_batches_per_epoch),
            desc=f"Epoch {epoch}: sampling",
            disable=not is_main_process(rank),
            position=0,
        ):
            transformer_ddp.module.set_adapter("default")
            if hasattr(train_sampler, "set_epoch") and isinstance(
                train_sampler, DistributedKRepeatSampler
            ):
                train_sampler.banned_prompts = (
                    stat_tracker.banned_prompts if config.sample.ban_prompt else set()
                )
                train_sampler.set_epoch(epoch * config.sample.num_batches_per_epoch + i)

            prompts, prompt_metadata, ref_images, prompt_with_image_paths = next(
                train_iter
            )

            prompt_embeds, pooled_prompt_embeds, _ = compute_text_embeddings(
                prompts,
                text_encoders,
                tokenizers,
                max_sequence_length=128,
                device=device,
            )
            prompt_ids = tokenizers[0](
                prompts,
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            ).input_ids.to(device)

            if i == 0 and epoch % config.eval_freq == 0 and not config.debug:
                eval_fn(
                    pipeline,
                    test_dataloader,
                    text_encoders,
                    tokenizers,
                    config,
                    device,
                    rank,
                    world_size,
                    global_step,
                    eval_reward_fn,
                    executor,
                    mixed_precision_dtype,
                    ema,
                    transformer_trainable_parameters,
                )

            if (
                i == 0
                and epoch % config.save_freq == 0
                and is_main_process(rank)
                and not config.debug
            ):
                save_ckpt(
                    config.save_dir,
                    transformer_ddp,
                    global_step,
                    rank,
                    ema,
                    transformer_trainable_parameters,
                    config,
                    optimizer,
                    scaler,
                )

            transformer_ddp.module.set_adapter("old")
            with torch_autocast(enabled=enable_amp, dtype=mixed_precision_dtype):
                with torch.no_grad():
                    images, latents, latent_ids, text_ids, image_latents, _ = (
                        pipeline_with_logprob(
                            pipeline,
                            image=ref_images,
                            prompt_embeds=prompt_embeds,
                            pooled_prompt_embeds=pooled_prompt_embeds,
                            num_inference_steps=config.sample.num_steps,
                            guidance_scale=config.sample.guidance_scale,
                            output_type="pt",
                            height=config.resolution,
                            width=config.resolution,
                            noise_level=config.sample.noise_level,
                            deterministic=config.sample.deterministic,
                            solver=config.sample.solver,
                            max_area=config.resolution**2,
                            _auto_resize=False
                        )
                    )
            transformer_ddp.module.set_adapter("default")

            latents = torch.stack(latents, dim=1)
            timesteps = pipeline.scheduler.timesteps.repeat(len(prompts), 1).to(device)

            rewards_future = executor.submit(
                reward_fn,
                images,
                prompts,
                prompt_metadata,
                ref_images,
                only_strict=True,
            )
            time.sleep(0)

            samples_data_list.append(
                {
                    "prompt_ids": prompt_ids,
                    "prompt_embeds": prompt_embeds,
                    "pooled_prompt_embeds": pooled_prompt_embeds,
                    "timesteps": timesteps,
                    "text_ids": text_ids.unsqueeze(0).repeat(len(prompt_ids), 1, 1),
                    "latent_ids": latent_ids.unsqueeze(0).repeat(len(prompt_ids), 1, 1),
                    "next_timesteps": torch.concatenate(
                        [timesteps[:, 1:], torch.zeros_like(timesteps[:, :1])], dim=1
                    ),
                    "latents_clean": latents[:, -1],
                    "image_latents": image_latents,
                    "rewards_future": rewards_future,  # Store future
                }
            )


        for sample_item in tqdm(
            samples_data_list,
            desc="Waiting for rewards",
            disable=not is_main_process(rank),
            position=0,
        ):
            rewards, reward_metadata = sample_item["rewards_future"].result()
            sample_item["rewards"] = {
                k: torch.as_tensor(v, device=device).float() for k, v in rewards.items()
            }
            del sample_item["rewards_future"]

        # Collate samples
        collated_samples = {
            k: (
                torch.cat([s[k] for s in samples_data_list], dim=0)
                if not isinstance(samples_data_list[0][k], dict)
                else {
                    sk: torch.cat([s[k][sk] for s in samples_data_list], dim=0)
                    for sk in samples_data_list[0][k]
                }
            )
            for k in samples_data_list[0].keys()
        }

        # Logging images (main process)
        if epoch % 10 == 0 and is_main_process(rank):
            images_to_log = images.cpu()  # from last sampling batch on this rank
            prompts_to_log = prompts  # from last sampling batch on this rank
            rewards_to_log = collated_samples["rewards"]["avg"][
                -len(images_to_log) :
            ].cpu()

            with tempfile.TemporaryDirectory() as tmpdir:
                num_to_log = min(15, len(images_to_log))
                for idx in range(num_to_log):  # log first N
                    img_data = images_to_log[idx]
                    pil = Image.fromarray(
                        (img_data.float().numpy().transpose(1, 2, 0) * 255).astype(
                            np.uint8
                        )
                    )
                    pil = pil.resize((config.resolution, config.resolution))
                    pil.save(os.path.join(tmpdir, f"{idx}.jpg"))

                wandb.log(
                    {
                        "images": [
                            wandb.Image(
                                os.path.join(tmpdir, f"{idx}.jpg"),
                                caption=f"{prompts_to_log[idx]:.100} | avg: {rewards_to_log[idx]:.2f}",
                            )
                            for idx in range(num_to_log)
                        ],
                    },
                    step=global_step,
                )
        collated_samples["rewards"]["avg"] = (
            collated_samples["rewards"]["avg"]
            .unsqueeze(1)
            .repeat(1, num_train_timesteps)
        )

        # Gather rewards across processes
        gathered_rewards_dict = {}
        for key, value_tensor in collated_samples["rewards"].items():
            gathered_rewards_dict[key] = gather_tensor_to_all(
                value_tensor, world_size
            ).numpy()

        if is_main_process(rank):  # logging
            wandb.log(
                {
                    "epoch": epoch,
                    **{
                        f"reward_{k}": v.mean()
                        for k, v in gathered_rewards_dict.items()
                        if "_strict_accuracy" not in k and "_accuracy" not in k
                    },
                },
                step=global_step,
            )

        if config.per_prompt_stat_tracking:
            prompt_ids_all = gather_tensor_to_all(
                collated_samples["prompt_ids"], world_size
            )
            prompts_all_decoded = pipeline.tokenizer.batch_decode(
                prompt_ids_all.cpu().numpy(), skip_special_tokens=True
            )
            # Stat tracker update expects numpy arrays for rewards
            advantages, stds, means = stat_tracker.update(
                prompts_all_decoded, gathered_rewards_dict["avg"]
            )

            if is_main_process(rank):
                group_size, trained_prompt_num = stat_tracker.get_stats()
                zero_std_ratio, reward_std_mean = calculate_zero_std_ratio(
                    prompts_all_decoded, gathered_rewards_dict
                )
                wandb.log(
                    {
                        "banned_prompt_num": len(stat_tracker.banned_prompts) if config.sample.ban_prompt else 0,
                        "group_size": group_size,
                        "trained_prompt_num": trained_prompt_num,
                        "zero_std_ratio": zero_std_ratio,
                        "reward_std_mean": reward_std_mean,
                        "mean_reward_100": stat_tracker.get_mean_of_top_rewards(100),
                        "mean_reward_75": stat_tracker.get_mean_of_top_rewards(75),
                        "mean_reward_50": stat_tracker.get_mean_of_top_rewards(50),
                        "mean_reward_25": stat_tracker.get_mean_of_top_rewards(25),
                        "mean_reward_10": stat_tracker.get_mean_of_top_rewards(10),
                    },
                    step=global_step,
                )
            stat_tracker.clear()
        else:
            avg_rewards_all = gathered_rewards_dict["avg"]
            advantages = (avg_rewards_all - avg_rewards_all.mean()) / (
                avg_rewards_all.std() + 1e-4
            )
        # Distribute advantages back to processes
        samples_per_gpu = collated_samples["timesteps"].shape[0]

        if stds.ndim == 1:
            stds = stds[:, None]
        if means.ndim == 1:
            means = means[:, None]
        if advantages.ndim == 1:
            advantages = advantages[:, None]

        if stds.shape[0] == world_size * samples_per_gpu and means.shape[0] == world_size * samples_per_gpu:
            collated_samples["stds"] = torch.from_numpy(
                stds.reshape(world_size, samples_per_gpu, -1)[rank]
            ).to(device)
            collated_samples["means"] = torch.from_numpy(
                means.reshape(world_size, samples_per_gpu, -1)[rank]
            ).to(device)
        else:
            assert False

        if advantages.shape[0] == world_size * samples_per_gpu:
            collated_samples["advantages"] = torch.from_numpy(
                advantages.reshape(world_size, samples_per_gpu, -1)[rank]
            ).to(device)
        else:
            assert False
        
        if is_main_process(rank):
            logger.info(
                f"Advantages mean: {collated_samples['advantages'].abs().mean().item()}"
            )

        del collated_samples["rewards"]
        del collated_samples["prompt_ids"]

        num_batches = (
            config.sample.num_batches_per_epoch
            * config.sample.train_batch_size
            // config.train.batch_size
        )

        filtered_samples = collated_samples
        if config.sample.ban_prompt:
            valid_mask = (collated_samples["stds"] > config.sample.ban_std_thres) & \
             (collated_samples["means"] < config.sample.ban_mean_thres)
        else:
            valid_mask = torch.ones_like(collated_samples["stds"])
        filtered_samples["valid_mask"] = valid_mask

        total_batch_size_filtered, num_timesteps_filtered = filtered_samples[
            "timesteps"
        ].shape

        # TRAINING
        transformer_ddp.train()  # Sets DDP model and its submodules to train mode.

        # Total number of backward passes before an optimizer step
        effective_grad_accum_steps = (
            config.train.gradient_accumulation_steps * num_train_timesteps
        )

        current_accumulated_steps = 0  # Counter for backward passes
        gradient_update_times = 0

        for inner_epoch in range(config.train.num_inner_epochs):

            if total_batch_size_filtered == 0: # If all samples are banned, break
                print("All samples are banned, break")
                break
            
            perm = torch.randperm(total_batch_size_filtered, device=device)
            shuffled_filtered_samples = {
                k: v[perm] for k, v in filtered_samples.items()
            }

            perms_time = torch.stack(
                [
                    torch.randperm(num_timesteps_filtered, device=device)
                    for _ in range(total_batch_size_filtered)
                ]
            )
            for key in ["timesteps", "next_timesteps"]:
                shuffled_filtered_samples[key] = shuffled_filtered_samples[key][
                    torch.arange(total_batch_size_filtered, device=device)[:, None],
                    perms_time,
                ]

            training_batch_size = total_batch_size_filtered // num_batches

            samples_batched_list = []
            for k_batch in range(num_batches):
                batch_dict = {}
                start = k_batch * training_batch_size
                end = (k_batch + 1) * training_batch_size
                for key, val_tensor in shuffled_filtered_samples.items():
                    batch_dict[key] = val_tensor[start:end]
                samples_batched_list.append(batch_dict)

            info_accumulated = defaultdict(
                list
            )  # For accumulating stats over one grad acc cycle

            for i, train_sample_batch in tqdm(
                list(enumerate(samples_batched_list)),
                desc=f"Epoch {epoch}.{inner_epoch}: training",
                position=0,
                disable=not is_main_process(rank),
            ):
                current_micro_batch_size = len(train_sample_batch["prompt_embeds"])
                embeds = train_sample_batch["prompt_embeds"]
                pooled_embeds = train_sample_batch["pooled_prompt_embeds"]
                
                # Loop over timesteps for this micro-batch
                for j_idx, j_timestep_orig_idx in tqdm(
                    enumerate(range(num_train_timesteps)),
                    desc="Timestep",
                    position=1,
                    leave=False,
                    disable=not is_main_process(rank),
                ):
                    assert j_idx == j_timestep_orig_idx
                    x0 = train_sample_batch["latents_clean"]

                    t = train_sample_batch["timesteps"][:, j_idx] / 1000.0

                    t_expanded = t.view(-1, *([1] * (len(x0.shape) - 1)))

                    noise = torch.randn_like(x0.float())

                    xt = (1 - t_expanded) * x0 + t_expanded * noise

                    xt_input = torch.cat(
                        [xt, train_sample_batch["image_latents"]], dim=1
                    )
                    guidance = torch.tensor(
                        [config.sample.guidance_scale], device=device
                    )
                    guidance = guidance.expand(xt_input.shape[0])

                    with torch_autocast(
                        enabled=enable_amp, dtype=mixed_precision_dtype
                    ):
                        transformer_ddp.module.set_adapter("old")
                        with torch.no_grad():
                            # prediction v
                            old_prediction = transformer_ddp(
                                hidden_states=xt_input,
                                guidance=guidance,
                                timestep=t,
                                encoder_hidden_states=embeds,
                                txt_ids=train_sample_batch["text_ids"][0],
                                img_ids=train_sample_batch["latent_ids"][0],
                                pooled_projections=pooled_embeds,
                                return_dict=False,
                            )[0].detach()[:, : xt.shape[1]]
                        transformer_ddp.module.set_adapter("default")

                        # prediction v
                        forward_prediction = transformer_ddp(
                            hidden_states=xt_input,
                            guidance=guidance,
                            timestep=t,
                            encoder_hidden_states=embeds,
                            txt_ids=train_sample_batch["text_ids"][0],
                            img_ids=train_sample_batch["latent_ids"][0],
                            pooled_projections=pooled_embeds,
                            return_dict=False,
                        )[0][:, : xt.shape[1]]

                        with torch.no_grad():  # Reference model part
                            # For LoRA, disable adapter.
                            if config.use_lora:
                                with transformer_ddp.module.disable_adapter():
                                    ref_forward_prediction = transformer_ddp(
                                        hidden_states=xt_input,
                                        guidance=guidance,
                                        timestep=t,
                                        encoder_hidden_states=embeds,
                                        pooled_projections=pooled_embeds,
                                        txt_ids=train_sample_batch["text_ids"][0],
                                        img_ids=train_sample_batch["latent_ids"][0],
                                        return_dict=False,
                                    )[0][:, : xt.shape[1]]
                                transformer_ddp.module.set_adapter("default")
                            else:  # Full model - this requires a frozen copy of the model
                                assert False
                    loss_terms = {}

                    valid_mask = train_sample_batch["valid_mask"][:, j_idx].float()
                    # Policy Gradient Loss
                    advantages_clip = torch.clamp(
                        train_sample_batch["advantages"][:, j_idx],
                        -config.train.adv_clip_max,
                        config.train.adv_clip_max,
                    )
                    if hasattr(config.train, "adv_mode"):
                        if config.train.adv_mode == "positive_only":
                            advantages_clip = torch.clamp(
                                advantages_clip, 0, config.train.adv_clip_max
                            )
                        elif config.train.adv_mode == "negative_only":
                            advantages_clip = torch.clamp(
                                advantages_clip, -config.train.adv_clip_max, 0
                            )
                        elif config.train.adv_mode == "one_only":
                            advantages_clip = torch.where(
                                advantages_clip > 0,
                                torch.ones_like(advantages_clip),
                                torch.zeros_like(advantages_clip),
                            )
                        elif config.train.adv_mode == "binary":
                            advantages_clip = torch.sign(advantages_clip)

                    # normalize advantage
                    normalized_advantages_clip = (
                        advantages_clip / config.train.adv_clip_max
                    ) / 2.0 + 0.5
                    r = torch.clamp(normalized_advantages_clip, 0, 1)
                    loss_terms["x0_norm"] = torch.mean(x0**2).detach()
                    loss_terms["x0_norm_max"] = torch.max(x0**2).detach()
                    loss_terms["old_deviate"] = torch.mean(
                        (forward_prediction - old_prediction) ** 2
                    ).detach()
                    loss_terms["old_deviate_max"] = torch.max(
                        (forward_prediction - old_prediction) ** 2
                    ).detach()

                    positive_prediction = (
                        config.beta * forward_prediction
                        + (1 - config.beta) * old_prediction.detach()
                    )
                    implicit_negative_prediction = (
                        1.0 + config.beta
                    ) * old_prediction.detach() - config.beta * forward_prediction

                    # adaptive weighting
                    x0_prediction = xt - t_expanded * positive_prediction
                    with torch.no_grad():
                        weight_factor = (
                            torch.abs(x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    # if is_main_process(rank):
                    #     print("x0_prediction", x0_prediction, x0_prediction.shape)
                    positive_loss = ((x0_prediction - x0) ** 2 / weight_factor).mean(
                        dim=tuple(range(1, x0.ndim))
                    )
                    negative_x0_prediction = (
                        xt - t_expanded * implicit_negative_prediction
                    )
                    with torch.no_grad():
                        negative_weight_factor = (
                            torch.abs(negative_x0_prediction.double() - x0.double())
                            .mean(dim=tuple(range(1, x0.ndim)), keepdim=True)
                            .clip(min=0.00001)
                        )
                    negative_loss = (
                        (negative_x0_prediction - x0) ** 2 / negative_weight_factor
                    ).mean(dim=tuple(range(1, x0.ndim)))
                    ori_policy_loss = (
                        r * positive_loss * valid_mask / config.beta
                        + (1.0 - r) * negative_loss * valid_mask / config.beta
                    )

                    def mean_by_mask(x, mask):
                        if mask.sum() == 0:
                            return x.sum() * 0
                        return x.sum() / mask.sum()

                    policy_loss = mean_by_mask(ori_policy_loss * config.train.adv_clip_max, valid_mask)
                    loss = policy_loss
                    loss_terms["policy_loss"] = policy_loss.detach()
                    loss_terms["unweighted_policy_loss"] = (
                        mean_by_mask(ori_policy_loss, valid_mask).detach()
                    )

                    kl_div_loss = (
                        (forward_prediction - ref_forward_prediction) ** 2
                    ).mean(dim=tuple(range(1, x0.ndim))) * valid_mask

                    loss += config.train.beta * mean_by_mask(kl_div_loss, valid_mask)
                    kl_div_loss = mean_by_mask(kl_div_loss, valid_mask)
                    loss_terms["kl_div_loss"] = kl_div_loss.detach()
                    loss_terms["kl_div"] = mean_by_mask(
                        ((forward_prediction - ref_forward_prediction) ** 2).mean(
                            dim=tuple(range(1, x0.ndim))
                        ) * valid_mask, valid_mask
                    ).detach()
                    loss_terms["old_kl_div"] = mean_by_mask(
                        ((old_prediction - ref_forward_prediction) ** 2).mean(
                            dim=tuple(range(1, x0.ndim))
                        ) * valid_mask, valid_mask
                    ).detach()
                    # if is_main_process(rank):
                    #     print("valid_mask", valid_mask.shape, valid_mask, positive_loss, positive_loss.shape)
                    # ori_policy_loss = (
                    #     r * positive_loss / config.beta
                    #     + (1.0 - r) * negative_loss / config.beta
                    # )
                    # policy_loss = (ori_policy_loss * config.train.adv_clip_max).mean()

                    # loss = policy_loss
                    # loss_terms["policy_loss"] = policy_loss.detach()
                    # loss_terms["unweighted_policy_loss"] = (
                    #     ori_policy_loss.mean().detach()
                    # )

                    # kl_div_loss = (
                    #     (forward_prediction - ref_forward_prediction) ** 2
                    # ).mean(dim=tuple(range(1, x0.ndim)))

                    # loss += config.train.beta * torch.mean(kl_div_loss)
                    # kl_div_loss = torch.mean(kl_div_loss)
                    # loss_terms["kl_div_loss"] = torch.mean(kl_div_loss).detach()
                    # loss_terms["kl_div"] = torch.mean(
                    #     ((forward_prediction - ref_forward_prediction) ** 2).mean(
                    #         dim=tuple(range(1, x0.ndim))
                    #     )
                    # ).detach()
                    # loss_terms["old_kl_div"] = torch.mean(
                    #     ((old_prediction - ref_forward_prediction) ** 2).mean(
                    #         dim=tuple(range(1, x0.ndim))
                    #     )
                    # ).detach()

                    loss_terms["total_loss"] = loss.detach()

                    # Scale loss for gradient accumulation and DDP (DDP averages grads, so no need to divide by world_size here)
                    scaled_loss = loss / effective_grad_accum_steps
                    if mixed_precision_dtype == torch.float16:
                        scaler.scale(scaled_loss).backward()  # one accumulation
                    else:
                        scaled_loss.backward()
                    current_accumulated_steps += 1

                    for k_info, v_info in loss_terms.items():
                        info_accumulated[k_info].append(v_info)

                    if current_accumulated_steps % effective_grad_accum_steps == 0:
                        if mixed_precision_dtype == torch.float16:
                            scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            transformer_ddp.module.parameters(),
                            config.train.max_grad_norm,
                        )
                        if mixed_precision_dtype == torch.float16:
                            scaler.step(optimizer)
                        else:
                            optimizer.step()
                        gradient_update_times += 1
                        if mixed_precision_dtype == torch.float16:
                            scaler.update()
                        optimizer.zero_grad()

                        log_info = {
                            k: torch.mean(torch.stack(v_list)).item()
                            for k, v_list in info_accumulated.items()
                        }
                        info_tensor = torch.tensor(
                            [log_info[k] for k in sorted(log_info.keys())],
                            device=device,
                        )
                        dist.all_reduce(info_tensor, op=dist.ReduceOp.AVG)
                        reduced_log_info = {
                            k: info_tensor[ki].item()
                            for ki, k in enumerate(sorted(log_info.keys()))
                        }
                        if is_main_process(rank):
                            wandb.log(
                                {
                                    "step": global_step,
                                    "gradient_update_times": gradient_update_times,
                                    "epoch": epoch,
                                    "inner_epoch": inner_epoch,
                                    **reduced_log_info,
                                }
                            )

                        global_step += 1  # gradient step
                        info_accumulated = defaultdict(
                            list
                        )  # Reset for next accumulation cycle

                if (
                    config.train.ema
                    and ema is not None
                    and (current_accumulated_steps % effective_grad_accum_steps == 0)
                ):
                    ema.step(transformer_trainable_parameters, global_step)

        if world_size > 1:
            dist.barrier()

        with torch.no_grad():
            decay = return_decay(global_step, config.decay_type)
            for src_param, tgt_param in zip(
                transformer_trainable_parameters,
                old_transformer_trainable_parameters,
                strict=True,
            ):
                tgt_param.data.copy_(
                    tgt_param.detach().data * decay
                    + src_param.detach().clone().data * (1.0 - decay)
                )

    if is_main_process(rank):
        wandb.finish()
    cleanup_distributed()


if __name__ == "__main__":
    app.run(main)
