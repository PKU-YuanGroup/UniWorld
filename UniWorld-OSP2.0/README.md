---
license: mit
---

<h1 align="left"> <a href="">Uniworld-OSP2.0</a></h1>

Uniworld-OSP2.0 is a next-generation **unified conditional video generation framework**, built upon the Fourier-Guided Latent Shifting I2V paradigm (FlashI2V). Compared with conventional I2V/T2V systems, Uniworld-OSP2.0 achieves **lossless semantic inheritance, motion stability, style controllability and superior aesthetic performance**, surpassing Wan2.1 on 6 key evaluation metrics.

本项目发布于 **2025 年 11 月**，由 **北京大学-兔展 AIGC 联合实验室** 提出。模型支持 GPU与**华为昇腾 910 全流程训练及推理**，并已实现 **端到端 NPU 训练+部署闭环**

Pull requests are welcome! ⭐ Star the repo to stay updated!



# 📣 News
* 🧩 **Model weights & stylized dataset release (Coming soon)** — 12 artistic styles & metadata will be open-sourced.
* 📘 **[2025-11-25]** Uniworld-OSP2.0 technical report has been released!.[Click to Download](./docs/Uniworld-OSP2.0.pdf)
* 🔥 **[2025-11-25]** We officially release **Uniworld-OSP2.0 (21B)** — featuring VLM-enhanced conditioning, unified Image-to-Stylized-Video (I2SV), and breakthrough performance beyond Wan2.1.


# 😮 Highlights

### ⚡ FlashI2V Stabilized Motion
- Prevents **conditional image leakage**
- Maintains **sharp structure + continuous motion**

### 🧠 VLM-Enhanced Semantic Conditioning
- Integrates a **7B Visual Language Model**
- Replaces shallow text encoders → **lossless semantic inheritance**

### 🎨 Unified I2SV (Image-to-Stylized-Video)
- Converts an input image into a video rendered in **any of 12 artistic styles**
- **Style fidelity without harming subject or dynamics**

### 📈 Scaling Up
- Model scaled to **21B parameters**
- Outperforms **Wan2.1** on 6 key I2V metrics

---

# 📊 Benchmark Results (I2V)


| **Model**       | **I2V Paradigm**               | **Subject Consistency ↑** | **Background Consistency ↑** | **Motion Smoothness ↑** | **Dynamic Degree ↑** | **Aesthetic Quality ↑** | **Imaging Quality ↑** | **I2V Subject Consistency ↑** | **I2V Background Consistency ↑** |
| ----------------------- | -------------------------------------- | ---------------------------------- | ------------------------------------- | -------------------------------- | ----------------------------- | -------------------------------- | ------------------------------ | -------------------------------------- | ----------------------------------------- |
| SVD-XT-1.0 (1.5B)     | Repeating Concat and Adding Noise    | 95.52                            | 96.61                               | 98.09                          | 52.36                       | 60.15                          | 69.80                        | 97.52                                | 97.63                                   |
| SVD-XT-1.1 (1.5B)     | Repeating Concat and Adding Noise    | 95.42                            | 96.77                               | 98.12                          | 43.17                       | 60.23                          | 70.23                        | 97.51                                | 97.62                                   |
| SEINE-512x512 (1.8B)  | Inpainting                           | 95.28                            | 97.12                               | 97.12                          | 27.07                       | 64.55                          | 71.39                        | 97.15                                | 96.94                                   |
| CogVideoX-5B-I2V      | Zero-padding Concat and Adding Noise | 94.34                            | 96.42                               | 98.40                          | 33.17                       | 61.87                          | 70.01                        | 97.19                                | 96.74                                   |
| Wan2.1-I2V-14B-720P   | Inpainting                           | 94.86                            | 97.07                               | 97.90                          | 51.38                       | 64.75                          | 70.44                        | 96.95                                | 96.44                                   |
| CogVideoX1.5-5B-I2V   | Zero-padding Concat and Adding Noise | 95.04                            | 96.52                               | 98.47                          | 37.48                       | 62.68                          | 70.99                        | 97.78                                | 98.73                                   |
| Wan2.1-I2V-14B-480P   | Inpainting                           | 95.68                            | 97.44                               | 98.46                          | 45.20                       | 61.44                          | 70.37                        | 97.83                                | 99.08                                   |
| **Uniworld-OSP2.0** | FlashI2V                             | **96.21**                            | **97.71**                               | **98.47**                          | **46.10**                      | **66.55**                          | 70.57                        | **97.99**                                | 98.94                                   |


# ⚙️ Requirements & Installation

```bash
conda create -n uniworld_osp2 python=3.10 -y
conda activate uniworld_osp2
pip install -r  Uniworld_OSP2/requirements_npu.txt
```
# Training

Make sure to properly configure configs/train/npu/uniworld_osp2_14b.yaml.

**uniworld_osp2_14b.yaml:**
```yaml
model_name: "flashi2v"
seed: 1024
output_dir: "XXXXX"
# /work/share/projects/gyy/zhubin_FlashI2V/configs
training_iteration: 1000000
# ddp_size: 1
fsdp_size: 8
cp_size: 2
use_context_parallel: True
reshard_after_forward: null
gradient_checkpointing: True
gradient_accumulation_steps: 1
init_max_grad_norm: 1.0
log_interval: 1
save_interval: 200
weight_dtype: "bf16"
ema_decay: 0.9999
ema_update_interval: 1
save_with_dcp_api: True

wandb_config:
  project_name: "uniworld_osp2"
  exp_name: "uniworld_osp2"

model_config:
  dim: 5120
  ffn_dim: 13824
  freq_dim: 256
  in_dim: 16
  num_heads: 40
  num_layers: 40
  out_dim: 16
  text_len": 512
  low_freq_energy_ratio: [0.05, 0.95]
  fft_return_abs: True
  conv3x3x3_proj: False
  pretrained_model_dir_or_checkpoint: "XXXXX"

scheduler_config:
  scheduler_name: "flashi2v_flow_matching"
  use_dynamic_shifting: True
  use_logitnorm_time_sampling: True

vae_config:
  vae_path: "XXXXX/Wan-AI/Wan2.1-T2V-14B/Wan2.1_VAE.pth"
  dtype: "fp32"

text_encoder_config:
  text_len: 512
  checkpoint_path: "XXXXXX/Wan-AI/Wan2.1-T2V-14B/models_t5_umt5-xxl-enc-bf16.pth"
  use_fsdp: True

vlm_encoder_config:
  checkpoint_path: "XXXXXX/Qwen2.5-VL-7B-Instruct"

data_config:
  batch_size: 1
  num_workers: 16
  pin_memory: False
  drop_last: True
  shuffle: True
  dataset_name: "flashi2v"
  dataset_config:
    metafile_or_dir_path: "/work/share1/caption/osp/lmdb/all_videos_81f_864p/filtered_samples_2088532.lmdb"
    text_tokenizer_path: "XXXXX/Wan-AI/Wan2.1-T2V-14B/google/umt5-xxl"
    text_drop_ratio: 0.1
    sample_height: 480
    sample_width: 832
    sample_num_frames: 49
    train_fps: 16
    tokenizer_max_length: 512
    return_prompt_mask: True
  sampler_name: "stateful_distributed"
  collator_name: "flashi2v"
  
optimizer_config:
  lr: 0.000005
  weight_decay: 0.0

```


# Training
```bash
cd  Uniworld_OSP2
bash scripts/train/npu/train_uniworld_osp2_14b.sh
```
# Inference
```bash
cd  Uniworld_OSP2
bash infer/infer_uniworld_osp2.py
```
