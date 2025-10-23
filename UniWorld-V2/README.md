<p align="center">
    <img src="https://s21.ax1x.com/2025/06/03/pVCBdw8.png" width="200"/>
<p>
<h2 align="center"> 
  <a href="https://arxiv.org/abs/2510.16888">
    UniWorld-V2: Reinforce Image Editing with Diffusion Negative-Aware Finetuning and
MLLM Implicit Feedback
  </a>
</h2>

[![UniWorld-V2](https://img.shields.io/badge/Arxiv-UniWorldV2-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2510.16888)
[![UniWorld-V1](https://img.shields.io/badge/Arxiv-UniWorldV1-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.03147)
[![ImgEdit](https://img.shields.io/badge/Arxiv-ImgEdit-b31b1b.svg?logo=arXiv)](https://arxiv.org/abs/2506.03147)
[![Collection](https://img.shields.io/badge/🤗-Collection-blue.svg)](https://huggingface.co/collections/chestnutlzj/edit-r1-68dc3ecce74f5d37314d59f4)
[![License](https://img.shields.io/badge/License-Apache-yellow)](https://github.com/PKU-YuanGroup/UniWorld-V2/blob/main/LICENSE)

## 📣 News

**[2025/10/19]**: We release **Edit-R1**, which employs [DiffusionNFT](https://github.com/NVlabs/DiffusionNFT) and a training-free reward
model derived from pretrained MLLMs to fine-tune diffusion models for image editing. [UniWorld-Qwen-Image-Edit-2509](https://huggingface.co/collections/chestnutlzj/edit-r1-68dc3ecce74f5d37314d59f4) and [UniWorld-FLUX.1-Kontext-Dev](https://huggingface.co/collections/chestnutlzj/edit-r1-68dc3ecce74f5d37314d59f4) are open-sourced.

## 🎨 Case Comparisons

| Original | Prompt | Nano-banana | GPT-4o | Qwen-Image-Edit | **UniWorld-V2 (Ours)** |
| :---: | :---: | :---: | :---: | :---: | :---: |
| <img src="imgs/0-0.jpg" width="400"> | **Case 1:** `把鸟移动到红框里，删除掉现在的鸟，最后移除红框` | <img src="imgs/0-1.webp" width="400"> | <img src="imgs/0-2.webp" width="400"> | <img src="imgs/0-3.webp" width="400"> | <img src="imgs/0-4.webp" width="400"> （✅正确执行指令）|
| <img src="imgs/1-0.jpg" width="400"> | **Case 2:** `把中间白色衣服戴口罩女生的手势改成OK` | <img src="imgs/1-1.webp" width="400"> | <img src="imgs/1-2.webp" width="400"> | <img src="imgs/1-3.webp" width="400"> | <img src="imgs/1-4.webp" width="400">  （✅OK手势 ）|
| <img src="imgs/2-0.jpg" width="400"> | **Case 3:** `提取画面中的吉他` | <img src="imgs/2-1.webp" width="400"> | <img src="imgs/2-2.webp" width="400"> | <img src="imgs/2-3.webp" width="400"> | <img src="imgs/2-4.webp" width="400">（✅弦钮上二下三 ） |
| <img src="imgs/3-0.png" width="400"> | **Case 4:** `把下面的所有文字并改用书法体。中间的“月满中秋”改成“千里团圆”。并且把月亮改成模糊的月饼。` | <img src="imgs/3-1.webp" width="400"> | <img src="imgs/3-2.webp" width="400"> | <img src="imgs/3-3.webp" width="400"> | <img src="imgs/3-4.webp" width="400"> （✅模糊月饼，✅书法字体）|
| <img src="imgs/4-0.jpg" width="400"> | **Case 5:** `让画面中的形象坐在高档西餐厅，双手拿刀叉吃牛排` | <img src="imgs/4-1.webp" width="400"> | <img src="imgs/4-2.webp" width="400"> | <img src="imgs/4-3.webp" width="400"> | <img src="imgs/4-4.webp" width="400"> （✅人物特征，✅刀叉）|

## 🗝️ Train

### Deploy vLLM Reward Server

Start the reward server:

```
python reward_server/reward_server.py
```

If you want to check the status of the reward server, you can test it by running:

```
python reward_server/test_reward_server.py
```

### Data Format

Directory structure:

```
- dataset-dir
  - images/
     - YOUR_IMAGE_DATA
     - ...
  - train_metadata.jsonl
  - test_metadata.jsonl
```

`train_metadata.jsonl` and `test_metadata.jsonl` format:

```
{"prompt": "PROMPT", "image": "IMAGE_RELATIVE_PATH", "requirement": "TASK_REQUIREMENT"}
...
```

### Configure Training

See `config/qwen_image_edit_nft.py` and `config/kontext_nft.py` for available configurations.

### Run Training

```shell
export REWARD_SERVER=[YOUR_REWARD_SERVICE_IP_ADDR]:12341

torchrun --nproc_per_node=8 \
    scripts/train_nft_qwen_image_edit.py --config config/qwen_image_edit_nft.py:config_name
```

And you can also refer to the example scripts in `examples/`.

## ⚡️ Reproduction

For reproducibility, we provide the reproduction scripts in `reproduction/`.

See [Reproduction Details](reproduction/README.md) for more details.

## 👍 Acknowledgement

- [**DiffusionNFT**](https://github.com/NVlabs/DiffusionNFT): Huge thanks for their elegant codebase 🤩!
- [Flow-GRPO](https://github.com/yifan123/flow_grpo)
- [ImgEdit](https://github.com/PKU-YuanGroup/ImgEdit)
- [UniWorld-V1](https://github.com/PKU-YuanGroup/UniWorld-V1)

## 🔒 License

See [LICENSE](LICENSE) for details. The FLUX weights fall under the [FLUX.1 [dev] Non-Commercial License](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md).

## ✏️ Citation

```
@article{li2025uniworldv2,
    title={Uniworld-V2: Reinforce Image Editing with Diffusion Negative-aware Finetuning and MLLM Implicit Feedback},
    author={Li, Zongjian and Liu, Zheyuan and Zhang, Qihui and Lin, Bin and Yuan, Shenghai and Yan, Zhiyuan and Ye, Yang and Yu, Wangbo and Niu, Yuwei and Yuan, Li},
    journal={arXiv preprint arXiv:2506.03147},
    year={2025}
}

@article{lin2025uniworld,
  title={Uniworld: High-resolution semantic encoders for unified visual understanding and generation},
  author={Lin, Bin and Li, Zongjian and Cheng, Xinhua and Niu, Yuwei and Ye, Yang and He, Xianyi and Yuan, Shenghai and Yu, Wangbo and Wang, Shaodong and Ge, Yunyang and others},
  journal={arXiv preprint arXiv:2506.03147},
  year={2025}
}

@article{ye2025imgedit,
  title={Imgedit: A unified image editing dataset and benchmark},
  author={Ye, Yang and He, Xianyi and Li, Zongjian and Lin, Bin and Yuan, Shenghai and Yan, Zhiyuan and Hou, Bohan and Yuan, Li},
  journal={arXiv preprint arXiv:2505.20275},
  year={2025}
}
```
