The original code is from [GenEval](https://github.com/djghosh13/geneval).

## Requirements and Installation

### Prepare the environment

> Official environment is **NOT** recommended.

Prepare conda environment:

```bash
conda create -n geneval_eval python=3.10 -y
conda activate geneval_eval
```

Install torch:

```bash
# The CUDA version in the virtual env must be identical to that of the physical env.
conda install pytorch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 pytorch-cuda=12.4 -c pytorch -c nvidia -y
conda install -c nvidia cudatoolkit=12.4 -y
conda install -c conda-forge nvcc_linux-64 -y
pip install -U openmim
```

Install the MMCV:


**H100/H800**
```
git clone https://github.com/open-mmlab/mmcv.git
cd mmcv

git checkout v2.2.0
pip install -r requirements/optional.txt
vim setup.py
# L160
extra_compile_args = {
    # 'nvcc': [cuda_args, '-std=c++14'] if cuda_args else ['-std=c++14'],
    'nvcc': [cuda_args, '-std=c++14', '-arch=sm_90'] if cuda_args else ['-std=c++14'],
    'cxx': ['-std=c++14'],
}
# Revert all changes to setup.py using Ctrl+Z. Then, Ctrl+S to save
pip install -v -e .
git checkout v1.7.0
vim setup.py
# L217
extra_compile_args = {
    # 'nvcc': [cuda_args, '-std=c++14'] if cuda_args else ['-std=c++14'],
    'nvcc': [cuda_args, '-std=c++14', '-arch=sm_90'] if cuda_args else ['-std=c++14'],
    'cxx': ['-std=c++14'],
}
pip install -v -e .
python .dev_scripts/check_installation.py
cd ..
```

**Other GPU**
```bash
mim install mmengine mmcv-full==1.7.2
```

Install the MMDet:

```bash
pip install -r requirements.txt
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection; git checkout 2.x
pip install -v -e .
```

## Eval

### Generate samples

We also support LLM rewrite prompt, just replace `geneval.yaml` with `geneval_long.yaml` (sourced from [BAGEL](https://github.com/ByteDance-Seed/Bagel/blob/main/eval/gen/geneval/prompts/evaluation_metadata_long.jsonl)) and change `$OUTPUT_DIR`.

```bash
# switch to univa env
MODEL_PATH='path/to/model'
OUTPUT_DIR='path/to/eval_output/geneval'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node 8 \
  -m step1_gen_samples \
  geneval.yaml \
  --pretrained_lvlm_name_or_path ${MODEL_PATH} \
  --output_dir ${OUTPUT_DIR}
```

### Evaluation


Download the Mask2Former object detection config and weights:

```bash
DETECTOR_PATH="/path/to/detector"
mkdir -p ${DETECTOR_PATH}
wget https://download.openmmlab.com/mmdetection/v2.0/mask2former/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco_20220504_001756-743b7d99.pth -O "${DETECTOR_PATH}/mask2former_swin-s-p4-w7-224_lsj_8x2_50e_coco.pth"

CACHE_DIR="/path/to/cache_dir"
mkdir -p ${CACHE_DIR}
wget https://openaipublic.azureedge.net/clip/models/b8cca3fd41ae0c99ba7e8951adf17d267cdb84cd88be6f7c2e0eca1737a03836/ViT-L-14.pt -O "${CACHE_DIR}/ViT-L-14.pt"
```

```bash
conda activate geneval_eval
IMAGE_DIR=${OUTPUT_DIR}
CUDA_VISIBLE_DEVICES=0 CACHE_DIR=${CACHE_DIR} python step2_run_geneval.py \
    ${IMAGE_DIR} \
    --outfile ${IMAGE_DIR}.jsonl \
    --model-path ${DETECTOR_PATH}
```

### Summary  

```bash
python step3_summary_score.py \
    ${IMAGE_DIR}.jsonl > ${IMAGE_DIR}.txt
cat ${IMAGE_DIR}.txt
```
