
The original code is from [DPG-Bench](https://github.com/TencentQQGYLab/ELLA).


## Requirements and Installation

> Official environment is **NOT** recommended.

Prepare conda environment:

```bash
conda create -n dpgbench_eval python=3.10 -y
conda activate geneval_eval
```

Install package:

```bash
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
pip install "pip<24.1"
pip install -r requirements.txt
```

## Eval

### Generate samples

```bash
# switch to univa env
MODEL_PATH='path/to/model'
OUTPUT_DIR='path/to/eval_output/dpgbench'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node 8 \
  -m step1_gen_samples \
  dpgbench.yaml \
  --pretrained_lvlm_name_or_path ${MODEL_PATH} \
  --output_dir ${OUTPUT_DIR}
```

### Evaluation & Summary


Download mplug model to `$MPLUG_LOCAL_PATH`:

```bash
conda activate dpgbench_eval
modelscope download --model 'iic/mplug_visual-question-answering_coco_large_en' --local_dir ${MPLUG_LOCAL_PATH}
```

```bash
conda activate dpgbench_eval
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
IMAGE_DIR=${OUTPUT_DIR}
accelerate launch --num_machines 1 --num_processes 8 \
    --multi_gpu --mixed_precision "fp16" \
    step2_compute_dpg_bench.py \
    --image_root_path ${IMAGE_DIR} \
    --resolution 1024 \
    --pic_num 4 \
    --res_path ${IMAGE_DIR}.txt \
    --vqa_model mplug \
    --mplug_local_path ${MPLUG_LOCAL_PATH} \
    --csv eval_prompts/dpgbench.csv
cat ${IMAGE_DIR}.txt
```
