
The original code is from [GEdit-Bench](https://github.com/stepfun-ai/Step1X-Edit/blob/main/GEdit-Bench/EVAL.md).

## Requirements and Installation

```
pip install megfile openai 
```

## Prepare Source Images
Prepare the original image and metadata json following the example code in `step0_generate_image_example.py`

```bash
GEDIT_ASSET="/path/to/gedit_asset"
python step0_prepare_gedit.py --save_path ${GEDIT_ASSET} --json_file_path gedit_edit.json
```

The file directory structure of the original image：
```folder
${GEDIT_ASSET}/
│   └── fullset/
│       └── edit_task/
│           ├── cn/  # Chinese instructions
│           │   ├── key1.png
│           │   ├── key2.png
│           │   └── ...
│           └── en/  # English instructions
│               ├── key1.png
│               ├── key2.png
│               └── ...
```

## Eval


### Generate samples

```bash
# switch to univa env
MODEL_PATH='path/to/model'
OUTPUT_DIR='path/to/eval_output/gedit'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node 8 \
  -m step1_gen_samples \
  gedit.yaml \
  --pretrained_lvlm_name_or_path ${MODEL_PATH} \
  --output_dir ${OUTPUT_DIR}
```

### Evaluation

Write your gpt-api-key to `secret_t2.env`.

```bash
IMAGE_DIR=${OUTPUT_DIR}
python step2_gedit_bench.py \
    --model_name UniWorld \
    --save_path ${IMAGE_DIR} \
    --backbone gpt4o \
    --source_path ${GEDIT_ASSET}
```

### Summary
```bash
python step3_calculate_statistics.py \
    --model_name UniWorld \
    --save_path ${IMAGE_DIR} \
    --backbone gpt4o \
    --language en > ${IMAGE_DIR}.txt
cat ${IMAGE_DIR}.txt
```
