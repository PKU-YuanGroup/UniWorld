
The original code is from [ImgEdit](https://huggingface.co/datasets/sysuyy/ImgEdit).

## Requirements and Installation
The benchmark images can be downloaded from huggingface [Benchmark.tar](https://huggingface.co/datasets/sysuyy/ImgEdit/blob/main/Benchmark.tar)

Install the required dependencies using `pip`:

```
pip install tqdm tenacity 
pip install -U openai
```



## Eval

### Generate samples

```bash
# switch to univa env
MODEL_PATH='path/to/model'
OUTPUT_DIR='path/to/eval_output/imgedit'
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 torchrun \
  --nproc_per_node 8 \
  -m step1_gen_samples \
  imgedit.yaml \
  --pretrained_lvlm_name_or_path ${MODEL_PATH} \
  --output_dir ${OUTPUT_DIR}
```

### Evaluation

The benchmark images can be downloaded from huggingface [Benchmark.tar](https://huggingface.co/datasets/sysuyy/ImgEdit/blob/main/Benchmark.tar)

```bash
ASSET_ROOT="imgedit_asset"
mkdir -p "$ASSET_ROOT"
wget -O "$ASSET_ROOT/Benchmark.tar" "https://huggingface.co/datasets/sysuyy/ImgEdit/resolve/main/Benchmark.tar"
cd $ASSET_ROOT
tar -xf "Benchmark.tar"
cd ..
```


```bash
# switch to univa env
IMAGE_DIR=${OUTPUT_DIR}
python step2_basic_bench.py \
   --result_img_folder ${IMAGE_DIR} \
   --result_json ${IMAGE_DIR}/imgedit_bench.json \
   --edit_json eval_prompts/basic_edit.json \
   --prompts_json eval_prompts/prompts.json \
   --origin_img_root ${ASSET_ROOT}/Benchmark/singleturn \
   --api_key ${OPENAI_API_KEY} 
```

### Summary  

```bash
python step3_get_avgscore.py \
   --input ${IMAGE_DIR}/imgedit_bench.json \
   --meta_json eval_prompts/basic_edit.json \
   --output_json ${IMAGE_DIR}.json
cat ${IMAGE_DIR}.json
```