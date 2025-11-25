export WANDB_MODE="offline"

export TOKENIZERS_PARALLELISM=false

export CUDA_VISIBLE_DEVICES=4,5,6,7

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29501}
NPRC_PER_NODE=${NPRC_PER_NODE:-4}
NNODES=${PET_NNODES:-1}
NODE_RANK=${RANK:-0}
WORLD_SIZE=$(($NNODES * $NPRC_PER_NODE))

torchrun \
  --nproc_per_node=${NPRC_PER_NODE} \
  --nnodes=${NNODES} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  train/train_uniworld_osp2.py \
  --config   configs/train/npu/uniworld_osp2_14b.yaml