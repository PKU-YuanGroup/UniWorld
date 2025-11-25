pkill -9 -f infer.py

export TOKENIZERS_PARALLELISM=false

export CUDA_VISIBLE_DEVICES=4,5,6,7

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29505}
NPRC_PER_NODE=${NPRC_PER_NODE:-4}
NNODES=${NNODES:-1}
WORLD_SIZE=$(($NNODES * $NPRC_PER_NODE))

torchrun \
  --nproc_per_node=${NPRC_PER_NODE} \
  --nnodes=${NNODES} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  infer/infer_flashi2v.py \
  --config configs/infer/gpu/flashi2v_14b.yaml
