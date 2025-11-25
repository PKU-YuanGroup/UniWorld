pkill -9 -f infer.py

export TOKENIZERS_PARALLELISM=false

# export CUDA_VISIBLE_DEVICES=0,1,2,3

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29505}
NPRC_PER_NODE=${NPRC_PER_NODE:-8}
NNODES=${NNODES:-1}
WORLD_SIZE=$(($NNODES * $NPRC_PER_NODE))

torchrun \
  --nproc_per_node=${NPRC_PER_NODE} \
  --nnodes=${NNODES} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  infer/infer_t2v.py \
  --config configs/infer/gpu/t2v_1_3b.yaml
