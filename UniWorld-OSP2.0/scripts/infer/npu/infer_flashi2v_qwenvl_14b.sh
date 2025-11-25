pkill -9 -f infer.py
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export TOKENIZERS_PARALLELISM=false

export CUDA_DEVICE_MAX_CONNECTIONS=1
export ASCEND_SLOG_PRINT_TO_STDOUT=0
export ASCEND_GLOBAL_LOG_LEVEL=3
export MULTI_STREAM_MEMORY_REUSE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export TASK_QUEUE_ENABLE=1
export COMBINED_ENABLE=1
export CPU_AFFINITY_CONF=1
export HCCL_CONNECT_TIMEOUT=3600
export HCCL_EXEC_TIMEOUT=0
export ACL_DEVICE_SYNC_TIMEOUT=3600


# export CUDA_VISIBLE_DEVICES=0,1,2,3

MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29505}
NPRC_PER_NODE=${NPRC_PER_NODE:-8}
NNODES=${NNODES:-1}
WORLD_SIZE=$(($NNODES * $NPRC_PER_NODE))


cd /work/share/projects/gyy/Uniworld_OSP2/ 
# NPRC_PER_NODE=1
torchrun \
  --nproc_per_node=${NPRC_PER_NODE} \
  --nnodes=${NNODES} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  infer/infer_uniworld_osp2.py \
  --config /work/share/projects/gyy/Uniworld_OSP2/configs/infer/npu/uniworld_osp2_14b.yaml
