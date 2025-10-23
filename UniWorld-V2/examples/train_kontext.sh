export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TOKENIZERS_PARALLELISM=true
export REWARD_SERVER=10.0.67.19:12341

export NCCL_IB_TC=136
export NCCL_IB_SL=5
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=eth
export NCCL_IB_HCA=mlx5
export NCCL_IB_TIMEOUT=22
export NCCL_IB_QPS_PER_CONNECTION=8
export NCCL_NET_PLUGIN=none

torchrun --nproc_per_node=8 \
    --nnodes=${WORLD_SIZE} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    --node_rank ${RANK} \
    scripts/train_nft_kontext.py --config config/kontext_nft.py:$1