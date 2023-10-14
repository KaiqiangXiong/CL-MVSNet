#!/usr/bin/env bash
source /mnt/data/xkq/anaconda3/etc/profile.d/conda.sh

conda activate mvs
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 --master_port=2340 main.py \
        --sync_bn \
