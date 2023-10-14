#!/usr/bin/env bash
source /mnt/data/xkq/anaconda3/etc/profile.d/conda.sh

conda activate mvs
CUDA_VISIBLE_DEVICES=0 python main.py \
        --test \
        --dataset_name "general_eval" \
        --datapath /mnt/xkq/Data/MVS/test/dtu/origin \
        --img_size 1184 1600 \
        --resume pretrained_model/model.ckpt \
        --testlist datasets/lists/dtu/test.txt