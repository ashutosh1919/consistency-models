#!/bin/sh

cd /notebooks/consistency_models &&
mpiexec --allow-run-as-root -n 1 python -m scripts.edm_train \
    --attention_resolutions 32,16,8 \
    --class_cond False \
    --use_scale_shift_norm False \
    --dropout 0.1 \
    --ema_rate 0.999,0.9999,0.9999432189950708 \
    --global_batch_size 256 \
    --image_size 256 \
    --lr 0.0001 \
    --num_channels 256 \
    --num_head_channels 64 \
    --num_res_blocks 2 \
    --resblock_updown True \
    --schedule_sampler lognormal \
    --use_fp16 True \
    --weight_decay 0.0 \
    --weight_schedule karras \
    --data_dir /notebooks/datasets/lsun_bedroom/train/processed/
