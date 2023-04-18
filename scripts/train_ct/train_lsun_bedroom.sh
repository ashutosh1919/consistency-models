#!/bin/sh

cd /notebooks/consistency_models &&
mpiexec --allow-run-as-root -n 1 python -m scripts.cm_train \
    --training_mode consistency_training \
    --target_ema_mode adaptive \
    --start_ema 0.95 \
    --scale_mode progressive \
    --start_scales 2 \
    --end_scales 150 \
    --total_training_steps 1000000 \
    --loss_norm lpips \
    --lr_anneal_steps 0 \
    --teacher_model_path /notebooks/checkpoints/edm_bedroom256_ema.pt \
    --attention_resolutions 32,16,8 \
    --class_cond False \
    --use_scale_shift_norm False \
    --dropout 0.0 \
    --teacher_dropout 0.1 \
    --ema_rate 0.9999,0.99994,0.9999432189950708 \
    --global_batch_size 256 \
    --image_size 256 \
    --lr 0.00005 \
    --num_channels 256 \
    --num_head_channels 64 \
    --num_res_blocks 2 \
    --resblock_updown True \
    --schedule_sampler uniform \
    --use_fp16 True \
    --weight_decay 0.0 \
    --weight_schedule uniform \
    --data_dir /notebooks/datasets/lsun_bedroom/train/processed/
