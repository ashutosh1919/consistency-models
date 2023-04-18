#!/bin/sh

cd /notebooks/consistency_models &&
mpiexec --allow-run-as-root -n 1 python -m scripts.cm_train \
    --training_mode consistency_training \
    --target_ema_mode adaptive \
    --start_ema 0.95 \
    --scale_mode progressive \
    --start_scales 2 \
    --end_scales 200 \
    --total_training_steps 800000 \
    --loss_norm lpips \
    --lr_anneal_steps 0 \
    --teacher_model_path /notebooks/checkpoints/edm_imagenet64_ema.pt \
    --attention_resolutions 32,16,8 \
    --class_cond True \
    --use_scale_shift_norm True \
    --dropout 0.0 \
    --teacher_dropout 0.1 \
    --ema_rate 0.999,0.9999,0.9999432189950708 \
    --global_batch_size 2048 \
    --image_size 64 \
    --lr 0.0001 \
    --num_channels 192 \
    --num_head_channels 64 \
    --num_res_blocks 3 \
    --resblock_updown True \
    --schedule_sampler uniform \
    --use_fp16 True \
    --weight_decay 0.0 \
    --weight_schedule uniform \
    --data_dir /notebooks/datasets/imagenet/
