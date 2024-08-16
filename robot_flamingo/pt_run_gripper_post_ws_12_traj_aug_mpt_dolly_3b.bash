#!/bin/bash
export PATH=$PATH:path/to/robot-flamingo/robot_flamingo
export PYTHONPATH=$PYTHONPATH:path/to/robot-flamingo/robot_flamingo
# dataset path
calvin_dataset_path='dataset/calvin_debug_dataset'
# language model path
lm_path='mosaicml/mpt-1b-redpajama-200b'
# tokenizer path
tokenizer_path='mosaicml/mpt-1b-redpajama-200b'
# openflamingo ckpt path
openflamingo_checkpoint='/home/edward/.cache/huggingface/hub/models--openflamingo--OpenFlamingo-3B-vitl-mpt1b/snapshots/ed3a0c3190b2fc2d1c39630738896d4e73ce1bbc/checkpoint.pt'

subfix=`date "+%Y%m%d-%H%M"`
log_file="logs/training_"${subfix}".log"
source /home/edward/miniforge3/bin/activate roboflamingo
#python3 -m torch.distributed.launch --nnodes=1 --nproc_per_node=2  --master_port=6042 robot_flamingo/train/train_calvin.py \
torchrun --nnodes=1 --nproc_per_node=1 --master_port=6042 robot_flamingo/train/train_calvin.py \
    --report_to_wandb \
    --llm_name mpt_dolly_3b \
    --traj_cons \
    --use_gripper \
    --fusion_mode post \
    --rgb_pad 10 \
    --gripper_pad 4 \
    --precision fp32 \
    --num_epochs 5 \
    --gradient_accumulation_steps 1 \
    --batch_size_calvin 6 \
    --run_name RobotFlamingoDBG \
    --calvin_dataset ${calvin_dataset_path} \
    --lm_path ${lm_path} \
    --tokenizer_path ${tokenizer_path} \
    --openflamingo_checkpoint ${openflamingo_checkpoint} \
    --cross_attn_every_n_layers 4 \
    --dataset_resampled \
    --loss_multiplier_calvin 1.0 \
    --workers 1 \
    --lr_scheduler constant \
    --warmup_steps 5000 \
    --learning_rate 1e-4 \
    --save_every_iter 10000 \
    --from_scratch \
    --window_size 12 \
    --report_to_wandb \
    --wandb_project RoboFlamingo \
    --wandb_entity aklab > ${log_file} 2>&1

