#!/bin/bash
#SBATCH --job-name=vqa-training-and-troubleshooting
#SBATCH --mail-user=cjcal@umich.edu
#SBATCH --mail-type=BEGIN,END 
#SBATCH --output=vqa_Troubleshooting_21APR_2.out
#SBATCH --error=vqa_Troubleshooting_21APR_2.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --mem=32GB
#SBATCH --time=7:59:59
#SBATCH --account=eecs545w24_class
#SBATCH --partition=spgpu

nvidia-smi
python3 train.py --model 7B --max_seq_len 128 --batch_size 4 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset star --blr 9e-2 --weight_decay 0.16 --output_dir ./checkpoint/star --accum_iter 1 --vaq --qav --num_workers 1 --resume ./checkpoint/star/checkpoint_best.pth 

