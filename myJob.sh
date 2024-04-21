#!/bin/bash
#SBATCH --job-name=vqa-training-and-troubleshooting
#SBATCH --mail-user=cjcal@umich.edu
#SBATCH --mail-type=BEGIN,END 
#SBATCH --output=vqa_Troubleshooting_18APR.out
#SBATCH --error=vqa_Troubleshooting_18APR.err
#SBATCH --nodes=1
#SBATCH --gres=gpu
#SBATCH --mem=8GB
#SBATCH --time=0:03:49
#SBATCH --account=eecs545w24_class
#SBATCH --partition=gpu

nvidia-smi
source /sw/pkgs/arc/python3.10-anaconda/2023.03/etc/profile.d/conda.sh
source ~/.bashrc
cd /home/cjcal/545/Flipped-VQA-S
conda activate flipped-vqa

train.py --model 7B --max_seq_len 128 --batch_size 4 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset star --blr 9e-2 --weight_decay 0.16 --output_dir ./checkpoint/star --accum_iter 1 --vaq --qav  

