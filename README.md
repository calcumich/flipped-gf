# Glance - Flipped - VQA Project

## Dataset Exploration
### STAR EDA
Look at STAR EDA python notebook
### YTCommentQA
Look at pre processing YTCommentQA python notebook

## Setup
To install requirements, run:
```
git clone https://github.com/calcumich/flipped-gf
cd Flipped-VQA
mkdir data
conda create -n flipped-vqa python=3.8
conda activate flipped-vqa
sh setup.sh
```

## Dataset & LLaMA Preparation

You can download preprocessed STAR dataset at [here](https://drive.google.com/drive/folders/1WuvatnwVXphXlSdcW9UpUuIjs1vn1Tms). Put them in ```./data```. Also, you can download original LLaMA at [here](https://github.com/facebookresearch/llama/tree/llama_v1), and put the checkpoint in ```./pretrained```. 

```
./pretrained
   └─ llama
       |─ 7B
       |   |─ consolidated.00.pth
       |   └─ params.json
       └─ generation.py
       └─ model.py
       └─ tokenizer.py

./data
   |
   |─ star
```

## Training LLaMA-VQA (LLaMA + Flipped-VQA)

```
torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 4 train.py --model 7B \
--max_seq_len 128 --batch_size 8 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset star \
--blr 9e-2 --weight_decay 0.16 --output_dir ./checkpoint/star --accum_iter 1 --vaq --qav
```

## Evaluation
From the training command, simply replace ```train.py``` to ```eval.py``` and add ```--resume ./your/checkpoint.pth```.



