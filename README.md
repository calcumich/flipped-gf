# Glance - Flipped - VQA Project

## Dataset Exploration
### STAR EDA
Look at STAR EDA python notebook (Star_EDA.ipynb)
### YTCommentQA
Look at pre processing YTCommentQA python notebook (pre-processing-YTCommentsQA.ipynb)

## Reference:
Our model is based heavily on [Flipped-VQA](https://github.com/mlvlab/Flipped-VQA), and we aim to combine it's innovations with those of [Glance-and-Focus](https://github.com/ByZ0e/Glance-Focus).

## Setup
To install requirements, run:
```
git clone https://github.com/calcumich/flipped-gf
cd flipped-gf
mkdir data
conda create -n flipped-gf python=3.8
conda activate flipped-gf
sh setup.sh
```

## Dataset & LLaMA Preparation

You can download preprocessed STAR dataset at [here](https://drive.google.com/drive/folders/1WuvatnwVXphXlSdcW9UpUuIjs1vn1Tms). Put them in ```./data```. Also, you can download the checkpoint for LLaMa by requesting access from Meta and put it in ```./pretrained```. 

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
If you have access to multiple nodes, you can take advantage of PyTorch's [distributed features](https://pytorch.org/tutorials/beginner/dist_overview.html)
```
torchrun --rdzv_endpoint 127.0.0.1:1234 --nproc_per_node 4 train.py --model 7B \
--max_seq_len 128 --batch_size 8 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. --max_feats 10 --dataset star \
--blr 9e-2 --weight_decay 0.16 --output_dir ./checkpoint/star --accum_iter 1 --vaq --qav
```
If not, a command like this can also work:
```
python3 train.py --model 7B --max_seq_len 128 --batch_size 2 --epochs 5 --warmup_epochs 2 --bias 3 --tau 100. \
--max_feats 10 --dataset star --blr 9e-2 --weight_decay 0.16 --output_dir ./checkpoint/star --accum_iter 1 --vaq --qav --num_workers 1
```

## Evaluation
From the training command, simply replace ```train.py``` to ```eval.py``` and add ```--resume ./your/checkpoint.pth```.

## Greatlakes at UM
To run training or inference script on Greatlakes refer example_greatlakes_job.sh 



