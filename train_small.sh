 ###
 # @Author: Jikun Kang
 # @Date: 1969-12-31 19:00:00
 # @LastEditTime: 2023-01-18 17:52:55
 # @LastEditors: Jikun Kang
 # @FilePath: /MDT/train_small.sh
### 
python train.py --use_wandb --n_embd=512 --n_layer=4 --n_head=8 --n_gpus --num_workers=1