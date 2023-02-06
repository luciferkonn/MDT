 ###
 # @Author: Jikun Kang
 # @Date: 1969-12-31 19:00:00
 # @LastEditTime: 2023-01-27 16:31:41
 # @LastEditors: Jikun Kang
 # @FilePath: /MDT/train_small.sh
### 
python train.py --use_wandb=1 --n_embd=512 --create_hnet --n_layer=4 --n_head=8 --n_gpus \
  --training_samples=8000 --num_workers=10 --train_game_list 'Amidar'\
  --eval_game_list 'Amidar'
