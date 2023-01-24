
###
 # @Author: Jikun Kang
 # @Date: 1969-12-31 19:00:00
 # @LastEditTime: 2023-01-24 12:30:10
 # @LastEditors: Jikun Kang
 # @FilePath: /MDT/train_small_hnet.sh
### 
python train.py --create_hnet --use_wandb --n_embd=512 --n_layer=4\
  --n_head=8 --n_gpus --num_workers=10\
  --train_game_list 'Amidar' 'Atlantis' 'BankHeist'