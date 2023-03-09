
###
 # @Author: Jikun Kang
 # @Date: 1969-12-31 19:00:00
 # @LastEditTime: 2023-02-02 10:26:11
 # @LastEditors: Jikun Kang
 # @FilePath: /MDT/train_medium_hnet.sh
### 
use_wandb=${2}
data_steps=${1}
samples=${3}

echo "training_samples="$samples"-data_steps="$data_steps

python train.py --create_hnet --n_embd=768 --n_layer=6 --use_wandb=$use_wandb\
  --n_head=12 --n_gpus --num_workers=10 --max_epochs=1000 --data_steps $data_steps --training_samples=$samples\
  --train_game_list 'Boxing'\
  --eval_game_list 'Boxing'