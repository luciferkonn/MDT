###
 # @Author: Jikun Kang
 # @Date: 1969-12-31 19:00:00
 # @LastEditTime: 2023-03-20 16:54:33
 # @LastEditors: Jikun Kang
 # @FilePath: /MDT/train_small_hnet.sh
### 

data_steps=${1}
use_wandb=${2}
samples=${3}
gw=${4}

echo "training_samples="$samples"-data_steps="$data_steps

python train.py --create_hnet --max_epochs=1000 --eval_freq 10 --n_embd=512 --n_layer=4 --use_wandb=$use_wandb\
  --n_head=8 --n_gpus --num_workers=10 --data_steps $data_steps --training_samples=$samples --use_gw=$gw\
  --train_game_list 'Boxing'\
  --eval_game_list 'Boxing'