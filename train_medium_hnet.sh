
###
 # @Author: Jikun Kang
 # @Date: 1969-12-31 19:00:00
 # @LastEditTime: 2023-03-20 17:01:48
 # @LastEditors: Jikun Kang
 # @FilePath: /MDT/train_medium_hnet.sh
### 
data_steps=${1}
use_wandb=${2}
samples=${3}
model_path=${4}
gw=${5}

echo "training_samples="$samples"-data_steps="$data_steps

python train.py --create_hnet --n_embd=768 --n_layer=6 --use_wandb=$use_wandb\
  --n_head=12 --n_gpus --num_workers=10 --max_epochs=5000 --data_steps $data_steps --training_samples=$samples\
  --load_path=$model_path --use_gw=$gw\
  --train_game_list 'Boxing'\
  --eval_game_list 'Boxing'