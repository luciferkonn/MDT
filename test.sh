###
 # @Author: Jikun Kang
 # @Date: 1969-12-31 19:00:00
 # @LastEditTime: 2023-03-22 10:10:38
 # @LastEditors: Jikun Kang
 # @FilePath: /MDT/train_small_hnet.sh
### 

data_steps=${1}
use_wandb=${2}
samples=${3}
model_path=${4}
gw=${5}
create_hnet=${6}
device=${7}

echo "python train.py data_steps use_wandb samples model_path gw create_hnet"

python train.py --create_hnet=$create_hnet --max_epochs=1000 --eval_freq 10 --n_embd=512 --n_layer=1 --use_wandb=$use_wandb\
  --n_head=1 --device=$device --n_gpus --num_workers=10 --data_steps $data_steps --training_samples=$samples --use_gw=$gw\
  --load_path=$model_path --use_gw=$gw\
  --train_game_list 'Boxing'\
  --eval_game_list 'Boxing'