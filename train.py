'''
Author: Jikun Kang
Date: 1969-12-31 19:00:00
LastEditTime: 2023-01-24 11:19:08
LastEditors: Jikun Kang
FilePath: /MDT/train.py
'''

import random
import namegenerator
import argparse
import tqdm
import os
import socket
from pathlib import Path
import time
from typing import Optional
import torch
import numpy as np
import wandb
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from src.env_wrapper import build_env_fn
from src.create_dataset import create_dataset
from src.env_utils import ATARI_NUM_ACTIONS, ATARI_NUM_REWARDS, ATARI_RETURN_RANGE
from src.model import DecisionTransformer
from torch.utils.data import Dataset
from src.trainer import Trainer

os.environ['CUDA_VISIBLE_DEVICES']="1,2,3,4,5,6,7"

class StateActionReturnDataset(Dataset):

    def __init__(
        self,
        obs,
        block_size,
        actions,
        done_idxs,
        rtgs,
        timesteps,
        rewards
    ):
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.obs = obs
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.rewards = rewards

    def __len__(self):
        return len(self.obs) - self.block_size

    def __getitem__(self, idx):
        block_size = self.block_size // 3
        done_idx = idx + block_size
        for i in self.done_idxs:
            if i > idx:  # first done_idx greater than idx
                done_idx = min(int(i), done_idx)
                break
        idx = done_idx - block_size
        if idx < 0:
            idx = 0
            done_idx = idx + block_size

        states = self.obs[idx:done_idx].to(
            dtype=torch.float32) #.reshape(block_size, -1)  # (block_size, 3*64*64)
        states = states / 255.
        actions = self.actions[idx:done_idx].to(
            dtype=torch.long).unsqueeze(1)  # (block_size, 1)
        rtgs = self.rtgs[idx:done_idx].to(
            dtype=torch.float32).unsqueeze(1)
        # timesteps = torch.tensor(
        #     self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)
        rewards = self.rewards[idx:done_idx].to(
            dtype=torch.float32).unsqueeze(1)

        return states, rtgs, actions, rewards

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def run(args):
    # set seed 
    set_seed(args.seed)

    # set saving directory
    run_dir = Path(os.path.dirname(os.path.abspath(__file__))
                   + "/results") / args.game_name / args.experiment_name / namegenerator.gen()
    print(f"The run dir is {str(run_dir)}")
    if not run_dir.exists():
        os.makedirs(str(run_dir))

    # Init Logger
    if args.use_wandb:
        run = wandb.init(
            config=args,
            project=args.experiment_name,
            entity=args.user_name,
            notes=socket.gethostname(),
            name=f"seed_{str(args.seed)}",
            group=args.game_name,
            dir=str(run_dir),
            job_type='training',
            reinit=True
        )
        logger = None
    else:
        logger = SummaryWriter(run_dir)

    # init model
    dt_model = DecisionTransformer(
        num_actions=ATARI_NUM_ACTIONS,
        num_rewards=ATARI_NUM_REWARDS,
        return_range=ATARI_RETURN_RANGE,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        seq_len=args.seq_len,
        attn_drop=args.attn_drop,
        resid_drop=args.resid_drop,
        predict_reward=True,
        single_return_token=True,
        device=args.device,
        create_hnet=args.create_hnet,
    )
    
    if args.n_gpus:
        dt_model = nn.DataParallel(dt_model)

    # init train_dataset
    obss, actions, returns, done_idxs, rtgs, timesteps, rewards = create_dataset(
        args.num_buffers, args.data_steps, args.data_dir_prefix,
        args.trajectories_per_buffer)
    train_dataset = StateActionReturnDataset(
        obss, args.seq_len*3, actions, done_idxs, rtgs, timesteps, rewards)
    env_fn = build_env_fn(args.eval_game_name)
    env_batch = [env_fn()
                 for i in range(args.num_eval_envs)]

    optimizer = torch.optim.AdamW(
            dt_model.parameters(),
            lr=args.optimizer_lr,
            weight_decay=args.weight_decay,
        )

    trainer = Trainer(model=dt_model,
                      train_dataset=train_dataset,
                      eval_envs=env_batch, 
                      args=args,
                      optimizer=optimizer,
                      run_dir=run_dir,
                      grad_norm_clip=args.grad_norm_clip,
                      log_interval=args.log_interval,
                      use_wandb=args.use_wandb,
                      n_gpus=args.n_gpus)
    total_params = sum(params.numel() for params in dt_model.parameters())
    print(f"======> Total number of params are {total_params}")
    trainer.train()

    # close logger
    if args.use_wandb:
        run.finish()
    else:
        logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model configs
    # parser.add_argument('--embed_dim', type=int, default=1024) # 1024
    parser.add_argument('--n_embd', type=int, default=1280) # 1280
    parser.add_argument('--n_layer', type=int, default=10) # 10
    parser.add_argument('--n_head', type=int, default=2)
    parser.add_argument('--seq_len', type=int, default=28)
    parser.add_argument('--attn_drop', type=float, default=0.1)
    parser.add_argument('--resid_drop', type=float, default=0.1)
    parser.add_argument('--create_hnet', action='store_true', default=False)

    # Logging configs
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--use_wandb', action='store_true', default=False)
    parser.add_argument("--user_name", type=str, default='jaxonkang',
                    help="[for wandb usage], to specify user's name for simply collecting training data.")
    parser.add_argument("--n_gpus", action='store_true', default=False)

    # Training configs
    parser.add_argument('--max_epochs', type=int, default=100)
    parser.add_argument('--steps_per_iter', type=int, default=10000)
    parser.add_argument('--seed', type=int, default=123)

    # Evaluation configs
    parser.add_argument('--eval_steps', type=int, default=5000)
    parser.add_argument('--eval_game_name', type=str, default='Amidar')
    parser.add_argument('--num_eval_envs', type=int, default=16)

    # Optimizer configs
    parser.add_argument('--optimizer_lr', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--grad_norm_clip', type=float, default=1.)
    parser.add_argument('--num_workers', type=int, default=0)

    # Dataset related
    parser.add_argument('--data_steps', type=int, default=500000)
    parser.add_argument('--num_buffers', type=int, default=50)
    parser.add_argument('--game_name', type=str, default='Amidar')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--trajectories_per_buffer', type=int, default=10,
                        help='Number of trajectories to sample from each of the buffers.')
    parser.add_argument('--data_dir_prefix', type=str, default='dataset/2/')
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument("--save_freq", default=10, type=int)
    parser.add_argument('--experiment_name', default='atari', type=str)

    args = parser.parse_args()
    run(args)