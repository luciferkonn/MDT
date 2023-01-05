'''
Author: Jikun Kang
Date: 1969-12-31 19:00:00
LastEditTime: 2023-01-04 15:34:47:2
LastEditors: Jikun Kang
FilePath: /MDT/train.py
'''

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
from torch.utils.tensorboard import SummaryWriter
from create_dataset import create_dataset
from env_utils import ATARI_NUM_ACTIONS, ATARI_RETURN_RANGE
from model import DecisionTransformer
from torch.utils.data import Dataset

from trainer import Trainer


class StateActionReturnDataset(Dataset):

    def __init__(
        self,
        data,
        block_size,
        actions,
        done_idxs,
        rtgs,
        timesteps,
        rewards
    ):
        self.block_size = block_size
        self.vocab_size = max(actions) + 1
        self.data = data
        self.actions = actions
        self.done_idxs = done_idxs
        self.rtgs = rtgs
        self.timesteps = timesteps
        self.rewards = rewards

    def __len__(self):
        return len(self.data) - self.block_size

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

        states = torch.stack(self.data[idx:done_idx]).to(
            dtype=torch.float32).reshape(block_size, -1)  # (block_size, 3*64*64)
        states = states / 255.
        actions = torch.stack(self.actions[idx:done_idx].tolist()).to(
            dtype=torch.long).squeeze(1)  # (block_size, 1)
        rtgs = torch.stack(self.rtgs[idx:done_idx].tolist()).to(
            dtype=torch.float32).squeeze(1)
        # timesteps = torch.tensor(
        #     self.timesteps[idx:idx+1], dtype=torch.int64).unsqueeze(1)
        rewards = torch.stack(self.rewards[idx:done_idx].tolist()).to(
            dtype=torch.float32).squeeze(1)

        return states, rtgs, actions, rewards


def run(args):
    run_dir = Path(os.path.dirname(os.path.abspath(__file__))
                   + "/results") / args.game_name / args.experiment_name
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
            name=f"mode_{args.mode}_seed_{str(args.seed)}",
            group=args.game_name,
            dir=str(run_dir),
            job_type='training',
            reinit=True
        )
        logger = None
    else:
        if not run_dir.exists():
            curr_run = 'run1'
        else:
            exst_run_nums = [int(str(folder.name).split('run')[1]) for folder in run_dir.iterdir() if
                             str(folder.name).startswith('run')]
            if len(exst_run_nums) == 0:
                curr_run = 'run1'
            else:
                curr_run = 'run%i' % (max(exst_run_nums) + 1)
        run_dir = run_dir / curr_run
        if not run_dir.exists():
            os.makedirs(str(run_dir))
        logger = SummaryWriter(run_dir)

    # init model
    dt_model = DecisionTransformer(
        num_actions=ATARI_NUM_ACTIONS,
        num_rewards=ATARI_RETURN_RANGE,
        n_layer=args.n_layer,
        n_embd=args.n_embd,
        n_head=args.n_head,
        seq_len=args.seq_len,
        attn_drop=args.attn_drop,
        resid_drop=args.resid_drop,
        predict_reward=True,
        single_return_token=True,
        conv_dim=args.conv_dim,
    )

    # init train_dataset
    obss, actions, returns, done_idxs, rtgs, timesteps, rewards = create_dataset(
        args.num_buffers, args.num_steps, args.game_name, args.data_dir_prefix,
        args.trajectories_per_buffer)
    train_dataset = StateActionReturnDataset(
        obss, args.context_length*3, actions, done_idxs, rtgs, timesteps)
    # TODO: init test_dataset

    trainer = Trainer(model=dt_model, train_dataset=train_dataset,
                      test_dataset=None, args=args)
    trainer.train()

    # close logger
    if args.use_wandb:
        run.finish()
    else:
        logger.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Model configs
    parser.add_argument('--embed_dim', type=int, default=1024)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=3)
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--attn_drop', type=float, default=0.1)
    parser.add_argument('--resid_drop', type=float, default=0.1)
    parser.add_argument('--conv_dim', type=int, default=256)

    # Logging configs
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--use_wandb', action='store_true', default=False)

    # Training configs
    parser.add_argument('--max_iter', type=int, default=100)
    parser.add_argument('--steps_per_iter', type=int, default=10000)

    # Optimizer configs
    parser.add_argument('--optimizer_lr', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    
    # Dataset related
    parser.add_argument('--num_steps', type=int, default=2)
    parser.add_argument('--num_buffers', type=int, default=50)
    parser.add_argument('--game_name', type=str, default='bigfish')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--trajectories_per_buffer', type=int, default=10, help='Number of trajectories to sample from each of the buffers.')
    parser.add_argument('--data_dir_prefix', type=str, default='../data/')
    parser.add_argument('--max_len', type=int, default=1)
    parser.add_argument('--device', type=str, default='cuda')

    parser.add_argument("--save_freq", default=10, type=int)

    args = parser.parse_args()
    run(args)
