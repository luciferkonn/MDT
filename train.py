'''
Author: Jikun Kang
Date: 1969-12-31 19:00:00
LastEditTime: 2023-01-04 10:22:40
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
from torch.utils._pytree import tree_map
from torch.utils.tensorboard import SummaryWriter
from env_utils import ATARI_NUM_ACTIONS, ATARI_RETURN_RANGE
from model import DecisionTransformer


def dataloader():
    pass


def train_step(
    iter_num: int,
    dt_agent,
    inputs,
    optimizer,
    use_wandb: bool,
    grad_norm_clip: float,
):
    result_dict = dt_agent(inputs=inputs, is_training=True)
    train_loss = result_dict['loss']
    # TODO: maybe add construction loss
    optimizer.zero_grad()
    train_loss.backward()
    torch.nn.utils.clip_grad_norm_(dt_agent.parameters(), grad_norm_clip)
    optimizer.step()
    train_loss = train_loss.detach().cpu().item()
    return train_loss


def train_iteration(
    loader,
    dt_agent,
    optimizer,
    iter_num: int,
    grad_norm_clip:float,
    num_steps_per_iter: int = 2500,
    log_interval: bool = None,
    use_wandb: bool = False,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
):
    # Prepare some log infos
    trainer_loss = []
    logs = dict()
    train_start = time.time()
    dt_agent.train()
    # Merge observations into a single dictionary
    obs_list = [env.reset() for env in envs]
    num_batch = len(envs)
    obs = tree_map(lambda *arr: np.stack(arr, axis=0), *obs_list)
    ret = np.zeros([num_batch, 8])
    done = np.zeros(num_batch, dtype=np.int32)
    rew_sum = np.zeros(num_batch, dtype=np.float32)
    frames = []

    pbar = tqdm(enumerate(loader), total=num_steps_per_iter)
    for t, (obs, rtg, actions, rewards) in range(num_steps_per_iter):
        inputs = {'observations': obs,
                  'returns-to-go': rtg,
                  'actions': actions,
                  'rewards': rewards}
        result_dict = dt_agent(inputs=inputs, is_training=True)
        train_loss = result_dict['loss']
        # TODO: maybe add construction loss
        optimizer.zero_grad()
        train_loss.backward()
        torch.nn.utils.clip_grad_norm_(dt_agent.parameters(), grad_norm_clip)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        if log_interval and t % log_interval == 0:
            print(
                f'====>Training iteration: {iter_num}, steps: {t}, current loss: {train_loss}')
            if use_wandb:
                wandb.log({"episode_loss": train_loss,
                           'epoch': (iter_num-1)*num_steps_per_iter+t})
    training_time = time.time() - train_start
    logs['time/training'] = training_time
    return logs


def evaluation_rollout(rng, envs, policy_fn, num_steps=2500, log_interval=None):
    """Roll out a batch of environments under a given policy function."""
    # observations are dictionaries. Merge into single dictionary with batched
    # observations.
    obs_list = [env.reset() for env in envs]
    num_batch = len(envs)
    obs = tree_map(lambda *arr: np.stack(arr, axis=0), *obs_list)
    ret = np.zeros([num_batch, 8])
    done = np.zeros(num_batch, dtype=np.int32)
    rew_sum = np.zeros(num_batch, dtype=np.float32)
    frames = []
    for t in range(num_steps):
        # Collect observations
        frames.append(
            np.concatenate([o['observations'][-1, ...] for o in obs_list], axis=1))
        done_prev = done

        actions, rng = policy_fn(rng, obs)

        # Collect step results and stack as a batch.
        step_results = [env.step(act) for env, act in zip(envs, actions)]
        obs_list = [result[0] for result in step_results]
        obs = tree_map(lambda *arr: np.stack(arr, axis=0), *obs_list)
        rew = np.stack([result[1] for result in step_results])
        done = np.stack([result[2] for result in step_results])
        # Advance state.
        done = np.logical_or(done, done_prev).astype(np.int32)
        rew = rew * (1 - done)
        rew_sum += rew
        if log_interval and t % log_interval == 0:
            print('step: %d done: %s reward: %s' % (t, done, rew_sum))
        # Don't continue if all environments are done.
        if np.all(done):
            break
    return rew_sum, frames, rng


def run(args):
    run_dir = Path(os.path.dirname(os.path.abspath(__file__))
                   + "/results") / args.game_name / args.experiment_name
    print(f"The run dir is {str(run_dir)}")
    if not run_dir.exists():
        os.makedirs(str(run_dir))

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

    device = args.device
    env_name, dataset = args.env, args.dataset
    model_type = args.model_type

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

    dt_optimizer = torch.optim.AdamW(
        dt_model.parameters(),
        lr=args.optimizer_lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        dt_optimizer, lambda steps: min((steps+1)/args.warmup_steps, 1))
    for iter_num in range(args.max_iter):
        logs = train_iteration(
            rng=,
            envs=,
            dt_agent=dt_model,
            optimizer=dt_optimizer,
            num_steps_per_iter=args.steps_per_iter,
            log_interval=args.log_interval,
            use_wandb=args.use_wandb,
        )

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

    args = parser.parse_args()
    run(args)
