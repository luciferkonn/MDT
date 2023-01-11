'''
Author: Jikun Kang
Date: 2022-05-12 13:11:43
LastEditTime: 2023-01-10 10:19:55
LastEditors: Jikun Kang
FilePath: /MDT/src/trainer.py
'''
from abc import abstractclassmethod, abstractmethod
import os
from typing import Callable, Optional, Union
import numpy as np
import torch
import time
import wandb
from tqdm import tqdm
from tensorboardX import SummaryWriter
from gym import spaces
from torch.utils._pytree import tree_map
from torch.utils.data.dataloader import DataLoader


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_dataset,
        test_dataset,
        args,
        optimizer: Union[torch.optim.Optimizer, Callable],
        run_dir: str,
        grad_norm_clip: float,
        num_steps_per_iter: int = 2500,
        log_interval: bool = None,
        use_wandb: bool = False,
    ) -> None:
        self.model = model
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.args = args
        self.optimizer = optimizer
        self.device = args.device
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lambda steps: min((steps+1)/args.warmup_steps, 1))
        self.num_steps_per_iter = num_steps_per_iter
        self.log_interval = log_interval
        self.use_wandb = use_wandb
        self.grad_norm_clip = grad_norm_clip
        self.save_freq = args.save_freq
        self.max_epochs = args.max_epochs
        self.run_dir = run_dir
        self.model.to(device=self.device)

    def train(self):
        for epoch in range(self.args.max_epochs):
            # train model
            self.run_epoch(iter_num=epoch)
            if epoch % self.save_freq == 0 or epoch == (self.max_epochs - 1):
                tf_file_loc = os.path.join(
                    self.run_dir, f'tf_model_{epoch}.pt')
                print(f"The model is saved to {tf_file_loc}")
                torch.save(self.model.state_dict(), tf_file_loc)
            # evaluate model
            # self.evaluation_rollout(envs=,
            #                         policy_fn=,
            #                         num_steps=log_interval=)

    def run_epoch(
        self,
        iter_num: int,
    ):
        # Prepare some log infos
        trainer_loss = []
        logs = dict()
        train_start = time.time()
        self.model.train()
        data = self.train_dataset
        loader = DataLoader(data, shuffle=True, pin_memory=True,
                            batch_size=self.args.batch_size,
                            num_workers=self.args.num_workers)

        pbar = tqdm(enumerate(loader), total=len(loader))
        for t, (obs, rtg, actions, rewards) in pbar:
            obs = obs.to(self.device)
            rtg = rtg.to(self.device)
            actions = actions.to(self.device)
            rewards = rewards.to(self.device)
            inputs = {'observations': obs,
                      'returns-to-go': rtg,
                      'actions': actions,
                      'rewards': rewards}
            with torch.set_grad_enabled(True):
                result_dict = self.model(inputs=inputs, is_training=True)
                train_loss = result_dict['loss']
                # TODO: maybe add construction loss
            self.optimizer.zero_grad()
            train_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.grad_norm_clip)
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
            if self.log_interval and t % self.log_interval == 0:
                acc = result_dict['accuracy']*100
                pbar.set_description(
                    f"epoch {iter_num} steps: {t}: train loss {train_loss:.5f} accuracy {acc:.3f}%.")
                if self.use_wandb:
                    wandb.log({"episode_loss": train_loss,
                               "accuracy": acc,
                               'epoch': (iter_num)*self.num_steps_per_iter+t})
        training_time = time.time() - train_start
        logs['time/training'] = training_time
        return logs

    def evaluation_rollout(
        self,
        envs,
        policy_fn,
        num_steps=2500,
        log_interval=None
    ):
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

            actions = policy_fn(obs)

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
        return rew_sum, frames
