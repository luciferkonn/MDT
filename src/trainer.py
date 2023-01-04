'''
Author: Jikun Kang
Date: 2022-05-12 13:11:43
LastEditTime: 2023-01-03 10:01:19
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
from tensorboardX import SummaryWriter
from buffer import RolloutBuffer, get_action_dim, get_obs_shape
from gym import spaces
from decision_transformer.decision_transformer import get_action


class Trainer:
    def __init__(
        self,
        tf_model: torch.nn.Module,
        head_model: torch.nn.Module,
        hnet,
        optimizer: Union[torch.optim.Optimizer, Callable],
        batch_size: int,
        get_batch: Callable,
        loss_fn: Callable,
        log_dir: str,
        logger: Optional[SummaryWriter],
        observation_space: spaces.Space,
        action_space: spaces.Space,
        envs,
        target_rew=2000,
        scheduler=None,
        eval_fns: Optional[Callable] = None,
        device: Union[str, torch.device] = 'cpu',
        opt_net_path: Optional['str'] = None,
        lr: float = 0.001,
        weight_decay: float = 0.0001,
        out_mul: float = 1,
        unroll: int = 20,
        grad_norm_clip: int = 1,
        save_to_main: bool = False,
        use_hypernet: bool = False,
        max_len: int = 20,
        buffer_size: int = 1024,
        vision=False,
        use_wandb=False,
        run_dir=None,
    ) -> None:

        self.tf_model = tf_model
        self.head_model = head_model
        self.hnet = hnet
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.start_time = time.time()
        self.diagnostics = dict()
        self.log_dir = log_dir
        self.logger = logger
        self.device = device
        self.out_mul = out_mul
        self.unroll = unroll
        self.grad_norm_clip = grad_norm_clip
        self.save_to_main = save_to_main
        self.use_hypernet = use_hypernet
        self.target_reward = target_rew
        self.envs = envs
        self.max_len = max_len
        self.observation_space = observation_space
        self.action_space = action_space
        self.vision = vision
        self.use_wandb = use_wandb
        self.run_dir = run_dir

        self.state_dim = observation_space.shape[0]
        self.act_dim = get_action_dim(action_space)
        # self.act_dim = action_space.shape[0]

        self.n_params = 0
        self.train_l2o = False

        self.all_losses_ever = []
        self.all_losses = None
        if self.train_l2o:
            self.meta_opt.zero_grad()

        self.best = float('inf')

        self.rollout_buffer_list = []

        if self.get_batch is None:
            self.on_policy = True
            for _ in range(len(self.envs)):
                rollout_buffer = RolloutBuffer(
                    buffer_size=buffer_size, observation_space=observation_space,
                    action_space=action_space, device=device, max_len=max_len,
                )
                self.rollout_buffer_list.append(rollout_buffer)
        else:
            self.on_policy = False

    @abstractmethod
    def rollout_step(
        self,
        target_return,
        state_mean: float = 0.,
        state_std: float = 1.,
    ):
        pass

    @abstractclassmethod
    def train_step(
        self,
        num_step: int,
        fine_tune: bool = False,
    ):
        pass

    def train_iteration(
        self,
        num_steps: int,
        iter_num: int = 0,
        print_logs: bool = True,
        fine_tune: bool = False,
    ):
        train_losses = []
        logs = dict()

        train_start = time.time()
        self.tf_model.train()
        self.head_model.train()
        for i in range(num_steps):
            if self.on_policy:
                if not self.rollout_buffer_list[-1].full:
                    # collect data
                    self.rollout_step(target_return=self.target_reward,)
                    continue
            train_loss = self.train_step(i, fine_tune)
            print(
                f'====>Training iteration: {iter_num}, steps: {i}, current loss: {train_loss}')
            if self.use_wandb:
                wandb.log({"episode_loss": train_loss,
                          'epoch': (iter_num-1)*num_steps+i})
            train_losses.append(train_loss)
            if self.scheduler is not None:
                self.scheduler.step()

        training_time = time.time() - train_start
        logs['time/training'] = training_time

        eval_start = time.time()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.tf_model, self.head_model, self.hnet)
            for k, v in outputs.items():
                logs[f'eval/{k}'] = v
                if not self.use_wandb:
                    self.logger.add_scalar(f'eval/{k}', v, iter_num)

        if np.mean(train_losses) < self.best and not fine_tune:
            print("="*80)
            print(f'New best model! Saving the model to dir {self.run_dir}')
            print("="*80)
            if self.save_to_main:
                print("Save tf_model to the model folder")
                print("="*80)
                torch.save(self.tf_model.state_dict(),
                           'results/models/tf_model.pt')
            tf_file_loc = os.path.join(self.run_dir, 'tf_model.pt')
            h_model_file_loc = os.path.join(self.run_dir, 'head_model.pt')
            torch.save(self.tf_model.state_dict(), tf_file_loc)
            torch.save(self.head_model.state_dict(), h_model_file_loc)
            if self.use_hypernet:
                torch.save(self.hnet.state_dict(), 'hnet_model.pt')
            self.best = np.mean(train_losses)

        total_time = time.time() - self.start_time
        eval_time = time.time() - eval_start
        train_loss_mean = np.mean(train_losses)
        train_loss_std = np.std(train_losses)

        if not self.use_wandb:
            self.logger.add_scalar(
                'train/loss_mean', train_loss_mean, iter_num)
            self.logger.add_scalar('train/loss_std', train_loss_std, iter_num)
            self.logger.add_scalar('time/total', total_time, iter_num)
            self.logger.add_scalar('time/evaluation', eval_time, iter_num)
        logs['time/total'] = total_time
        logs['time/evaluation'] = eval_time
        logs['train/loss_mean'] = train_loss_mean
        logs['train/loss_std'] = train_loss_std

        for k, v in self.diagnostics.items():
            logs[k] = v

        if print_logs:
            print('=' * 80)
            print(f'Iteration: {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')
        return logs


class SequenceTrainer(Trainer):
    def __init__(
        self,
        scale: float = 1000.,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.env_dones = [True]*len(self.envs)
        self.env_states = [None]*len(self.envs)
        self.scale = scale

    def rollout_step(
        self,
        target_return,
        state_mean: float = 0.,
        state_std: float = 1.,
    ):
        self.tf_model.to(device=self.device)
        self.head_model.to(device=self.device)
        if self.hnet is not None:
            self.hnet.to(device=self.device)
        # state_total_dims = np.array(self.observation_space.shape)

        with torch.no_grad():
            self.tf_model.eval()
            self.head_model.eval()
            if self.hnet is not None:
                self.hnet.eval()

            for i, env, env_state, env_done in zip(range(len(self.envs)), self.envs, self.env_states, self.env_dones):
                if self.hnet is not None:
                    weights = self.hnet(cond_id=i)
                else:
                    weights = None
                if env_done:
                    # FIXME: procgen reset issue
                    # state = env.reset()
                    state = env_state
                else:
                    state = env_state
                if self.vision:
                    states = torch.from_numpy(state).to(
                        device=self.device, dtype=torch.float32).unsqueeze(1)
                else:
                    states = torch.from_numpy(state).reshape(1, self.state_dim).to(
                        device=self.device, dtype=torch.float32)
                actions = torch.zeros(
                    (0, self.act_dim), device=self.device, dtype=torch.float32)
                rewards = torch.zeros(
                    0, device=self.device, dtype=torch.float32)

                for t in range(self.max_len):
                    actions = torch.cat([actions, torch.zeros(
                        (1, self.act_dim), device=self.device)], dim=0)
                    rewards = torch.cat(
                        [rewards, torch.zeros(1, device=self.device)])

                    # target return
                    if isinstance(target_return, torch.Tensor):
                        target_return = target_return.clone().detach()
                    else:
                        target_return = torch.tensor(
                            target_return, device=self.device, dtype=torch.float32).reshape(1, 1)

                    # timesteps
                    timesteps = torch.tensor(
                        0, device=self.device, dtype=torch.long).reshape(1, 1)

                    action = get_action(
                        (states.to(dtype=torch.float32)-state_mean) / state_std,
                        actions.to(dtype=torch.float32),
                        target_return.to(dtype=torch.float32),
                        timesteps.to(dtype=torch.long),
                        self.state_dim,
                        self.act_dim,
                        tf_model=self.tf_model,
                        head_module=self.head_model,
                        weights=weights,
                        vision=self.vision,
                    )
                    actions[-1] = action
                    action = action.detach().cpu().numpy()

                    next_state, reward, done, _ = env.step(action)

                    if isinstance(reward, np.ndarray):
                        tensor_reward = torch.from_numpy(
                            reward).to(device=self.device)
                    pred_return = target_return[0, -
                                                1] - (tensor_reward/self.scale)

                    rtg = torch.cat(
                        [target_return, pred_return.reshape(1, 1)], dim=1)

                    timesteps = torch.cat([timesteps, torch.ones(
                        (1, 1), device=self.device, dtype=torch.long) * (t+1)], dim=1)

                    # add data to the buffer
                    self.rollout_buffer_list[i].add(
                        state, action, reward, done, rtg, timesteps, t)
                    state = next_state
                    if self.vision:
                        next_state = torch.from_numpy(next_state).to(
                            device=self.device, dtype=torch.float32).unsqueeze(1)
                    else:
                        next_state = torch.from_numpy(next_state).reshape(1, self.state_dim).to(
                            device=self.device, dtype=torch.float32)
                    states = torch.cat([states, next_state], dim=1)
                    if done:
                        break
                self.env_states[i] = state
                self.env_dones[i] = done

    def train_step(
        self,
        num_step: int,
        fine_tune: bool = False,
    ):
        if self.hnet is not None:
            self.hnet.train()
        self.head_model.train()
        if self.on_policy:
            states_list, actions_list, rewards_list = [], [], []
            dones_list, rtg_list, timesteps_list, attention_mask_list = [], [], [], []
            for rollout_buffer in self.rollout_buffer_list:
                states, actions, rewards, dones, rtg, timesteps, attention_mask = rollout_buffer.get(
                    self.batch_size)
                states_list.append(states)
                actions_list.append(actions)
                rewards_list.append(rewards)
                dones_list.append(dones)
                rtg_list.append(rtg)
                timesteps_list.append(timesteps)
                attention_mask_list.append(attention_mask)

        else:
            states_list, actions_list, rewards_list, dones_list, rtg_list,\
                timesteps_list, attention_mask_list = self.get_batch(
                    self.batch_size)
        states = states_list
        actions = actions_list
        rewards = rewards_list
        dones = dones_list
        rtg = rtg_list
        timesteps = timesteps_list
        attention_mask = attention_mask_list
        if self.vision:
            states = states.reshape(-1,
                                    states.shape[2], states.shape[3], states.shape[4])

        action_target = torch.clone(actions)

        if fine_tune:
            self.tf_model.eval()
            with torch.no_grad():
                if self.vision:
                    action_embeds = self.tf_model(
                        states, actions, rewards, rtg[:,
                                                      :, :-1], timesteps[:, :, :-1].squeeze(2),
                        attention_mask=attention_mask,
                    )
                else:
                    action_embeds = self.tf_model(
                        states, actions, rewards, rtg[:, :-
                                                      1], timesteps, attention_mask=attention_mask,
                    )
        else:
            if self.vision:
                action_preds = self.tf_model.forward_with_head(
                    states, actions, rewards, rtg[:, :-1], timesteps,
                    attention_mask=attention_mask,
                )
            else:
                action_embeds = self.tf_model(
                    states, actions, rewards, rtg[:, :-
                                                  1], timesteps, attention_mask=attention_mask,
                )

        loss = self.loss_fn(
            action_preds,
            action_target,
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
