'''
Author: Jikun Kang
Date: 1969-12-31 19:00:00
LastEditTime: 2022-12-28 18:16:01
LastEditors: Jikun Kang
FilePath: /MDT/src/utils.py
'''

from typing import Tuple
import torch.nn.functional as F
import torch


def encode_return(ret: torch.Tensor, ret_range: Tuple[int]) -> torch.Tensor:
    """Encode return values into discrete return tokens."""
    ret = ret.type(torch.int32)
    ret = torch.clip(ret, ret_range[0], ret_range[1])
    ret = ret-ret_range[0]
    return ret


def encode_reward(rew: torch.Tensor) -> torch.Tensor:
    """
    Encode reward values
    # 0: no reward 1: positive reward 2: terminal reward 3: negative reward
    """
    rew = (rew > 0)*1+(rew < 0)*3
    return rew.to(dtype=torch.int32)


def cross_entropy(logits, labels):
    """Applies sparse cross entropy loss between logits and target labels"""
    labels = F.one_hot(logits, logits.shape[-1])
    loss = -labels * F.log_softmax(logits)
    return torch.mean(loss)


def accuracy(logits, labels):
    predicted_label = torch.argmax(logits, -1)
    acc = torch.equal(predicted_label, labels).to(dtype=torch.float32)
    return torch.mean(acc)
