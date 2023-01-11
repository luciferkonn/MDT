'''
Author: Jikun Kang
Date: 1969-12-31 19:00:00
LastEditTime: 2023-01-09 17:16:55
LastEditors: Jikun Kang
FilePath: /MDT/src/utils.py
'''

from typing import Optional, Tuple
import torch.nn.functional as F
import torch


def encode_return(ret: torch.Tensor, ret_range: Tuple[int]) -> torch.Tensor:
    """Encode return values into discrete return tokens."""
    ret = ret.to(dtype=torch.int32)
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
    labels = F.one_hot(labels.to(dtype=torch.int64),
                       logits.shape[-1]).squeeze(2)
    loss = -labels * F.log_softmax(logits)
    return torch.mean(loss)


def accuracy(logits, labels):
    predicted_label = torch.argmax(logits, -1)
    acc = torch.eq(predicted_label, labels.squeeze(-1)).to(dtype=torch.float32)
    return torch.mean(acc)


def sample_from_logits(
    logits: torch.Tensor,
    deterministic: Optional[bool] = False,
    top_k: Optional[int] = None,
    top_percentile: Optional[float] = None
):
    if deterministic:
        sample = torch.argmax(logits, dim=-1)
    else:
        if top_percentile is not None:
            percentile = torch.quantile(logits, top_percentile/100, dim=-1)
            logits = torch.where(logits>percentile.unsqueeze(-1), logits, -torch.inf)
        if top_k is not None:
            logits, top_indices = torch.topk
