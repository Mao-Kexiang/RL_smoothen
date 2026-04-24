"""目标函数定义：Rastrigin 函数（崎岖的多局部极值函数）"""

import torch
import numpy as np


def rastrigin(x: torch.Tensor) -> torch.Tensor:
    """Rastrigin 函数（最小值为 0，在原点取到）。

    Args:
        x: shape (..., 2)，取值范围建议 [-5, 5]²
    Returns:
        shape (...)，函数值（越小越好）
    """
    A = 10.0
    d = x.shape[-1]
    return A * d + (x ** 2 - A * torch.cos(2 * np.pi * x)).sum(dim=-1)


def reward(x: torch.Tensor) -> torch.Tensor:
    """奖励函数 = 负 Rastrigin（越大越好，最优值 0 在原点取到）。"""
    return -rastrigin(x)


def reward_np(x: np.ndarray) -> float:
    """numpy 版本，供 CMA-ES 使用。"""
    A = 10.0
    d = x.shape[-1]
    return -(A * d + np.sum(x ** 2 - A * np.cos(2 * np.pi * x), axis=-1))
