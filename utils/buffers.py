from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, NamedTuple

import numpy as np
import torch as th
from gymnasium import spaces

try:
    import psutil
except ImportError:
    psutil = None

__all__ = [
    "BaseBuffer",
    "ReplayBuffer",
    "ReplayBufferSamples",
]


class ReplayBufferSamples(NamedTuple):
    observations: th.Tensor
    actions: th.Tensor
    next_observations: th.Tensor
    dones: th.Tensor
    rewards: th.Tensor


class BaseBuffer(ABC):

    observation_space: spaces.Space
    obs_shape: tuple[int, ...]

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        device: th.device | str = "auto",
        n_envs: int = 1,
):

