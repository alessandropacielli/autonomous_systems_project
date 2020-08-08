import random

import gym
import torch
import torch.nn as nn


def epsilon_greedy(
    state: torch.Tensor, env: gym.Env, policy_network: nn.Module, epsilon: float
) -> int:
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_network(state).max(1)[1].view(1, 1).item()
    else:
        return env.action_space.sample()
