import random

import gym
import torch
import torch.nn as nn


def epsilon_greedy(
    state: torch.Tensor, env: gym.Env, policy_network: nn.Module, epsilon: float
) -> int:
    if random.random() > epsilon:
        with torch.no_grad():
            return policy_network(state).argmax().item()
    else:
        return env.action_space.sample()
