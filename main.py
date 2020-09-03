import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.wrappers import AtariPreprocessing, FrameStack, TransformObservation
from torch.optim import Adam

from autonomous_systems_project.dqn import AtariDQN, SimpleDQN
from autonomous_systems_project.memory import RandomReplayMemory
from autonomous_systems_project.policy import epsilon_greedy
from autonomous_systems_project.training import train_dqn

frames = 4

frame_h = 84
frame_w = 84

env = gym.make("CartPole-v1")
# env = AtariPreprocessing(env, scale_obs=True)
# env = FrameStack(env, frames)

policy_net = SimpleDQN(4, env.action_space.n)
target_net = SimpleDQN(4, env.action_space.n)
target_net.load_state_dict(policy_net.state_dict())

optimizer = Adam(policy_net.parameters())

memory = RandomReplayMemory(10000)
train_dqn(
    env,
    policy_net,
    target_net,
    optimizer,
    memory,
    gamma=0.9,
    batch_size=128,
    epsilon_steps=1000,
    episodes=1000,
)
