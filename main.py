import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.wrappers import AtariPreprocessing, FrameStack, TransformObservation
from torch.optim import Adam

from autonomous_systems_project.dqn import AtariDQN
from autonomous_systems_project.memory import RandomReplayMemory
from autonomous_systems_project.policy import epsilon_greedy
from autonomous_systems_project.training import train_dqn

frames = 4

frame_h = 84
frame_w = 84

env = gym.make("BreakoutNoFrameskip-v0")
env = AtariPreprocessing(env, scale_obs=True)
env = FrameStack(env, frames)
policy_net = AtariDQN(frame_h, frame_w, frames, env.action_space.n)
target_net = AtariDQN(frame_h, frame_w, frames, env.action_space.n)
target_net.load_state_dict(policy_net.state_dict())

optimizer = Adam(policy_net.parameters(), lr=0.00025)

memory = RandomReplayMemory(10000)
train_dqn(env, policy_net, target_net, optimizer, memory, episodes=1000)
