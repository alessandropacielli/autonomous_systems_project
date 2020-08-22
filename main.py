import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.wrappers import AtariPreprocessing, FrameStack, TransformObservation

from autonomous_systems_project.dqn import DQN
from autonomous_systems_project.policy import epsilon_greedy
from autonomous_systems_project.preprocessing import crop, rescale
from autonomous_systems_project.training import train_dqn

frames = 4

final_y = 84
final_x = 84

hidden = 512

env = gym.make("BreakoutNoFrameskip-v0")
# env = TransformObservation(env, crop(31, -1, 7, -8))
env = AtariPreprocessing(env, scale_obs=True)
# env = TransformObservation(env, rescale((final_y, final_x)))
env = FrameStack(env, frames)
policy_net = DQN(final_y, final_x, frames, hidden, env.action_space.n)
train_dqn(policy_net, env, epsilon_greedy, eps_decay=1000, num_episodes=100)
