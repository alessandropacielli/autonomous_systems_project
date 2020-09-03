import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.wrappers import AtariPreprocessing, FrameStack
from torch.optim import Adam

import autonomous_systems_project.callbacks as cb
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
gamma = 0.9
batch_size = 256
epsilon_steps = 1000
episodes = 100
lr = 0.0001

memory = RandomReplayMemory(10000)
train_dqn(
    env,
    policy_net,
    target_net,
    optimizer,
    memory,
    gamma=gamma,
    batch_size=batch_size,
    epsilon_steps=epsilon_steps,
    episodes=episodes,
    callbacks=[
        cb.LogToStdout(),
        cb.LogToMLFlow(
            "http://127.0.0.1:5000",
            "test",
            {
                "env": "CartPole-v1",
                "gamma": gamma,
                "batch_size": batch_size,
                "epsilon_steps": epsilon_steps,
                "episodes": episodes,
                "optimizer": "Adam",
                "lr": lr,
            },
        ),
    ],
)
