import datetime

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.wrappers import FrameStack, GrayScaleObservation, Monitor, TransformObservation
from torch.optim import Adam

import autonomous_systems_project.callbacks as cb
from autonomous_systems_project.agents import DoubleDQNAgent, DQNAgent, SimpleDQN
from autonomous_systems_project.memory import RandomReplayMemory

env = gym.make("CartPole-v1")
env = TransformObservation(env, lambda obs: np.array(obs).astype(np.float32))

policy_net = SimpleDQN(4, env.action_space.n)
target_net = SimpleDQN(4, env.action_space.n)
target_net.load_state_dict(policy_net.state_dict())


gamma = 0.9
batch_size = 32
epsilon_steps = 1000
memory_size = 100000
episodes = 500
lr = 0.001
target_update = 5

parameters = {
    "env": env.unwrapped.spec.id,
    "gamma": gamma,
    "batch_size": batch_size,
    "epsilon_steps": epsilon_steps,
    "memory_size": memory_size,
    "episodes": episodes,
    "lr": lr,
    "target_update": target_update,
}

callbacks = [
    cb.LogToStdout(),
    cb.LogToMLFlow(parameters, run_name="DoubleDQN - Cartpole"),
]

memory = RandomReplayMemory(memory_size, env.observation_space.shape)
agent = DoubleDQNAgent(
    env,
    memory,
    policy_net,
    target_net,
    lr=lr,
    batch_size=batch_size,
    target_update=target_update,
    callbacks=callbacks,
    exploration_steps=epsilon_steps,
)
agent.train(num_episodes=episodes, render=True)
