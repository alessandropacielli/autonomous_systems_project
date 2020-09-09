import datetime

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym.wrappers import FrameStack, GrayScaleObservation, Monitor, TransformObservation
from torch.optim import Adam

import autonomous_systems_project.callbacks as cb
from autonomous_systems_project.agents import (
    ActorCriticAgent,
    DoubleDQNAgent,
    DQNAgent,
    SimpleActorNetwork,
    SimpleCriticNetwork,
    SimpleDQN,
)
from autonomous_systems_project.memory import RandomReplayMemory

env = gym.make("CartPole-v1")
env = TransformObservation(env, lambda obs: np.array(obs).astype(np.float32))

actor_net = SimpleActorNetwork(env.observation_space.shape[0], env.action_space.n)
critic_net = SimpleCriticNetwork(env.observation_space.shape[0])

gamma = 0.9
episodes = 500
actor_lr = 0.0001
critic_lr = 0.001

parameters = {
    "env": env.unwrapped.spec.id,
    "gamma": gamma,
    "episodes": episodes,
    "actor_lr": actor_lr,
    "critic_lr": critic_lr,
}

callbacks = [
    cb.LogToStdout(),
    cb.LogToMLFlow(parameters, run_name="Actor Critic"),
]

agent = ActorCriticAgent(
    env,
    actor_net,
    critic_net,
    actor_lr=actor_lr,
    critic_lr=critic_lr,
    callbacks=callbacks,
)
agent.train(num_episodes=episodes, render=True)
