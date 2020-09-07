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

actor_net = SimpleActorNetwork(4, env.action_space.n)
critic_net = SimpleCriticNetwork(4)
target_critic_net = SimpleCriticNetwork(4)
target_critic_net.load_state_dict(critic_net.state_dict())

gamma = 0.9
batch_size = 128
epsilon_steps = 100000
memory_size = 10000
episodes = 10000
actor_lr = 0.0025
critic_lr = 0.0025
target_update = 15

parameters = {
    "env": env.unwrapped.spec.id,
    "gamma": gamma,
    "batch_size": batch_size,
    "epsilon_steps": epsilon_steps,
    "memory_size": memory_size,
    "episodes": episodes,
    "actor_lr": actor_lr,
    "critic_lr": critic_lr,
    "target_update": target_update,
}

callbacks = [
    cb.LogToStdout(),
    cb.LogToMLFlow(parameters, run_name="Actor Critic - Cartpole"),
]

memory = RandomReplayMemory(memory_size, env.observation_space.shape)
agent = ActorCriticAgent(
    env,
    memory,
    actor_net,
    critic_net,
    target_critic_net,
    actor_lr=actor_lr,
    critic_lr=critic_lr,
    batch_size=batch_size,
    target_update=target_update,
    callbacks=callbacks,
    exploration_steps=epsilon_steps,
)
agent.train(num_episodes=episodes, render=True)
