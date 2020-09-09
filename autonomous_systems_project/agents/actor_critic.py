import math
import random

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm

from autonomous_systems_project.agents.common import Agent


class AtariCriticNetwork(nn.Module):
    def __init__(self, input_shape):
        super(AtariCriticNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, 1)
        )

    def _get_conv_out(self, shape):
        # Computes the output of the convolutional layers "empirically"
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # Compute convolution
        conv_out = self.conv(x)

        # Reshape according to batch dimension
        conv_out = conv_out.view(x.size()[0], -1)

        # Compute dense layer
        return self.fc(conv_out)


class AtariActorNetwork(nn.Module):
    def __init__(self, input_shape, num_actions):
        super(AtariActorNetwork, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )

        conv_out_size = self._get_conv_out(input_shape)
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions),
            nn.Softmax(),
        )

    def _get_conv_out(self, shape):
        # Computes the output of the convolutional layers "empirically"
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        # Compute convolution
        conv_out = self.conv(x)

        # Reshape according to batch dimension
        conv_out = conv_out.view(x.size()[0], -1)

        # Compute dense layer
        probs = self.fc(conv_out)
        return Categorical(probs)


class SimpleCriticNetwork(nn.Module):
    def __init__(self, num_inputs: int):
        super(SimpleCriticNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.fc(x)


class SimpleActorNetwork(nn.Module):
    def __init__(self, num_inputs: int, num_actions: int):
        super(SimpleActorNetwork, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(num_inputs, 64),
            nn.LeakyReLU(),
            nn.Linear(64, 32),
            nn.LeakyReLU(),
            nn.Linear(32, num_actions),
            nn.Softmax(),
        )

    def forward(self, x):
        distribution = Categorical(self.fc(x))
        return distribution


class ActorCriticAgent(Agent):
    def __init__(
        self,
        env,
        actor_net,
        critic_net,
        actor_lr=0.00025,
        critic_lr=0.00025,
        gamma=0.9,
        callbacks=[],
    ):
        super().__init__()

        # Env
        self.env = env
        self.state_shape = env.observation_space.shape

        # Device used for training
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Double DQN
        self.actor_net = actor_net
        self.critic_net = critic_net

        # TODO optimizer choice
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optimizer = torch.optim.Adam(
            self.actor_net.parameters(), lr=self.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_net.parameters(), lr=self.critic_lr
        )

        # Training parameters
        self.gamma = gamma
        self.loss_fn = nn.SmoothL1Loss().to(
            self.device
        )  # Also known as Huber loss <-- TODO loss choice

        # Counters and stats
        self.steps = 0
        self.episodes = 0
        self.reward_history = []
        self.critic_loss_history = []
        self.actor_loss_history = []

        # Callbacks
        self.callbacks = callbacks

    def get_metrics(self):
        episode = self.episodes
        return {
            "episode": episode,
            "step": self.steps,
            "episode_reward": self.reward_history[episode - 1],
            "average_reward": np.mean(self.reward_history),
            "average_reward_last_100": np.mean(self.reward_history[-100:]),
            "episode_actor_loss": self.actor_loss_history[episode - 1],
            "episode_critic_loss": self.critic_loss_history[episode - 1],
        }

    def train(self, num_episodes, render=False):
        try:
            for episode in tqdm(range(num_episodes)):

                state = self.env.reset()
                state = torch.tensor(state, device=self.device)

                episode_reward = 0
                episode_actor_loss = []
                episode_critic_loss = []

                done = False
                while not done:

                    if render:
                        self.env.render()

                    self.steps += 1
                    distribution = self.actor_net(state.view((1,) + self.state_shape))
                    action = distribution.sample()

                    next_state, reward, done, info = self.env.step(int(action))

                    next_state_tensor = torch.tensor(next_state, device=self.device)
                    reward_tensor = torch.tensor(reward, device=self.device)
                    done_tensor = torch.tensor(int(done), device=self.device).unsqueeze(
                        0
                    )

                    # Optimize critic
                    self.critic_optimizer.zero_grad()

                    # If using a semi-gradient method to train the critic, the target values shouldn't be used in the computation of the gradient
                    # The parameter update we're trying to get is w_t+1 <- w_t + (R_t+1 + v(S_t+1, w_t) - v(S_t, w_t))∇v(S_t, w_t)
                    with torch.no_grad():
                        next_state_value = self.critic_net(next_state_tensor)

                    next_state_value = next_state_value.detach()

                    critic_target = reward_tensor + self.gamma * next_state_value * (
                        1 - done_tensor
                    )

                    critic_prediction = self.critic_net(state)

                    critic_loss = self.loss_fn(critic_prediction, critic_target)
                    critic_loss.backward()
                    self.critic_optimizer.step()

                    # Optimize actor
                    self.actor_optimizer.zero_grad()
                    log_prob = distribution.log_prob(action).unsqueeze(0)
                    advantage = critic_target - critic_prediction
                    actor_loss = -(log_prob * advantage.detach())

                    actor_loss.backward()
                    self.actor_optimizer.step()

                    # Transition to next state
                    state = next_state_tensor
                    episode_reward += reward
                    episode_actor_loss.append(actor_loss.item())
                    episode_critic_loss.append(critic_loss.item())

                self.reward_history.append(episode_reward)
                self.actor_loss_history.append(np.mean(episode_actor_loss))
                self.critic_loss_history.append(np.mean(episode_critic_loss))
                self.episodes += 1

                self.invoke_callbacks()

        finally:
            self.close_callbacks()
