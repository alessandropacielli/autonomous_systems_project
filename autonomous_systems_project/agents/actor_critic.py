import math
import random

import torch
import torch.nn as nn
from torch.distributions import Categorical
from tqdm import tqdm


class AtariCriticNetwork(nn.Module):
    def __init__(self, input_shape):
        super(AtariDQN, self).__init__()
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
        super(AtariDQN, self).__init__()
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
            nn.Linear(num_inputs, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, num_actions),
            nn.Softmax(),
        )

    def forward(self, x):
        distribution = Categorical(self.fc(x))
        return distribution


class ActorCriticAgent:
    def __init__(
        self,
        env,
        actor_net,
        critic_net,
        actor_lr=0.00025,
        critic_lr=0.00025,
        gamma=0.9,
        exploration_max=0.9,
        exploration_min=0.002,
        exploration_steps=1000000,
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
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_steps = exploration_steps

        # Counters and stats
        self.steps = 0
        self.episodes = 0
        self.reward_history = []

        # Callbacks
        self.callbacks = callbacks

    def train(self, num_episodes, render=False):
        try:
            for episode in tqdm(range(num_episodes)):

                state = self.env.reset()
                state = torch.tensor(state, device=self.device)

                episode_reward = 0

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
                    next_state_value = self.critic_net(next_state_tensor)

                    # If using a semi-gradient method to train the critic, the target values shouldn't be used in the computation of the gradient
                    # The parameter update we're trying to achieve is w_t+1 <- w_t + (R_t+1 + v(S_t+1, w_t) - v(S_t, w_t))∇v(S_t, w_t)
                    next_state_value.detach()
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

                self.reward_history.append(episode_reward)
                self.episodes += 1

                for callback in self.callbacks:
                    callback(
                        {
                            "episode": self.episodes,
                            "total_steps": self.steps,
                            "actor_net": self.actor_net,
                            "critic_net": self.critic_net,
                            "actor_optimizer": self.actor_optimizer,
                            "critic_optimizer": self.critic_optimizer,
                            "reward_history": self.reward_history,
                        }
                    )
        finally:
            for callback in self.callbacks:
                callback.close()
