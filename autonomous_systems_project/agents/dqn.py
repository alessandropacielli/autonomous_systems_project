import math
import random

import numpy as np
import torch
import torch.nn as nn
from autonomous_systems_project.agents.common import Agent
from tqdm import tqdm


class SimpleDQN(nn.Module):
    def __init__(self, num_inputs: int, num_actions: int):
        """
        Constructs a DQN for problems with 1-dimensional inputs.
        """
        super(SimpleDQN, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(num_inputs, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 64),
            nn.LeakyReLU(),
            nn.Linear(64, num_actions),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.fc(x)


class AtariDQN(nn.Module):
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
            nn.Linear(conv_out_size, 512), nn.ReLU(), nn.Linear(512, num_actions)
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


class DQNAgent(Agent):
    def __init__(
        self,
        env,
        memory,
        policy_net,
        target_net,
        batch_size=32,
        lr=0.00025,
        target_update=5000,
        gamma=0.9,
        exploration_max=0.9,
        exploration_min=0.02,
        exploration_steps=1000000,
        callbacks=[],
    ):
        super().__init__()

        # Env
        self.env = env
        self.state_shape = env.observation_space.shape
        self.action_space = env.action_space.n

        # Device used for training
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Double DQN
        self.policy_net = policy_net.to(self.device)
        self.target_net = target_net.to(self.device)

        # DO NOT COMPUTE GRADIENTS FOR TARGET NET
        self.target_net.eval()

        # TODO optimizer choice
        self.lr = lr
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=self.lr)

        # Memory can be chose
        self.memory = memory

        # Training parameters
        self.gamma = gamma
        self.loss_fn = nn.SmoothL1Loss().to(
            self.device
        )  # Also known as Huber loss <-- TODO loss choice
        self.exploration_max = exploration_max
        self.exploration_rate = exploration_max
        self.exploration_min = exploration_min
        self.exploration_steps = exploration_steps
        self.target_update = target_update
        self.batch_size = batch_size

        # Counters and stats
        self.steps = 0
        self.episodes = 0
        self.reward_history = []
        self.loss_history = []

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
            "episode_loss": np.mean(self.loss_history[episode - 1]),
            "exploration_rate": self.exploration_rate,
        }

    def select_action(self, state):
        self.steps += 1
        if random.random() < self.exploration_rate:
            return torch.tensor(random.randrange(self.action_space))
        else:
            return torch.argmax(self.policy_net(state.to(self.device))).cpu()

    def _update_target_net(self):
        """
        Updates the target network by loading the state of the policy network
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def _update_exploration_rate(self):
        self.exploration_rate = self.exploration_min + (
            self.exploration_max - self.exploration_min
        ) * math.exp(-1.0 * self.steps / self.exploration_steps)

    def optimize_model(self):
        # Update target DQN every `target_update` steps
        if self.steps % self.target_update == 0:
            self._update_target_net()

        # Abort if replay buffer isn't full enough
        if self.batch_size > len(self.memory):
            return

        # Sample from replay memory
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = self.memory.sample(self.batch_size)

        # Send to GPU
        state_batch = state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        next_state_batch = next_state_batch.to(self.device)
        done_batch = done_batch.to(self.device)

        # Zero optimizer gradients
        self.optimizer.zero_grad()

        # Compute target state-action values
        expected_state_action_values = reward_batch + torch.mul(
            (self.gamma * self.target_net(next_state_batch).max(1).values.unsqueeze(1)),
            1 - done_batch,
        )

        # Compute state-action values from policy_net
        state_action_values = self.policy_net(state_batch).gather(
            1, action_batch.long()
        )

        # Compute loss
        loss = self.loss_fn(state_action_values, expected_state_action_values)

        # Optimize model
        loss.backward()
        self.optimizer.step()

        # Decay exploration rate
        self._update_exploration_rate()

        return loss.item()

    def train(self, num_episodes, render=False):
        try:
            for episode in tqdm(range(num_episodes)):

                state = self.env.reset()
                state = torch.tensor(state)

                episode_reward = 0
                episode_loss = []

                done = False
                while not done:

                    if render:
                        self.env.render(mode="ipython")

                    self.steps += 1
                    action = self.select_action(state.view((1,) + self.state_shape))

                    next_state, reward, done, info = self.env.step(int(action))

                    next_state_tensor = torch.tensor(next_state)
                    reward_tensor = torch.tensor(reward)
                    done_tensor = torch.tensor(int(done)).unsqueeze(0)

                    self.memory.push(
                        state, action, reward_tensor, next_state_tensor, done_tensor
                    )

                    loss = self.optimize_model()

                    if loss is not None:
                        episode_loss.append(loss)

                    episode_reward += reward

                    # Transition to next state
                    state = next_state_tensor

                self.reward_history.append(episode_reward)
                self.loss_history.append(episode_loss)
                self.episodes += 1

                self.invoke_callbacks()

        finally:
            self.close_callbacks()

    
class IllegalActionsDQNAgent(DQNAgent):
    
    def select_action(self, state):
        self.steps += 1
        if random.random() < self.exploration_rate:
            action = torch.tensor(np.random.choice(self.env.legal_actions(), size=1)).float()
        else:
            q_values = self.policy_net(state.float().to(self.device))
            # print("Q_values = " + str(q_values[0]))
            illegal_indices = list(set(range(self.action_space)) - set(self.env.legal_actions()))
            # print("Legal actions " + str(self.env.legal_actions()))
            # print("Illegal indices" + str(illegal_indices))

            q_values[0][illegal_indices] = float("-Inf")
            # print("Q_values after filtering = " + str(q_values))

            action = torch.tensor(torch.argmax(q_values).float().cpu())

        
        # print("Agent selected action {}".format(action))
        return action
