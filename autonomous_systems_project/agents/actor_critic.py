import math
import random

import torch
import torch.nn as nn
from tqdm import tqdm


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
        return self.fc(x)


class ActorCriticAgent:
    def __init__(
        self,
        env,
        memory,
        actor_net,
        critic_net,
        target_critic_net,
        batch_size=32,
        actor_lr=0.00025,
        critic_lr=0.00025,
        target_update=5000,
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
        self.action_space = env.action_space.n

        # Device used for training
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Double DQN
        self.actor_net = actor_net
        self.critic_net = critic_net
        self.target_critic_net = target_critic_net

        # DO NOT COMPUTE GRADIENTS FOR TARGET NET
        self.target_critic_net.eval()

        # TODO optimizer choice
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.actor_optimizer = torch.optim.Adam(
            self.actor_net.parameters(), lr=self.actor_lr
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic_net.parameters(), lr=self.critic_lr
        )

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

    def select_action(self, state):
        self.steps += 1
        if random.random() < self.exploration_rate:
            return torch.tensor(random.randrange(self.action_space))
        else:
            return torch.argmax(self.actor_net(state.to(self.device))).cpu()

    def _update_target_net(self):
        """
        Updates the target network by loading the state of the policy network
        """
        self.target_critic_net.load_state_dict(self.critic_net.state_dict())

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

        # Optimize critic
        self.critic_optimizer.zero_grad()
        expected_state_values = reward_batch + torch.mul(
            (self.gamma * self.target_critic_net(next_state_batch)), 1 - done_batch,
        )
        state_values = self.critic_net(state_batch)
        loss = self.loss_fn(state_values, expected_state_values)
        loss.backward()
        self.critic_optimizer.step()
        self._update_exploration_rate()

        # Optimize actor
        self.actor_optimizer.zero_grad()
        max_likelihood = torch.log(
            self.actor_net(state_batch).gather(1, action_batch.long())
        )
        actor_target = reward_batch + torch.mul(
            torch.mul(
                (self.gamma + self.critic_net(next_state_batch).detach()),
                max_likelihood,
            ),
            1 - done_batch,
        )

        actor_prediction = torch.mul(
            (-self.critic_net(state_batch).detach()), max_likelihood
        )
        actor_loss = self.loss_fn(actor_prediction, actor_target)
        actor_loss.backward()
        self.actor_optimizer.step()

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
                        self.env.render()

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

                for callback in self.callbacks:
                    callback(
                        {
                            "episode": self.episodes,
                            "total_steps": self.steps,
                            "epsilon": self.exploration_rate,
                            "actor_net": self.actor_net,
                            "critic_net": self.critic_net,
                            "target_critic_net": self.target_critic_net,
                            "actor_optimizer": self.actor_optimizer,
                            "critic_optimizer": self.critic_optimizer,
                            "reward_history": self.reward_history,
                            "loss_history": self.loss_history,
                        }
                    )
        finally:
            for callback in self.callbacks:
                callback.close()
