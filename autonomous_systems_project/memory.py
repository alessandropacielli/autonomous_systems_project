import numpy as np
import torch


class RandomReplayMemory:
    def __init__(self, capacity, state_shape):
        self.capacity = capacity

        self.states = torch.zeros(capacity, *state_shape)
        self.actions = torch.zeros(capacity, 1)
        self.rewards = torch.zeros(capacity, 1)
        self.next_states = torch.zeros(capacity, *state_shape)
        self.done = torch.zeros(capacity, 1)
        self.position = 0
        self.length = 0

    def push(self, state, action, reward, next_state, done):
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.done[self.position] = done

        self.position = (self.position + 1) % self.capacity
        self.length = min(self.length + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(range(self.length), size=batch_size)
        state_sample = self.states[indices]
        action_sample = self.actions[indices]
        reward_sample = self.rewards[indices]
        next_state_sample = self.next_states[indices]
        done_sample = self.done[indices]

        return (
            state_sample,
            action_sample,
            reward_sample,
            next_state_sample,
            done_sample,
        )

    def __len__(self):
        return self.length
