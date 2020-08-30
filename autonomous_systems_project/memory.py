import random
from collections import namedtuple

import torch

Transition = namedtuple("Transition", ("state", "action", "reward", "next_state"))


class RandomReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        return Transition(*zip(*random.sample(self.memory, batch_size)))

    def __len__(self):
        return len(self.memory)
