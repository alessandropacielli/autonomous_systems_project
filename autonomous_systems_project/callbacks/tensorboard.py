import numpy as np
from torch.utils.tensorboard import SummaryWriter

from autonomous_systems_project.callbacks.common import Callback


class LogToTensorboard(Callback):
    def __init__(self, **kwargs):
        super().__init__()
        self.writer = SummaryWriter(**kwargs)

    def __call__(self, state):
        episode = state["episode"]
        self.writer.add_scalar("Reward", state["reward_history"][episode], episode)
        self.writer.add_scalar(
            "Reward (avg)", np.mean(state["reward_history"]), episode
        )
        self.writer.add_scalar(
            "Reward (avg, last 100)", np.mean(state["reward_history"][-100:]), episode
        )
        if len(state["loss_history"][episode]) > 0:
            self.writer.add_scalar(
                "Loss", np.mean(state["loss_history"][episode]), episode
            )
            self.writer.add_scalar(
                "Loss (max)", np.max(state["loss_history"][episode]), episode
            )
