import numpy as np
from torch.utils.tensorboard import SummaryWriter

from autonomous_systems_project.callbacks.common import Callback


class LogToTensorboard(Callback):
    def __init__(self, **kwargs):
        super().__init__()
        self.writer = SummaryWriter(**kwargs)

    def __call__(self, agent):
        for key, value in agent.get_metrics().items():
            self.writer.add_scalar(key, value)
