import numpy as np

from autonomous_systems_project.callbacks.common import Callback


class LogToStdout(Callback):
    def __init__(self):
        super().__init__()

    def __call__(self, agent):
        for key, value in agent.get_metrics().items():
            print("%s: %f" % (key, value))
