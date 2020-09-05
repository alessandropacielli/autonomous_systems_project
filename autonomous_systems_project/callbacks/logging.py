import numpy as np

from autonomous_systems_project.callbacks.common import Callback


class LogToStdout(Callback):
    def __init__(self):
        super().__init__()

    def __call__(self, state):
        episode = state["episode"]
        print(
            "Episode %d/%d - total steps %d  - reward %f - avg total rewards %f - epsilon %f - loss %f"
            % (
                state["episode"] + 1,
                state["num_episodes"],
                state["total_steps"],
                state["reward_history"][episode],
                np.mean(state["reward_history"]),
                state["epsilon"],
                np.mean(state["loss_history"][episode]),
            )
        )
