import numpy as np

from autonomous_systems_project.callbacks.common import Callback


class LogToStdout(Callback):
    def __init__(self):
        super().__init__()

    def __call__(self, state):
        episode = state["episode"]
        print("Episode: %d" % (episode))
        print("Total steps: %d" % (state["total_steps"]))
        print("Episode reward: %f" % (state["reward_history"][episode - 1]))
        print("Average reward: %f" % (np.mean(state["reward_history"]))),
        print(
            "Average reward (last 100 episodes): %f"
            % (np.mean(state["reward_history"][-100:]))
        )
        print("Episode loss (avg): %f" % (np.mean(state["loss_history"][episode - 1])))
        print("Exploration rate: %f" % (state["epsilon"]))
