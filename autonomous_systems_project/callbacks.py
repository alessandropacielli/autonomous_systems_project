import numpy as np


def log_stdout(state):
    episode = state["episode"]
    print(
        "Episode %d/%d - total steps %d  - reward %f - avg total rewards %f - epsilon %f - loss %f"
        % (
            state["episode"],
            state["num_episodes"],
            state["total_steps"],
            state["reward_history"][episode],
            np.mean(state["reward_history"]),
            state["epsilon"],
            np.mean(state["loss_history"][episode]),
        )
    )
