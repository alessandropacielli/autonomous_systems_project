import numpy as np

import mlflow
from autonomous_systems_project.callbacks.common import Callback


class LogToMLFlow(Callback):
    def __init__(self, tracking_uri, experiment, parameters: dict):
        mlflow.start_run()
        # mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)

        mlflow.log_params(parameters)

    def __call__(self, state):
        episode = state["episode"]
        mlflow.log_metric("Reward", state["reward_history"][episode], episode)
        mlflow.log_metric("Reward (avg)", np.mean(state["reward_history"]), episode)
        mlflow.log_metric(
            "Reward avg (last 100)", np.mean(state["reward_history"][-100:]), episode
        )
        if len(state["loss_history"][episode]) > 0:
            mlflow.log_metric(
                "Loss (avg)", np.mean(state["loss_history"][episode]), episode
            )
            mlflow.log_metric(
                "Max loss", np.max(state["loss_history"][episode]), episode
            )

    def close(self):
        mlflow.end_run()
