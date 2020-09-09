import numpy as np

import mlflow
from autonomous_systems_project.callbacks.common import Callback


class LogToMLFlow(Callback):
    def __init__(self, parameters: dict, tracking_uri=None, run_name=None):
        if tracking_uri is not None:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.start_run(run_name=run_name)

        mlflow.log_params(parameters)

    def __call__(self, agent):
        for key, value in agent.get_metrics().items():
            mlflow.log_metric(key, value)

    def close(self):
        mlflow.end_run()
