from autonomous_systems_project.callbacks.common import Callback
from tqdm import tqdm


class UpdateTqdmProgress(Callback):
    def __init__(self, **kwargs):
        super().__init__()
        self.pbar = tqdm(**kwargs)

    def __call__(self, state):
        self.pbar.update(1)

    def close(self):
        self.pbar.close()
