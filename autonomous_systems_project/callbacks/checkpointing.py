from autonomous_systems_project.callbacks.common import Callback


class CheckpointModel(Callback):

    def __init__(self, checkpoint_every):
        super().__init__()
        self.checkpoint_every = checkpoint_every

    def __call__(self, state):
        

