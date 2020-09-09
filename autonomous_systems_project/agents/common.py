class Agent:
    def invoke_callbacks(self):
        for callback in self.callbacks:
            callback(self)

    def close_callbacks(self):
        for callback in self.callbacks:
            callback.close()
