import gym
import numpy as np
from kaggle_environments import make

import autonomous_systems_project.callbacks as cb
from autonomous_systems_project.agents import IllegalActionsDQNAgent, SimpleDQN
from autonomous_systems_project.memory import RandomReplayMemory


class TicTacToe(gym.Env):
    def __init__(self):
        self.env = make("tictactoe", debug=True)
        self.trainer = self.env.train([None, "reaction"])

        # Define required gym fields (examples):
        self.action_space = gym.spaces.Discrete(9)
        self.observation_space = gym.spaces.Discrete(9)

    def step(self, action):
        next_state, reward, done, info = self.trainer.step(action)
        # print(str(next_state))
        # print("Done: " + str(done))
        return next_state.board, reward, done, info

    def reset(self):
        return self.trainer.reset().board

    def render(self, **kwargs):
        return self.env.render(**kwargs)

    def legal_actions(self):
        actions = np.where(np.array(ttt.env.state[0].observation.board) == 0)[0]
        return actions


ttt = TicTacToe()
state_shape = ttt.observation_space.n
n_actions = ttt.action_space.n

memory = RandomReplayMemory(10000, (state_shape,))

policy_net = SimpleDQN(state_shape, n_actions)
target_net = SimpleDQN(state_shape, n_actions)
target_net.load_state_dict(policy_net.state_dict())


parameters = {
    "gamma": 0.9,
    "batch_size": 128,
    "exploration_steps": 1000,
    "lr": 0.01,
    "target_update": 5,
}
callbacks = [
    cb.LogToStdout(),
    cb.LogToMLFlow(parameters, run_name="DQN - TicTacToe"),
]

agent = IllegalActionsDQNAgent(
    ttt, memory, policy_net, target_net, callbacks=callbacks, **parameters
)

# TODO fix DQNAgent (remove env dependency)
agent.state_shape = (state_shape,)

agent.train(10000)
