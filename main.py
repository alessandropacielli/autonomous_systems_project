import cv2
import gym
from gym.wrappers import FrameStack, GrayScaleObservation, TransformObservation

from autonomous_systems_project.dqn import DQN
from autonomous_systems_project.policy import epsilon_greedy
from autonomous_systems_project.preprocessing import crop, rescale
from autonomous_systems_project.training import train_dqn

rescale_fn = lambda observation: cv2.resize(
    observation, (84, 84), interpolation=cv2.INTER_AREA
)

# Number of frames for framestack
frames = 4

env = gym.make("Breakout-v0")
env = GrayScaleObservation(env)  # Black & white filter
env = TransformObservation(env, crop(31, -1, 7, -8))  # Crop image
env = TransformObservation(env, rescale((84, 84)))  # Rescale to 84 x 84
env = FrameStack(env, frames)  # Stack consecutive frames
policy_net = DQN(84, 84, 4, env.action_space.n)
try:
    train_dqn(policy_net, env, epsilon_greedy, render=True, eps_decay=10000)
finally:
    env.close()
