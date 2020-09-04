import math

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from autonomous_systems_project.memory import RandomReplayMemory, Transition
from autonomous_systems_project.policy import epsilon_greedy

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_epsilon(epsilon_start, epsilon_end, epsilon_steps, total_steps):
    return epsilon_end + (epsilon_start - epsilon_end) * math.exp(
        -1.0 * total_steps / epsilon_steps
    )


def optimize_model(
    policy_net,
    target_net,
    optimizer,
    memory,
    batch_size,
    gamma,
    frame_stack=4,
    frame_h=84,
    frame_w=84,
):
    # Sample replay buffer
    state_batch, action_batch, reward_batch, next_state_batch = memory.sample(
        batch_size
    )

    # Convert transitions to tensors
    state_batch = torch.stack(state_batch)
    state_batch = state_batch.view((batch_size,) + policy_net.input_shape)
    action_batch = torch.stack(action_batch)
    reward_batch = torch.stack(reward_batch)
    non_final_next_states = torch.cat([s for s in next_state_batch if s is not None])

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(
        list(map(lambda s: s is not None, next_state_batch)), dtype=torch.bool
    )

    # Obtain the action values for the states in the batch
    state_action_values = policy_net(state_batch)

    # Select the values of the actions that were taken, i.e. Q(S_t, A_t)
    state_action_values = state_action_values.gather(
        1, action_batch.reshape((batch_size, 1))
    )

    # Compute the max q-values for non final next states only
    next_state_values = torch.ones(batch_size, device=device)
    next_state_values[non_final_mask] = (
        target_net(non_final_next_states).max(dim=1)[0].float().detach()
    )

    # Compute the q-learning target: R_t + gamma * max_a(Q(S_t+1, a))
    expected_state_action_values = reward_batch + gamma * next_state_values

    # Compute Huber loss
    loss = F.smooth_l1_loss(
        state_action_values, expected_state_action_values.unsqueeze(1)
    )

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss


def train_dqn(
    env,
    policy_net,
    target_net,
    optimizer,
    memory,
    frame_h=84,
    frame_w=84,
    frame_stack=4,
    target_update=10,
    batch_size=32,
    episodes=100,
    gamma=0.99,
    epsilon_start=0.9,
    epsilon_end=0.05,
    epsilon_steps=1000000,
    render=False,
    callbacks=[],
):

    # TODO handle preconditions (assert)

    # Send both networks to device
    policy_net.to(device)
    target_net.to(device)

    # Do not compute gradients for target_net
    target_net.eval()

    reward_history = []
    loss_history = []

    total_steps = 0

    epsilon = epsilon_start

    for episode in range(episodes):

        done = False
        state = (
            torch.tensor(env.reset(), device=device)
            .float()
            .view((1,) + policy_net.input_shape)
        )

        reward_history.append(0)
        loss_history.append([])

        while not done:

            if render:
                env.render()

            action = epsilon_greedy(state, env, policy_net, epsilon)

            next_state, reward, done, _ = env.step(action)

            reward_history[episode] += reward

            action_tensor = torch.tensor(action, device=device, dtype=torch.int64)
            reward_tensor = torch.tensor(reward, device=device, dtype=torch.float)
            next_state = torch.tensor(
                next_state, device=device, dtype=torch.float
            ).view((1,) + policy_net.input_shape)
            if done:
                next_state = None

            memory.push(state, action_tensor, reward_tensor, next_state)

            state = next_state

            if len(memory) >= batch_size:
                loss = optimize_model(
                    policy_net, target_net, optimizer, memory, batch_size, gamma
                )
                loss_history[episode].append(loss.item())

            if total_steps % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

            total_steps += 1
            epsilon = update_epsilon(
                epsilon_start, epsilon_end, epsilon_steps, total_steps
            )

        for callback in callbacks:
            callback(
                {
                    "episode": episode,
                    "num_episodes": episodes,
                    "total_steps": total_steps,
                    "epsilon": epsilon,
                    "policy_net": policy_net,
                    "target_net": target_net,
                    "optimizer": optimizer,
                    "reward_history": reward_history,
                    "loss_history": loss_history,
                }
            )

    for callback in callbacks:
        callback.close()
