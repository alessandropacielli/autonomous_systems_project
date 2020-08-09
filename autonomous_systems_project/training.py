import math

import gym
import torch
import torch.nn.functional as F
import torch.optim as optim

from autonomous_systems_project.dqn import DQN
from autonomous_systems_project.memory import RandomReplayMemory, Transition
from autonomous_systems_project.policy import epsilon_greedy

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def update_epsilon(
    eps_curr: float,
    eps_start: float,
    eps_end: float,
    eps_decay: float,
    total_steps: int,
):
    return eps_end + (eps_start - eps_end) * math.exp(-1.0 * total_steps / eps_decay)


def train_dqn(
    policy_net: DQN,
    env: gym.Env,
    collection_policy=epsilon_greedy,
    optimizer_fn: optim.Optimizer = optim.RMSprop,
    num_episodes: int = 100,
    render: bool = False,
    memory=RandomReplayMemory(10000),
    batch_size: int = 128,
    gamma: float = 0.999,
    eps_start: float = 0.9,
    eps_end: float = 0.05,
    eps_decay: int = 200,
    target_update: int = 10,
    target_net: DQN = None,
):
    """
    Trains a DQN model
    """

    # TODO handle preconditions (assert)

    # If no target network is given, build a copy of the policy network and load its weights
    if target_net is None:
        target_net = DQN(
            policy_net.input_height,
            policy_net.input_width,
            policy_net.input_channels,
            policy_net.outputs,
        )
        target_net.load_state_dict(policy_net.state_dict())

    # Send both networks to device
    policy_net.to(device)
    target_net.to(device)

    # Target net should be in eval mode (no grads are computed)
    target_net.eval()

    # Create optimizer for policy net
    optimizer = optimizer_fn(policy_net.parameters())

    # Count steps for exploration rate updates
    total_steps = 0

    # Init exploration rate
    eps_current = eps_start

    for episode in range(num_episodes):

        # Reset environment
        state = torch.tensor(env.reset(), device=device, dtype=torch.float)
        done = False

        while not done:

            # Update epsilon
            eps_current = update_epsilon(
                eps_current, eps_start, eps_end, eps_decay, total_steps
            )

            # Increment steps
            total_steps += 1

            # Choose next action
            action = collection_policy(
                state.reshape(
                    1,
                    policy_net.input_channels,
                    policy_net.input_height,
                    policy_net.input_width,
                ),
                env,
                policy_net,
                eps_current,
            )

            # Perform transition
            next_state, reward, done, _ = env.step(action)

            # Turn into torch tensors
            action = torch.tensor([action], device=device, dtype=torch.long)
            reward = torch.tensor([reward], device=device, dtype=torch.float)
            if not done:
                next_state = torch.tensor(next_state, device=device, dtype=torch.float)

                # Move to next state
                state = next_state
            else:
                next_state = None

            # Remember transition
            memory.push(state, action, reward, next_state)

            # Optimize model
            if len(memory) >= batch_size:

                # Sample replay buffer
                transitions = memory.sample(batch_size)

                # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
                # detailed explanation). This converts batch-array of Transitions
                # to Transition of batch-arrays.
                batch = Transition(*zip(*transitions))

                # Compute a mask of non-final states and concatenate the batch elements
                # (a final state would've been the one after which simulation ended)
                non_final_mask = torch.tensor(
                    tuple(map(lambda s: s is not None, batch.next_state)),
                    device=device,
                    dtype=torch.bool,
                )
                non_final_next_states = torch.stack(
                    [s for s in batch.next_state if s is not None]
                )

                state_batch = torch.stack(batch.state)
                action_batch = torch.stack(batch.action)
                reward_batch = torch.cat(batch.reward)

                # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
                # columns of actions taken. These are the actions which would've been taken
                # for each batch state according to policy_net
                state_action_values = policy_net(state_batch).gather(1, action_batch)

                # Compute V(s_{t+1}) for all next states.
                # Expected values of actions for non_final_next_states are computed based
                # on the "older" target_net; selecting their best reward with max(1)[0].
                # This is merged based on the mask, such that we'll have either the expected
                # state value or 0 in case the state was final.
                next_state_values = torch.zeros(batch_size, device=device)
                next_state_values[non_final_mask] = (
                    target_net(non_final_next_states).max(1)[0].detach()
                )
                # Compute the expected Q values
                expected_state_action_values = (
                    next_state_values * gamma
                ) + reward_batch

                # Compute Huber loss
                loss = F.smooth_l1_loss(
                    state_action_values, expected_state_action_values.unsqueeze(1)
                )

                # Optimize the model
                optimizer.zero_grad()
                loss.backward()
                for param in policy_net.parameters():
                    param.grad.data.clamp_(-1, 1)
                optimizer.step()

        # Update target
        if episode % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())
