import torch

from autonomous_systems_project.agents.dqn import DQNAgent


class DoubleDQNAgent(DQNAgent):
    def optimize_model(self):
        # Update target DQN every `target_update` steps
        if self.steps % self.target_update == 0:
            self._update_target_net()

        # Abort if replay buffer isn't full enough
        if self.batch_size > len(self.memory):
            return

        # Sample from replay memory
        (
            state_batch,
            action_batch,
            reward_batch,
            next_state_batch,
            done_batch,
        ) = self.memory.sample(self.batch_size)

        # Zero optimizer gradients
        self.optimizer.zero_grad()

        # Compute target state-action values
        current_best_actions = (
            self.policy_net(next_state_batch).argmax(dim=1).unsqueeze(1)
        )
        target_state_action_values = self.target_net(next_state_batch).gather(
            1, current_best_actions.long()
        )
        expected_state_action_values = reward_batch + torch.mul(
            (self.gamma * target_state_action_values), 1 - done_batch,
        )

        # Compute state-action values from policy_net
        state_action_values = self.policy_net(state_batch).gather(
            1, action_batch.long()
        )

        # Compute loss
        loss = self.loss_fn(state_action_values, expected_state_action_values)

        # Optimize model
        loss.backward()
        self.optimizer.step()

        # Decay exploration rate
        self._update_exploration_rate()

        return loss.item()
