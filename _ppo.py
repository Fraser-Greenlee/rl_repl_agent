import time
import os
import wandb
import torch
from torch import nn
from torchviz import make_dot # use: make_dot({variable}).view()

from _actor_critic import ActorCritic
from _logs_ppo import LogsPPO
from _constants import (
    DEVICE,
    config,
    MODELS_FOLDER
)


class PPO:
    '''
        Uses PPO to optimise a code writer.
    '''
    def __init__(self, indices):
        self.policy = ActorCritic(config.ACTION_DIM).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.PPO_LEARNING_RATE, betas=config.BETAS)
        self.policy_old = ActorCritic(config.ACTION_DIM).to(DEVICE)
        self.MseLoss = nn.MSELoss()

    def _monte_carlo_estimate_state_rewards(self, old_rewards, old_is_terminals):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(old_rewards), reversed(old_is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (config.GAMMA * discounted_reward)
            rewards.insert(0, discounted_reward)
        return rewards

    @staticmethod
    def _normalise(rewards):
        rewards = torch.tensor(rewards).to(DEVICE)
        return (rewards - rewards.mean()) / (rewards.std() + 1e-5)

    def _surrogate_loss(self, rewards, state_values, new_logprobs, old_logprobs, dist_entropy):
        # Find new_policy.probability( action | state ) / old_policy.probability( action | state )
        policy_prob_ratio = torch.exp(new_logprobs - old_logprobs.detach())
        advantages = rewards - state_values.detach()
        surr1 = policy_prob_ratio * advantages
        surr2 = torch.clamp(policy_prob_ratio, 1 - config.EPS_CLIP, 1 + config.EPS_CLIP) * advantages
        policy_loss = -torch.min(surr1, surr2)
        # TODO try clipping value_loss
        value_loss = self.MseLoss(state_values, rewards)

        wandb.log({'PPO.policy loss': policy_loss.mean().item()})
        wandb.log({'PPO.value loss': value_loss.mean().item()})

        return policy_loss + config.CRITIC_DISCOUNT * self.MseLoss(state_values, rewards) - config.ENTROPY_BETA * dist_entropy

    @staticmethod
    def _explained_variance(predicted_y, y):
        """
        Computes fraction of variance that predicted_y explains about y.
        Returns 1 - Var[y - predicted_y] / Var[y]
        interpretation:
            result = 0  =>  might as well have predicted zero
            result = 1  =>  perfect prediction
            result < 0  =>  worse than just predicting zero
        """
        assert y.ndim == 1 and predicted_y.ndim == 1
        if y.var() == 0:
            raise Exception('variance of y is 0')
        return (1 - (y - predicted_y).var()/y.var()).item()

    def _optimize_policy(self, states, state_lens, actions, old_logprobs, rewards, batch_i, tqdm):
        # TODO this method takes soooo long, try and improve it
        for epoch_i in range(config.K_EPOCHS):
            # Evaluating old actions and values:
            actions_logprobs, state_values, dist_entropy = self.policy.evaluate(states, state_lens, actions)

            loss = self._surrogate_loss(rewards, state_values, actions_logprobs, old_logprobs, dist_entropy)

            # take gradient step
            self.optimizer.zero_grad()
            # TODO `.backward()` takes 12 seconds, computational graph is likely too large, try and shrink it
            loss.mean().backward()
            # TODO try using `nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.max_grad_norm)`

            self.optimizer.step()

            wandb.log({'PPO.entropy': dist_entropy.mean().item()})
            wandb.log({'PPO.overall loss': loss.mean().item()})
            wandb.log({'PPO.value estimate (since last update)': state_values.mean().item()})
            wandb.log({'PPO.value function explained variance': self._explained_variance(state_values.detach(), rewards)})


    def update(self, memory, batch_i, tqdm):
        '''
            Update the policy using the `memory` of what happened since last update.
        '''
        rewards = self._monte_carlo_estimate_state_rewards(memory.rewards, memory.is_terminals)
        rewards = self._normalise(rewards)

        self._optimize_policy(memory.states, memory.state_lens, memory.actions, memory.logprobs, rewards, batch_i, tqdm)

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save(self, folder):
        folder += '_ppo/'
        os.makedirs(os.path.join(wandb.run.dir, folder), exist_ok=True)
        torch.save(
            self.policy_old.action_lstm.state_dict(),
            os.path.join(wandb.run.dir, folder, 'actor.pt')
        )
        torch.save(
            self.policy_old.value_lstm.state_dict(),
            os.path.join(wandb.run.dir, folder, 'critic.pt')
        )

    def load(self, name):
        folder = MODELS_FOLDER + name + '/'
        self.policy_old.action_lstm.load_state_dict(
            torch.load(
                os.path.join(wandb.run.dir, folder, 'actor.pt')
            )
        )
        self.policy_old.value_lstm.load_state_dict(
            torch.load(
                os.path.join(wandb.run.dir, folder, 'critic.pt')
            )
        )
