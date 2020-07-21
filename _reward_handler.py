import torch

from _constants import config, DEVICE


class RewardHandler:
    def __init__(self, indices):
        self.indices = indices
        self.previous_avg_losses = torch.zeros((config.MAX_EPISODES_TIMESTEPS, config.N_PREVIOUS_LOSSES), device=DEVICE, dtype=torch.float)
        self.curr_prev_loss_index = torch.zeros(config.N_PREVIOUS_LOSSES, device=DEVICE, dtype=torch.int)
        self.no_previous_avg_losses = torch.zeros(config.MAX_EPISODES_TIMESTEPS, device=DEVICE, dtype=torch.bool)

    def _find_rewards(self, loss, code_tensors):
        rewards = (loss -  self.previous_avg_losses[self.indices.timestep].mean()).detach().clamp(-1.0, 1.0)
        rewards[loss < config.MIN_PPO_LOSS] = -1.0
        _unique_tensors, code_tensors_to_duplicate_counts, duplicate_counts = code_tensors.unique(dim=0, return_counts=True, return_inverse=True)
        duplicate_subtraction = duplicate_counts / float(code_tensors.size(0))
        return rewards - duplicate_subtraction[code_tensors_to_duplicate_counts]

    def _add_to_previous_avg_loss(self, avg_loss):
        # TODO bug broke this, maybe it should be left?
        self.previous_avg_losses[self.indices.timestep, self.curr_prev_loss_index[self.indices.timestep]] = avg_loss
        self.curr_prev_loss_index[self.indices.timestep] += 1
        if self.curr_prev_loss_index[self.indices.timestep] >= config.N_PREVIOUS_LOSSES:
            self.curr_prev_loss_index[self.indices.timestep] = 0

    def _initial_loss(self, loss):
        if self.no_previous_avg_losses[self.indices.timestep]:
            self.previous_avg_losses[self.indices.timestep] = loss
            self.no_previous_avg_losses[self.indices.timestep] = True

    def reward_from_loss(self, loss, code_tensors):
        '''
            Get PPO reward for a given seq2seq loss on code_tensors
        '''
        self._initial_loss(loss)
        rewards = self._find_rewards(loss, code_tensors)
        self._add_to_previous_avg_loss(loss)
        return rewards
