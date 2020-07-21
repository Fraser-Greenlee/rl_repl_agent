import wandb
import torch

from _language import InputLang, OutputLang
from _logs import Logs
from _constants import (
    config,
    DEVICE
)


class LogsPPO(Logs):
    logs = {
        'PPO': {
            'cumulative reward': None,
            'entropy': None,
            'learning rate': None,
            'policy loss': None,
            'value estimate': None,
            'value loss': None,
            'KL divergence of old from new policy': None,
            'explained variance': None,
            'seq2seq accuracy': None
        }
    }
    sample_input_output_pairs_file = 'sample_input_output_pairs.csv'
    sample_input_output_pairs = None

    def __init__(self, parent, indices):
        super().__init__(indices)
        # wandb.watch(parent.policy)
        self.init_log_stores()

    def init_log_stores(self):
        '''
            Create tensors to store raw data for aggregating into log metrics later.
            All are averaged across each batch where possible.
        '''
        if config.LOG_EVERY_N_EPISODES < config.UPDATE_EVERY_N_EPISODES:
            raise RuntimeError("Logging more frequently than updating. Won't have a loss for every log.")
        self.logged_losses = torch.zeros(config.K_EPOCHS, device=DEVICE)
        self.updates_per_log = config.LOG_EVERY_N_EPISODES // config.UPDATE_EVERY_N_EPISODES
        self.logged_rewards = torch.zeros(self.updates_per_log, device=DEVICE)
        self.logged_rewards_std = torch.zeros(self.updates_per_log, device=DEVICE)
        self.logged_seq2seq_accuracy = torch.zeros(self.updates_per_log, device=DEVICE)
        self.code_with_lens = None
        self.output_with_lens = None
        self.sample_pairs = []

    def store_memory(self, memory):
        '''
            Take a full memory instance and store it in tensors.
        '''
        if self._update_this_episode():
            self.logged_rewards[self.indices.episode % self.updates_per_log] = memory.rewards.mean()
            self.logged_rewards_std[self.indices.episode % self.updates_per_log] = memory.rewards.std()
            self.logged_seq2seq_accuracy[self.indices.episode % self.updates_per_log] = memory.seq2seq_accuracy.mean()
            self.code_with_lens = (memory.code_tensors.clone(), memory.code_lens.clone())
            self.output_with_lens = (memory.output_tensors.clone(), memory.output_lens.clone())

    def _update_this_episode(self):
        return self.indices.episode % config.UPDATE_EVERY_N_EPISODES == 0

    def store_avg_loss(self, epoch_i, avg_loss):
        if self._log_this_episode():
            self.logged_losses[epoch_i] = avg_loss

    def _sum_top_k_most_common_chars(self, lang, seq_tensors, lens):
        '''
            Measures how much seq_tensors vary.
        '''
        sums = torch.zeros(seq_tensors.size(1), device=DEVICE)
        for col_i in range(seq_tensors.size(1)):
            mask = ((1 + len(lang.chars)) * (col_i >= lens)).long()
            values, raw_counts = (seq_tensors[:, col_i] - mask).unique(return_counts=True)
            counts = raw_counts * (values >= 0).long()
            if counts.size(0) > 1:
                sums[col_i] = counts.topk(2).values.sum()
            else:
                sums[col_i] = counts[0]
        return sums.sum().item()

    def _get_samples(self):
        sample_indices = (
            self.code_with_lens[0].size(0) * torch.rand(config.NUM_OF_SAMPLES_PER_LOG, device=DEVICE)
        ).int()
        samples = []
        for i in range(config.NUM_OF_SAMPLES_PER_LOG):
            samples.append([
                InputLang.tensor_to_str(
                    self.code_with_lens[0][sample_indices[i]],
                    self.code_with_lens[1][sample_indices[i]]
                ),
                OutputLang.tensor_to_str(
                    self.output_with_lens[0][sample_indices[i]]
                )
            ])
        return samples

    def log(self):
        '''
            Aggregate the raw data stores and save them in the logs dict.
        '''
        self.logs['PPO']['loss'] = self.logged_losses.mean().cpu()
        self.logs['PPO']['seq2seq accuracy'] = self.logged_seq2seq_accuracy.mean().cpu()
        self.logs['PPO']['reward']['average'] = self.logged_rewards.mean().cpu()
        self.logs['PPO']['reward']['std'] = self.logged_rewards_std.mean().cpu()
        self.logs['PPO']['variety']['code']['sum top K most common chars'] = self._sum_top_k_most_common_chars(InputLang, *self.code_with_lens)
        self.logs['PPO']['variety']['output']['sum top K most common chars'] = self._sum_top_k_most_common_chars(OutputLang, *self.output_with_lens)
        self.logs['PPO']['sample input output pairs'] = self._get_samples()
        wandb.log(self.logs)
        self.init_log_stores()
