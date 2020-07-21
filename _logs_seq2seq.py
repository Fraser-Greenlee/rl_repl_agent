import torch
import wandb
from random import randint

from _logs import Logs
from _pretrain_seq2seq import random_code_output_samples
from _language import InputLang, OutputLang
from _constants import (
    config,
    DEVICE
)


class LogsSeq2seq(Logs):
    logs = {
        'Seq2seq': {
            'test': {
                'random': {
                    'loss': None,
                    'accuracy': None
                }
            },
            'loss': {
                'average': None,
                'std': None
            }
        }
    }

    def __init__(self, parent, indices):
        super().__init__(indices)
        # wandb.watch(parent.encoder)
        # wandb.watch(parent.decoder)
        self.init_log_stores()

    def init_log_stores(self):
        self.episode_losses = torch.zeros(config.LOG_EVERY_N_EPISODES, device=DEVICE)

    def log_episode_loss(self, avg_loss):
        self.episode_losses[self.indices.episode % config.LOG_EVERY_N_EPISODES] = avg_loss

    def _test_random_pairs(self, seq2seq, repl):
        code_batches, code_lengths, output_batches, output_lengths = random_code_output_samples(repl, n_batches=1)
        losses, accuracy = seq2seq.translation_loss(
            code_batches[0], code_lengths[0], output_batches[0], output_lengths[0], return_accruacy=True
        )
        return losses.sum().item(), accuracy

    def log(self, seq2seq, repl):
        '''
            Aggregate the raw data stores and save them in the logs dict.
            Also run tests and measure the accuracy of the seq2seq model.
        '''
        # TODO decide how to handle these logs
        self.logs['Seq2seq']['loss']['average'] = self.episode_losses.mean().item()
        self.logs['Seq2seq']['loss']['std'] = self.episode_losses.std().item()
        self.logs['Seq2seq']['test']['random']['loss'], self.logs['Seq2seq']['test']['random']['accuracy'] = self._test_random_pairs(seq2seq, repl)
        wandb.log(self.logs)
        self.init_log_stores()
