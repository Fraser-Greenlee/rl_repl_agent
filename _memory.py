import torch

from _memory_interface import MemoryInterface
from _language import OutputLang
from _constants import config, DEVICE


class Memory(MemoryInterface):
    '''
        Records what happened for PPO.
    '''
    rows = {
        'states',
        'code_lens',
        'state_lens',
        'actions',
        'rewards',
        'logprobs',
        'seq2seq_train_accuracy',
        'seq2seq_loss',
        'is_terminals'
    }
    max_state_len = config.MAX_STATE_LEN

    def _init_value_stores(self):
        self.states = self._padded_sequence(OutputLang.pad_char_i, self.max_state_len)
        self.code_lens = self._seq_len_holder()
        self.state_lens = self._seq_len_holder()
        self.actions = torch.zeros(self.update_row_count, device=DEVICE)
        self.rewards = torch.zeros(self.update_row_count, device=DEVICE)
        self.logprobs = torch.zeros(self.update_row_count, device=DEVICE)
        self.seq2seq_train_accuracy = torch.zeros(self.update_row_count, device=DEVICE)
        self.seq2seq_loss = torch.zeros(self.update_row_count, device=DEVICE)
        self.is_terminals = torch.zeros(self.update_row_count, device=DEVICE).byte()
