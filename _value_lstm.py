import torch
from torch import nn
import torch.nn.functional as F

from _language import OutputLang
from _constants import DEVICE, config


class ValueLSTM(nn.Module):
    def __init__(self, action_size):
        super(ValueLSTM, self).__init__()
        # Add 1 to input size to handle padding char.
        input_size = len(OutputLang.chars) + 1
        self.embedding = nn.Embedding(
            input_size, config.PPO_CRITIC_HIDDEN_SIZE, padding_idx=OutputLang.pad_char_i
        )
        self.lstm = nn.LSTM(config.PPO_CRITIC_HIDDEN_SIZE, config.PPO_CRITIC_HIDDEN_SIZE, num_layers=config.PPO_CRITIC_LAYER_COUNT, batch_first=True)
        self.out = nn.Linear(config.PPO_CRITIC_HIDDEN_SIZE, 1)

    @staticmethod
    def _init_hidden(batch_size):
        hidden_states = torch.zeros(config.PPO_CRITIC_LAYER_COUNT, batch_size, config.PPO_CRITIC_HIDDEN_SIZE, device=DEVICE)
        cell_states = torch.zeros(config.PPO_CRITIC_LAYER_COUNT, batch_size, config.PPO_CRITIC_HIDDEN_SIZE, device=DEVICE)
        return (hidden_states, cell_states)

    def forward(self, input_sequences, input_lengths):
        '''
            Finds the value estimate for given input sequences.
            Input sequences should be in a padded batch

            @params:
                input_sequences:    Tensor of size (batch_size, max_seq_len)
                input_lengths:      Tensor of lengths corresponding to each sequence
                                    in `input_sequences`.

            @returns:
                output:             Value estimates of each sequence.
                                    Has size (batch_size, 1)
        '''
        # TODO try adding `embedded = F.relu(embedded)`
        batch_size = input_sequences.size(0)
        hidden = self._init_hidden(batch_size)

        output = self.embedding(input_sequences)

        output = nn.utils.rnn.pack_padded_sequence(
            output, input_lengths, batch_first=True, enforce_sorted=False
        )
        output, _ = self.lstm(output, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        output = output.contiguous()
        output = output.view(-1, output.shape[2])

        output = self.out(output)
        return output.view(batch_size, max(input_lengths), 1)[range(batch_size), input_lengths-1]
