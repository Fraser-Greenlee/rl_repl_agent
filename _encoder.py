import torch
from torch import nn
from torch import optim

from _language import InputLang
from _constants import DEVICE, config


class EncoderLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(EncoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(
            input_size, hidden_size, padding_idx=InputLang.pad_char_i
        )
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=config.SEQ2SEQ_ENCODER_LAYER_COUNT, batch_first=True)
        self.optimizer = optim.SGD(self.parameters(), lr=config.SEQ2SEQ_ENCODER_LEARNING_RATE)

    def _init_hidden(self, batch_size=1):
        hidden_states = torch.zeros(config.SEQ2SEQ_ENCODER_LAYER_COUNT, batch_size, self.hidden_size, device=DEVICE)
        cell_states = torch.zeros(config.SEQ2SEQ_ENCODER_LAYER_COUNT, batch_size, self.hidden_size, device=DEVICE)
        return (hidden_states, cell_states)

    def forward(self, input_sequences, input_lengths):
        '''
            Creates an encoding for a snippet of code.

            @params:
                input_sequences:    Tensor of size (batch_size, max_seq_len)
                input_lengths:      Tensor of lengths corresponding to each sequence
                                    in `input_sequences`

            @returns:
                output:             The final hidden state (encoding) for each code snippet
        '''
        batch_size = input_sequences.size(0)
        hidden, cell = self._init_hidden(batch_size)
        
        embedded = self.embedding(input_sequences)
        padded_embedding = nn.utils.rnn.pack_padded_sequence(
            embedded, input_lengths, batch_first=True, enforce_sorted=False
        )
        _output, (hidden, cell) = self.lstm(padded_embedding, (hidden, cell))
        # TODO should I return & use cell_state in the decoder? YES
        return hidden, cell
