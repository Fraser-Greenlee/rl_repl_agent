import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from _language import OutputLang
from _constants import DEVICE, config


class DecoderLSTM(nn.Module):
    def __init__(self, hidden_size, output_size, max_length=OutputLang.maxlen):
        super(DecoderLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.embedding = nn.Embedding(output_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=config.SEQ2SEQ_DECODER_LAYER_COUNT, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)
        self.optimizer = optim.SGD(self.parameters(), lr=config.SEQ2SEQ_DECODER_LEARNING_RATE)

    def _init_hidden(self, batch_size=1):
        hidden_states = torch.zeros(config.SEQ2SEQ_DECODER_LAYER_COUNT, batch_size, self.hidden_size, device=DEVICE)
        cell_states = torch.zeros(config.SEQ2SEQ_DECODER_LAYER_COUNT, batch_size, self.hidden_size, device=DEVICE)
        return (hidden_states, cell_states)

    def forward(self, string_starts, hidden_cell):
        '''
            Write a single char for each row in a batch of code outputs given hidden states and single input chars.

            @params:
                string_starts:  Tensor of single char inputs (batch_size, 1)
                hidden:         Decoders hidden state, (hidden, cell) starting with the encoders (batch_size, self.hidden_size)

            @returns:
                output:         Scores for each char to be a single char for each output in the batch (batch_size, self.output_size)
        '''
        batch_size = string_starts.size(0)
        embedded = self.embedding(string_starts)
        # convert shape to (batch_size, seq_len, hidden_size)
        embedded = embedded.view(batch_size, 1, self.hidden_size)
        output, (hidden, cell) = self.lstm(embedded, hidden_cell)
        output = output.contiguous()
        output = output.view(-1, output.shape[2])
        output = self.out(output)
        output = self.softmax(output)
        return output, (hidden, cell)
