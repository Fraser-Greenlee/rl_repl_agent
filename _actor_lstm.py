import torch
from torch import nn

from _language import OutputLang
from _constants import DEVICE, config


class ActorLSTM(nn.Module):
    '''
        Reads the program state (code & stdout) and gives probabilities for each
        char to write next to most confuse the seq2seq model.
    '''
    def __init__(self, action_size):
        super(ActorLSTM, self).__init__()
        # Add 1 to input size to handle padding char.
        input_size = len(OutputLang.chars) + 1
        # When the embedding sees the padding index it'll make the embeded vector zeros.
        self.embedding = nn.Embedding(
            input_size, config.PPO_ACTOR_HIDDEN_SIZE, padding_idx=OutputLang.pad_char_i
        )
        self.lstm = nn.LSTM(config.PPO_ACTOR_HIDDEN_SIZE, config.PPO_ACTOR_HIDDEN_SIZE, num_layers=config.PPO_ACTOR_LAYER_COUNT, batch_first=True)
        self.out = nn.Linear(config.PPO_ACTOR_HIDDEN_SIZE, action_size)
        self.hidden = None
        self.softmax = nn.Softmax(dim=1)
        self.action_size = action_size

    @staticmethod
    def _init_hidden(batch_size):
        'weights are of the form (config.PPO_ACTOR_LAYER_COUNT, batch_size, config.PPO_ACTOR_HIDDEN_SIZE)'
        hidden_states = torch.zeros(config.PPO_ACTOR_LAYER_COUNT, batch_size, config.PPO_ACTOR_HIDDEN_SIZE, device=DEVICE)
        cell_states = torch.zeros(config.PPO_ACTOR_LAYER_COUNT, batch_size, config.PPO_ACTOR_HIDDEN_SIZE, device=DEVICE)
        return (hidden_states, cell_states)

    def forward(self, input_sequences, input_lengths):
        '''
            Finds the action probability for given input sequences.
            Input sequences should be in a padded batch

            @params:
                input_sequences:    Tensor of size (batch_size, max_seq_len)
                input_lengths:      Tensor of lengths corresponding to each sequence
                                    in `input_sequences`.

            @returns:
                output:             The probability of each action after reading the input sequence.
                                    Has size (batch_size, 1)
        '''
        # TODO try adding `embedded = F.relu(embedded)`
        batch_size = input_sequences.size(0)
        hidden = self._init_hidden(batch_size)
        # Dim change: (batch_size, seq_len) -> (batch_size, seq_len, embedding_dim)
        output = self.embedding(input_sequences)
        # Dim change: (batch_size, seq_len, embedding_dim) -> (batch_size, seq_len, hidden_size)
        output = nn.utils.rnn.pack_padded_sequence(
            output, input_lengths, batch_first=True, enforce_sorted=False
        )
        output, _ = self.lstm(output, hidden)
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        # Dim change: (batch_size, seq_len, hidden_size) -> (batch_size * seq_len, hidden_size)
        output = output.contiguous()
        output = output.view(-1, output.shape[2])
        # run through actual linear layer
        output = self.out(output)
        # Dim change: (batch_size * seq_len, hidden_size) -> (batch_size, seq_len, action_size)
        output = self.softmax(output)
        # only return output for cell at final char for each batch
        return output.view(batch_size, max(input_lengths), self.action_size)[range(batch_size), input_lengths-1]
