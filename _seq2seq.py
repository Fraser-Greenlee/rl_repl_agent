import os
import wandb
import torch
from torch import nn
from torch.nn import functional as F
from torchviz import make_dot # use doing `make_dot({variable}).view()`

from _logs_seq2seq import LogsSeq2seq
from _language import InputLang, OutputLang
from _transformer import TransformerModel
from _constants import (
    DEVICE,
    MODELS_FOLDER,
    config
)


class Seq2Seq:
    def __init__(self, indices, load_from=None):
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        self.volcab_size = len(OutputLang.chars) + 1 # +1 for padding character
        self.transformer  = TransformerModel(
            self.volcab_size, config.SEQ2SEQ_EMBEDDING_SIZE, config.SEQ2SEQ_NUM_HEADS, config.SEQ2SEQ_HIDDEN_SIZE,
            config.SEQ2SEQ_NUM_LAYERS, config.SEQ2SEQ_DROPOUT
        ).to(DEVICE)
        self.optimizer = torch.optim.SGD(self.transformer.parameters(), lr=config.SEQ2SEQ_LEARNING_RATE)

        if load_from:
            raise NotImplementedError()

    @staticmethod
    def _char_i_from_softmax_output(output):
        _, top_char_index = output.topk(1)
        return top_char_index.squeeze().detach()  # detach from history as input

    def _find_accuracy(self, output, target_sequences, mask):
        output_chars = self._char_i_from_softmax_output(output)
        num_right = (mask * (output_chars == target_sequences)).sum().item()
        num_samples = mask.sum().item()
        return float(num_right)/num_samples

    def _find_loss(self, output, target_sequences, mask):
        # TODO does using `.sum()` OR `.mean()` make a difference?
        return (mask.view(-1) * self.criterion( output.view(-1, self.volcab_size), target_sequences.reshape(-1) )).mean()

    def next_char_loss(self, input_sequences, target_sequences, mask):
        '''
            Finds the loss of converting `input_sequences` to 1 character adjacent `target_sequences`
            e.g
                input: '2**9=51' target: '**9=512' (masked: 'MMMM512')
        '''
        # TODO use mask to give 0 loss on code chars
        output = self.transformer(input_sequences)
        loss = self._find_loss(output, target_sequences, mask).item()
        return loss, self._find_accuracy(output, target_sequences, mask)

    def train_step(self, input_sequences, target_sequences, mask):
        self.optimizer.zero_grad()
        output = self.transformer(input_sequences)
        loss = self._find_loss(output, target_sequences, mask)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), config.SEQ2SEQ_MAX_GRADIENT_NORM)
        self.optimizer.step()
        return loss.item(), self._find_accuracy(output, target_sequences, mask)

    def save(self, folder):
        folder += '_seq2seq/'
        os.makedirs(os.path.join(wandb.run.dir, folder), exist_ok=True)
        torch.save(
            self.transformer.state_dict(),
            os.path.join(wandb.run.dir, folder, 'transformer.pt')
        )

    def load(self, name):
        folder = MODELS_FOLDER + name + '/'
        self.transformer.load_state_dict(
            torch.load(
                os.path.join(wandb.run.dir, folder, 'transformer.pt')
            )
        )
