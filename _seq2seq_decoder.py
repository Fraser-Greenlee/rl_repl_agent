import torch

from _decoder import DecoderLSTM
from _language import OutputLang
from _constants import DEVICE


class Seq2SeqDecoder:
    '''
        Wrapper for `DecoderLSTM`
    '''
    def __init__(self, criterion, *lstm_args):
        self.model = DecoderLSTM(*lstm_args).to(DEVICE)
        self.criterion = criterion

    def _loss_starting_state(self, trackers):
        self.batch_size = self.target_tensors.size(0)
        self.input_char =            torch.ones(self.batch_size, dtype=torch.long,  device=DEVICE) * OutputLang.i_for_char['SOS']
        self.output_is_pending =     torch.ones(self.batch_size, dtype=torch.bool,  device=DEVICE)
        self.losses =               torch.zeros(self.batch_size, dtype=torch.float, device=DEVICE)
        self._run_trackers(trackers, starting_state=True)

    def _step(self):
        # need this to be long for the multiplication
        # TODO fix bug `RuntimeError: output with shape [] doesn't match the broadcast shape [1]`
        masked_input = self.input_char * self.output_is_pending.long()
        output, self.hidden_cell = self.model(masked_input, self.hidden_cell)
        self.input_char = self.char_i_from_lstm_output(output)
        self.losses += self.output_is_pending.float() * self.criterion(output, self.target_tensors[:, self.index])
        self.output_is_pending = (self.output_is_pending & (self.target_tensors[:, self.index] != OutputLang.i_for_char['EOS']))

    def loss(self, hidden_cell, target_tensors, trackers):
        '''
            Finds the loss when converting `hidden_cell` into `target_tensors`
        '''
        self.hidden_cell = hidden_cell
        self.target_tensors = target_tensors
        self._loss_starting_state(trackers)
        for i in range(OutputLang.maxlen):
            self.index = i
            self._step()
            self._run_trackers(trackers)
        return self.losses

    @staticmethod
    def char_i_from_lstm_output(output):
        _, top_char_index = output.topk(1)
        return top_char_index.squeeze().detach()  # detach from history as input

    def _run_trackers(self, trackers, **kwargs):
        for a_tracker in trackers:
            a_tracker(**kwargs)

    # TRACKERS
    def track_output(self, starting_state=False):
        if starting_state:
            self.output_tensors = torch.zeros((self.batch_size, OutputLang.maxlen), dtype=torch.long, device=DEVICE, requires_grad=False)
        else:
            self.output_tensors[:, self.index] = self.input_char

    def get_output(self):
        return self.output_tensors.clone()

    def track_accuracy(self, starting_state=False):
        if starting_state:
            self.is_correct = torch.ones(self.batch_size, dtype=torch.bool, device=DEVICE, requires_grad=False)
        else:
            self.is_correct = self.is_correct & (
                (self.input_char == self.target_tensors[:, self.index]) | (self.output_is_pending == False)
            )

    def get_accuracy(self):
        return self.is_correct.sum().item() / self.batch_size
