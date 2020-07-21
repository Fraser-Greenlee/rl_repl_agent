from collections import namedtuple
import sys
import string
import torch
import numpy as np
from six import StringIO

from _constants import DEVICE, config
from _language import InputLang, OutputLang


class ReplEnv():
    '''
        The REPL enviroment our agent works in.

        At each step the agent enters a char and gets a new output.
        Runs multiple batches at the same time.
    '''
    act_code_map = [c for c in string.digits + '+-*/']

    def __init__(self, batch_size):
        self.metadata = {'render.modes': ['human', 'ansi']}
        self.batch_size = batch_size
        self.code = [''] * batch_size
        self.output = [''] * batch_size
        self.is_done = torch.zeros(batch_size, dtype=torch.bool, device=DEVICE)

    @staticmethod
    def run_code(code):
        '''
            Runs code & returns the output
        '''
        try:
            output = str(eval(code, None, None))
        except Exception:
            output = 'E'
        return output[:OutputLang.maxlen]

    def encoded_state(self, code=None, output=None):
        '''
            Encodes the current state (code-output-) encoded and with padding.
            returns:
                encoded state with padding
                length of the state
        '''
        code = self.code if code is None else code
        output = self.output if output is None else output
        # TODO refactor this
        states = OutputLang.pad_char_i * torch.ones(
            (self.batch_size, config.MAX_STATE_LEN), dtype=torch.long, device=DEVICE
        )
        state_lens = torch.zeros(self.batch_size, dtype=torch.long, device=DEVICE)
        code_lens = torch.zeros(self.batch_size, dtype=torch.long, device=DEVICE)
        for state_i in range(state_lens.size(0)):
            state_chars = list(code[state_i]) + ['SOS'] + list(output[state_i]) + ['EOS']
            for char_i, char in enumerate(state_chars):
                states[state_i, char_i] = OutputLang.i_for_char[char]
            state_lens[state_i] = char_i+1
            code_lens[state_i] = len(code[state_i])
        # TODO check state_lens is correct
        return states, state_lens, code_lens

    def _encode_a_state(self, lang, str_source):
        tensor = torch.zeros((self.batch_size, lang.maxlen), dtype=torch.long, device=DEVICE)
        lens = torch.zeros(self.batch_size, dtype=torch.long, device=DEVICE)
        for i in range(self.batch_size):
            tensor[i], lens[i] = lang.str_to_padded_tensor(str_source[i])
        return tensor, lens

    def encoded_code(self):
        return self._encode_a_state(InputLang, self.code)

    def encoded_output(self):
        return self._encode_a_state(OutputLang, self.output)

    @staticmethod
    def _add_action_to_code(code, action):
        if len(code) >= config.MAX_INPUT_OUTPUT_LENGTH:
            return ''
        return code + action

    def _update_state(self, state_i, action):
        self.code[state_i] = self._add_action_to_code(self.code[state_i], action)
        self.output[state_i] = self.run_code(self.code[state_i])

    def step(self, action_indices):
        '''
            Take 1 step in the REPL enviroment across all batches.

            @returns
                tensor: new encoded state
                tensor: boolean tensor showing which episodes are done
        '''
        for state_i in range(self.batch_size):
            if self.is_done[state_i]:
                continue
            action = self.act_code_map[action_indices[state_i]]
            self._update_state(state_i, action)
        return self.encoded_state()

    def render(self, mode='human', close=False):
        '''
            Renders state at 0 index in the batch.
        '''
        raise NotImplementedError()
        if mode not in self.metadata['render.modes']:
            raise RuntimeError('Render mode %s not a valid.' % mode)
        if close:
            return None

        outfile = StringIO() if mode == "ansi" else sys.stdout
        outfile.write('>>> {}\n{}\n'.format(self.code[0], self.output[0]))
        return outfile

    def reset(self):
        '''
            Reset the enviroment and return an encoded empty state.
        '''
        self.code = [''] * self.batch_size
        self.output = [''] * self.batch_size
        return self.encoded_state()
