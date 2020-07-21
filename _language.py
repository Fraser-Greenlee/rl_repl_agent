import torch
import string

from _constants import DEVICE, config


# NOTE: Currently InputLang'input_str chars MUST be a subset of OutputLang'input_str chars.

INPUT_CHARS = [c for c in string.digits + '+-*/'] + ['EOS']
OUTPUT_CHARS = INPUT_CHARS + ['SOS'] + [c for c in 'E.infenan']

class InputLang:
    maxlen = config.MAX_INPUT_OUTPUT_LENGTH
    chars = INPUT_CHARS
    char_for_i = dict(enumerate(chars))
    i_for_char = {v: k for k, v in char_for_i.items()}
    pad_char_i = len(chars)

    @staticmethod
    def str_to_tensor(input_str, add_eos=True, pad=False):
        # TODO add sos?
        if pad:
            return InputLang.str_to_padded_tensor(input_str, add_eos=add_eos)
        return torch.LongTensor(
            [InputLang.i_for_char[char] for char in list(input_str) + (['EOS'] if add_eos else [])],
            device=DEVICE
        )

    @staticmethod
    def str_to_padded_tensor(input_str, add_eos=True):
        pad = InputLang.pad_char_i * torch.ones(InputLang.maxlen, dtype=torch.long, device=DEVICE)
        for i, char in enumerate(input_str):
            pad[i] = InputLang.i_for_char[char]
        if add_eos:
            pad[min(i+1, InputLang.maxlen-1)] = InputLang.i_for_char['EOS']
        return pad, len(input_str)

    @staticmethod
    def tensor_to_str(input_tensor, length=None):
        i = 0
        code = ''
        for tensor_value in input_tensor:
            char = InputLang.char_for_i[tensor_value.item()]
            if char == 'EOS' or length and i == length:
                return code
            code += char
            i += 1
        raise RuntimeError('No string cut or length.')

    @staticmethod
    def random_samples(n_samples):
        '''
            Makes a tensor of indices representing random inputs.

            @params:
                shape, A tuple for the shape of the batches.

            @returns:
                batches, A tensor of batches without EOS chars or padding.
                end_indices, A tensor of batch lengths.
        '''
        batches = torch.randint(
            0, InputLang.i_for_char['EOS'], (n_samples, InputLang.maxlen,), dtype=torch.long, device=DEVICE
        )
        end_indices = torch.randint(
            1, InputLang.maxlen, (n_samples,), dtype=torch.long, device=DEVICE
        )
        return batches, end_indices

    @staticmethod
    def debug_view_batch(batch, start_row, end_row, to_str=True):
        results = []
        for i in range(start_row, end_row):
            results.append([])
            for tensor_value in batch[i]:
                val = tensor_value.item()
                if to_str:
                    if val == InputLang.pad_char_i:
                        val = 'PAD'
                    else:
                        val = InputLang.char_for_i[val]
                results[-1].append(val)
        return results




class OutputLang:
    maxlen = config.MAX_INPUT_OUTPUT_LENGTH
    chars = OUTPUT_CHARS
    char_for_i = dict(enumerate(chars))
    i_for_char = {v: k for k, v in char_for_i.items()}
    pad_char_i = len(chars)

    @staticmethod
    def str_to_tensor(input_str, add_eos=True, pad=False):
        if pad:
            return OutputLang.str_to_padded_tensor(input_str, add_eos=add_eos)
        return torch.LongTensor(
            [OutputLang.i_for_char[char] for char in list(input_str) + (['EOS'] if add_eos else [])],
            device=DEVICE
        )

    @staticmethod
    def str_to_padded_tensor(input_str, add_eos=True):
        pad = OutputLang.pad_char_i * torch.ones(OutputLang.maxlen, dtype=torch.long, device=DEVICE)
        for i, char in enumerate(input_str):
            pad[i] = OutputLang.i_for_char[char]
        if add_eos:
            pad[min(i+1, OutputLang.maxlen-1)] = OutputLang.i_for_char['EOS']
        return pad, len(input_str)

    @staticmethod
    def tensor_to_str(input_tensor, length=None):
        i = 0
        code = ''
        for tensor_value in input_tensor:
            char = OutputLang.char_for_i[tensor_value.item()]
            if char == 'EOS' or length and i == length:
                return code
            code += char
            i += 1
        raise RuntimeError('No string cut or length.')

    @staticmethod
    def debug_view_batch(batch, start_row, end_row, to_str=True):
        results = []
        for i in range(start_row, end_row):
            results.append([])
            for tensor_value in batch[i]:
                val = tensor_value.item()
                if to_str:
                    if val == OutputLang.pad_char_i:
                        val = 'PAD'
                    else:
                        val = OutputLang.char_for_i[val]
                results[-1].append(val)
        return results
