import torch
import wandb

from _language import InputLang, OutputLang
from _constants import config, DEVICE


def _assign_output_values(repl, code_tensor, code_lens):
    # TODO update to return states
    code, output = [], []
    for i in range(code_tensor.size(0)):
        code.append(InputLang.tensor_to_str(code_tensor[i], length=code_lens[i]))
        output.append(repl.run_code(code[i]))
    return repl.encoded_state(code=code, output=output)

def random_code_output_samples(repl, n_samples):
    code_batches, code_lengths = InputLang.random_samples(n_samples)
    output_batches = torch.zeros((n_samples, OutputLang.maxlen), dtype=torch.long, device=DEVICE)
    return _assign_output_values(repl, code_batches, code_lengths)

def pretrain_seq2seq(seq2seq, repl):
    '''
        pretrains self.seq2seq on random code snippets.
    '''
    raise NotImplementedError()
