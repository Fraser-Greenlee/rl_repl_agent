import torch

from _constants import config, DEVICE

def _randomly_re_arrange(rows):
    return rows[torch.randperm(rows.size(0))]

def format_training_data(joined_states):
    '''
        Formats training data for the Transformer.
        Converts joined state_len-code_len-state rows into input & target rows with a mask.
    '''
    joined_states = _randomly_re_arrange(joined_states)
    input_sequences, target_sequences = joined_states[:, 2:-1], joined_states[:, 3:]
    mask = torch.ones(target_sequences.size(), dtype=torch.bool, device=DEVICE)
    for col_i in range(1, 1 + target_sequences.size(1)):
        mask[:, col_i - 1] = (col_i < joined_states[:,0]) & (col_i >= joined_states[:,1])
    return input_sequences, target_sequences, mask

def join_states(states, states_lens, code_lens):
    return torch.cat(
        [states_lens.view(-1, 1), code_lens.view(-1, 1), states], dim=1
    )
