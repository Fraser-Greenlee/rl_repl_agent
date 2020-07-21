import wandb
import torch

from _format_training_data import join_states
from _constants import config, DEVICE


def _memory_tensors_for_episode(episode_size, n_episodes, timestep_i, memory_tensor):
    # TODO check this gets all episodes at timestep
    timestep_row_i = timestep_i * config.PPO_BATCH_SIZE
    return torch.cat(
        [
            memory_tensor[
                episode_size * ep_i + timestep_row_i : episode_size * ep_i + timestep_row_i + config.PPO_BATCH_SIZE
            ] for ep_i in range(n_episodes)
        ]
    )


def _get_code_len_output_len_code_output(filtered_memory, episode_size, n_episodes, timestep_i):
    code_lens = _memory_tensors_for_episode(episode_size, n_episodes, timestep_i, filtered_memory['code_lens'])
    code_tensors = _memory_tensors_for_episode(episode_size, n_episodes, timestep_i, filtered_memory['code_tensors'])
    output_lens = _memory_tensors_for_episode(episode_size, n_episodes, timestep_i, filtered_memory['output_lens'])
    output_tensors = _memory_tensors_for_episode(episode_size, n_episodes, timestep_i, filtered_memory['output_tensors'])
    return torch.cat(
        [code_lens.view(-1, 1), output_lens.view(-1, 1), code_tensors, output_tensors], dim=1
    )


def _unique_code_len_output_len_code_outputs(filtered_memory):
    '''
        Returns indices for unique code snippets in memory.
        Code is checked for uniqueness using chars before EOS.
    '''
    episode_size = (config.MAX_EPISODES_TIMESTEPS - 1) * config.PPO_BATCH_SIZE
    n_episodes = filtered_memory['code_tensors'].size(0) // episode_size
    # stores code and output len at 0 and 1 indices, stoes the concatenated code & output after
    all_code_len_output_len_code_output = torch.tensor([], dtype=torch.long, device=DEVICE)
    for timestep_i in range(config.MAX_EPISODES_TIMESTEPS):
        code_len_output_len_code_output = _get_code_len_output_len_code_output(
            filtered_memory, episode_size, n_episodes, timestep_i
        )
        unique_code_output_pairs = code_len_output_len_code_output.unique(dim=0)
        all_code_len_output_len_code_output = torch.cat(
            [all_code_len_output_len_code_output, unique_code_output_pairs], dim=0
        )
    wandb.log({
        'PPO.ratio of episode steps that were not duplicates': all_code_len_output_len_code_output.size(0) / filtered_memory['code_tensors'].size(0)
    })
    return all_code_len_output_len_code_output


def _memory_without_low_losses(memory):
    indices = memory.seq2seq_loss > config.MIN_PPO_LOSS
    wandb.log({ 'PPO.ratio of episode steps above min loss': indices.sum().item() / indices.size(0) })
    return {
        'states': memory.states[indices],
        'code_lens': memory.code_lens[indices],
        'state_lens': memory.state_lens[indices],
    }

def _unique_states(joined_states):
    unique_states = joined_states.unique(dim=0)
    wandb.log({
        'PPO.ratio of non duplicate episode steps': unique_states.size(0) / joined_states.size(0)
    })
    return unique_states

def _split_joined_states(joined_states):
    state_lens = joined_states[: ,0]
    code_lens = joined_states[: ,1]
    states = joined_states[:, 2:]
    return state_lens, code_lens, states

def best_training_data_from_memory(memory):
    filtered_memory = _memory_without_low_losses(memory)
    joined_states = join_states(filtered_memory['states'], filtered_memory['state_lens'], filtered_memory['code_lens'])
    joined_states = _unique_states(joined_states)
    return joined_states
