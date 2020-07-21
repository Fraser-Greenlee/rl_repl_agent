import torch
from torch import nn
from torch.distributions import Categorical

from _actor_lstm import ActorLSTM
from _value_lstm import ValueLSTM
from _constants import DEVICE


class ActorCritic(nn.Module):
    '''
        Holds the actor (ActorLSTM) and critic (ValueLSTM).

        Use this class to decide on actions and evaluate previous ones.
    '''
    def __init__(self, action_size):
        super(ActorCritic, self).__init__()
        self.action_size = action_size
        # actor
        self.action_lstm = ActorLSTM(action_size)
        # critic
        self.value_lstm = ValueLSTM(action_size)

    def forward(self):
        raise NotImplementedError

    def act(self, states, states_lens, memory):
        '''
            Find actions for a batch of states & record them.
            Since `ActorLSTM` uses batches we have to format the single action into a batch.

            @params
                states:         A padded tensor representing a batch of programs states (code-output-).
                states_lens:    Tensor of state lengths, one for each state in the batch.
                memory:         An instance of `Memory` holding training records for the current episode.
            @returns
                A tensor of actions, one for each batch.
        '''
        actions_probs = self.action_lstm(states, states_lens)
        dists = Categorical(actions_probs)
        actions = dists.sample()
        memory.store(
            actions=actions, logprobs=dists.log_prob(actions)
        )
        return actions

    def evaluate(self, states_batch, states_lengths, actions):
        '''
        @params
            padded_states:
                A batch of variable lengthed padded state sequences.
                Has shape (batch_size, max_seq_len)
            actions:
                A tensor of actions picked for the batch.
        '''
        # TODO this takes 2.5 seconds, try and speed it up
        # TODO this produces a massive computational graph, can it be smaller?
        action_probs = self.action_lstm(states_batch, states_lengths)
        dists = Categorical(action_probs)

        actions_logprobs = dists.log_prob(actions)
        dist_entropys = dists.entropy()

        state_values = self.value_lstm(states_batch, states_lengths)

        return actions_logprobs, torch.squeeze(state_values), dist_entropys
