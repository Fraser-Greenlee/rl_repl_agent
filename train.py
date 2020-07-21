import os
import torch
import math
import wandb
import datetime
from tqdm import tqdm

from _format_training_data import format_training_data, join_states
from _pretrain_seq2seq import random_code_output_samples
from _find_best_training_data import best_training_data_from_memory
from _progress_indices import ProgressIndices
from _logs_train import LogsTrain
from _ppo import PPO
from _seq2seq import Seq2Seq
from _memory import Memory
from _repl_env import ReplEnv
from _pretrain_seq2seq import pretrain_seq2seq
from _language import InputLang, OutputLang
from _reward_handler import RewardHandler
from _constants import (
    config,
    RENDER_FILE,
    MODELS_FOLDER,
    DEVICE
)


class Trainer:
    '''
        Trains the Seq2Seq & PPO models.
    '''
    def __init__(self, load_from=None):
        self.indices = ProgressIndices()
        self._load_models(load_from)
        self.repl = ReplEnv(config.PPO_BATCH_SIZE)
        self.memory = Memory()
        self.rewad_handler = RewardHandler(self.indices)

    def _load_models(self, load_from):
        self.seq2seq = Seq2Seq(self.indices)
        self.ppo = PPO(self.indices)
        if load_from:
            self.seq2seq.load(load_from)
            self.ppo.load(load_from)

    def _run_old_policy(self, states, state_lens, code_lens):
        actions = self.ppo.policy_old.act(states, state_lens, self.memory)
        new_states, new_states_lens, new_code_lens = self.repl.step(actions)
        loss, accuracy = self.seq2seq.next_char_loss(
            *format_training_data(join_states(new_states, new_states_lens, new_code_lens))
        )
        rewards = self.rewad_handler.reward_from_loss(loss, self.repl.encoded_code()[0])
        is_finished_episodes = torch.ones(
            config.PPO_BATCH_SIZE, dtype=torch.bool, device=DEVICE
        ) * torch.tensor(self.indices.timestep + 1 == config.MAX_EPISODES_TIMESTEPS)
        self.memory.store(
            states=states, code_lens=code_lens, state_lens=state_lens, rewards=rewards, is_terminals=is_finished_episodes,
            seq2seq_loss=loss, seq2seq_train_accuracy=accuracy
        )
        return new_states, new_states_lens, new_code_lens, rewards

    def _render(self, reward, run_render):
        if config.RENDER and run_render:
            string_io = self.repl.render(mode='ansi')
            open(RENDER_FILE, 'a').write(
                'Episode i: {}\treward: {:.4f}\n{}'.format(
                    self.indices.episode, reward,
                    string_io.getvalue()
                )
            )

    @staticmethod
    def _update_lr(optimizer, new_value):
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_value

    def _update_learning_rates(self, seq2seq_inaccuracy):
        # TODO try setting a minimum learning rate
        self._update_lr(self.seq2seq.optimizer, seq2seq_inaccuracy * config.SEQ2SEQ_LEARNING_RATE)
        self._update_lr(self.ppo.optimizer,     seq2seq_inaccuracy * config.PPO_LEARNING_RATE)

    @staticmethod
    def _batch_start_end(batch_i):
        start = batch_i * config.SEQ2SEQ_BATCH_SIZE
        end = start + config.SEQ2SEQ_BATCH_SIZE
        return start, end

    def _update_seq2seq(self):
        '''
            Use the code from memory that gave a loss > config.MIN_TRAINING_SAMPLE_REWARD
            and remove duplicates.

            `self.memory.code_tensors` has repeats with structure:
                * self.memory.code_tensors[0] = code with 1 char
                * self.memory.code_tensors[batch_size] = code with 2 chars
                * self.memory.code_tensors[2*batch_size] = code with 2 chars
                * ...repeats
        '''
        input_sequences, target_sequences, mask = format_training_data(best_training_data_from_memory(self.memory))
        n_batches = input_sequences.size(0) // config.SEQ2SEQ_BATCH_SIZE
        all_accuracies, all_losses = 2*[torch.zeros(n_batches, dtype=torch.float, device=DEVICE)]
        for batch_i in range(n_batches):
            start, end = self._batch_start_end(batch_i)
            all_accuracies[batch_i], all_losses[batch_i] = self.seq2seq.train_step(
                input_sequences[start:end], target_sequences[start:end], mask[start:end]
            )
        wandb.log({
            'seq2seq.train.loss.avg': all_losses.mean(),		
            'seq2seq.train.accuracy.avg': all_accuracies.mean()		
        })		
        self._update_learning_rates(		
            (1 - all_accuracies.mean().item())		
        )

    def _update(self):
        if self.indices.episode % config.UPDATE_EVERY_N_EPISODES == 0:
            self.ppo.update(self.memory, self.indices.episode, tqdm)
            if (self.indices.episode / config.UPDATE_EVERY_N_EPISODES) > config.PRE_TRAIN_UPDATES:
                self._update_seq2seq()
            self.memory.clear_memory()

    def _model_folder(self, model_name):
        folder = '{}{}/'.format(MODELS_FOLDER, model_name)
        os.makedirs(folder, exist_ok=True)
        return folder

    def _save(self):
        # TODO save logs
        if self.indices.episode % config.SAVE_EVERY_N_EPISODES == 0:
            model_name = 'episode_{}'.format(self.indices.episode)
            folder = self._model_folder(model_name)
            self.seq2seq.save(folder)
            self.ppo.save(folder)

    @staticmethod
    def _all_done(is_finished_episodes):
        # TODO must all episodes end at the same time? - I think so but unsure, it's just when the output is ''
        return is_finished_episodes.min().item() is True

    def _run_episode(self, render=False):
        all_rewards = torch.zeros((config.MAX_EPISODES_TIMESTEPS, config.PPO_BATCH_SIZE), dtype=torch.float, device=DEVICE)
        states, state_lens, code_lens = self.repl.reset()
        for i in range(config.MAX_EPISODES_TIMESTEPS):
            self.indices.timestep = i
            states, state_lens, code_lens, reward = self._run_old_policy(states, state_lens, code_lens)
            all_rewards[i] = reward
            # self._render(reward, render)
        wandb.log({'PPO.cumulative reward': all_rewards.sum().item()})


    def _test_random_pairs(self):
        states, state_lens, code_lens = random_code_output_samples(self.repl, config.SEQ2SEQ_BATCH_SIZE)
        losses, accuracy = self.seq2seq.next_char_loss(
            *format_training_data(join_states(states, state_lens, code_lens))
        )
        return losses, accuracy

    def _log_seq2seq_test(self):
        loss, accuracy = self._test_random_pairs()
        wandb.log({
            'seq2seq.test.loss.avg': loss,
            'seq2seq.test.accuracy.avg': accuracy,
        })

    def _after_each_episode(self):
        self._update()
        self._save()
        self._log_seq2seq_test()

    def train(self):
        '''
            Starts training.
        '''
        try:
            for i in tqdm(range(1, config.PPO_N_TRAINING_EPISODES + 1)):
                self.indices.episode = i
                self._run_episode()
                self._after_each_episode()
        except KeyboardInterrupt:
            self._save()
        print('Reached config.PPO_N_TRAINING_EPISODES, stopping.')


if __name__ == '__main__':
    Trainer().train()
