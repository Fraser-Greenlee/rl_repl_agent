import torch
from _constants import config, DEVICE
from _language import InputLang, OutputLang


class MemoryInterface:
    '''
        Interface for recording what happened each episode for each update, expects data in batches.
        Uses tensors for efficiency.
    '''
    rows = None
    update_row_count = config.UPDATE_EVERY_N_EPISODES * config.PPO_BATCH_SIZE * config.MAX_EPISODES_TIMESTEPS

    def __init__(self):
        self._init_stores()

    def _init_stores(self):
        self.current_row_i = 0
        self.row_progress = self._new_row_progress()
        self._init_value_stores()

    def _init_value_stores(self):
        raise NotImplimentedError()

    def _padded_sequence(self, pad_char_i, maxlen):
        # batch of only padding chars
        return pad_char_i * torch.ones(
            (self.update_row_count, maxlen), dtype=torch.long, device=DEVICE
        )

    def _seq_len_holder(self):
        return torch.zeros(self.update_row_count, dtype=torch.long, device=DEVICE)

    def _new_row_progress(self):
        return set(self.rows)

    def _store_row_dict_to_attrs(self, name_to_val):
        for key, batch_of_vals in name_to_val.items():
            self.row_progress.remove(key)
            if type(batch_of_vals) is torch.tensor:
                batch_of_vals = batch_of_vals.detach()
            getattr(self, key)[self.current_row_i:self.current_row_i + config.PPO_BATCH_SIZE] = batch_of_vals

    def _check_finished_row(self):
        if len(self.row_progress) == 0:
            # TODO check how much of the tensors has been filled
            self.current_row_i += config.PPO_BATCH_SIZE
            self.row_progress = self._new_row_progress()

    def store(self, **name_to_val):
        '''
            Store a batch of values.
        '''
        self._store_row_dict_to_attrs(name_to_val)
        self._check_finished_row()

    def clear_memory(self):
        self._init_stores()
