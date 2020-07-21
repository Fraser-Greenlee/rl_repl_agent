import time
import wandb

from _logs import Logs
from _constants import LOGS_FOLDER


class LogsTrain(Logs):
    '''
        Holds all the Logs used in training.
        Ensures they use `indices` and all have all their fields filled.
    '''
    logs = {
        'seq2seq': None,
    }

    def __init__(self, indices, logs_seq2seq):
        super().__init__(indices)
        self.logs['seq2seq'] = logs_seq2seq
        logs_seq2seq.indices = indices

    def _check_no_None_in_logs(self, logs):
        for k, v in logs.items():
            if v is None:
                raise RuntimeError("Log key '{}' has unset value.".format(k))
            elif type(v) is dict:
                self._check_no_None_in_logs(v)

    def _check_valid_logs(self):
        self._check_no_None_in_logs(self.logs)

    def _save_logs(self):
        for logs_instance in self.logs.values():
            wandb.log(logs_instance.logs)
            logs_instance.init_log_stores()

    def run_logs(self, seq2seq, repl, tqdm):
        if self._log_this_episode():
            self.logs['seq2seq'].log(seq2seq, repl)
            self._check_valid_logs()
            self._save_logs()

    def _assign_None_to_logs(self, logs):
        for k, v in logs.items():
            if type(v) is not dict:
                logs[k] = None
            else:
                logs[k] = self._assign_None_to_logs(v)
        return logs

    def prepare_logs(self):
        if self._log_this_episode():
            for logs_instance in self.logs.values():
                if type(logs_instance) is dict:
                    self._assign_None_to_logs(logs_instance)
                else:
                    self._assign_None_to_logs(logs_instance.logs)
