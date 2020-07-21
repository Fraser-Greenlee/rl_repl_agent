
from _constants import config


class Logs(object):
    '''
        Base class for Logs.
    '''
    indices = None
    logs = None # Store logs dict here

    def __init__(self, indices):
        self.indices = indices

    def _log_this_episode(self):
        return self.indices.episode > 0 and self.indices.episode % config.LOG_EVERY_N_EPISODES == 0

    def init_log_stores(self):
        raise NotImplementedError()

    def log(self):
        raise NotImplementedError()
