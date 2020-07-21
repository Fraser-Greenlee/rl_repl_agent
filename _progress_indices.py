
class ProgressIndices(object):
    '''
        Shared variable used across `Logs` to track progress during training.
    '''
    # Current PPO batch being run
    episode = 0
    # Timestep in the current episode
    timestep = 0
