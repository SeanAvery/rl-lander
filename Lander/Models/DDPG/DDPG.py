import numpy as np
import tensorflow as tf

hyperparams = {
    'gamma' : 0.99,
    'tau' : 0.001,
    'normalize_observations' : True,
    'normalize_returns' : False,
    'action_noise' : None,
    'param_noise' : None,
    'action_range' : (-1., 1.),
    'return_range' : (-np.inf, np.inf),
    'observation_range' : (-5., 5.)
}

memory = 'hey'

class DDPG():
    def __init__(self, hyperparms, memory):
        self.init_hyperparams(hyperparams)
        self.memory = memory

    def init_hyperparams(self, params):
        self.gamma = params['gamma']
        self.tau = params['tau']
        self.normalize_observations = params['normalize_observations']
        self.normalize_returns = params['normalize_returns']
        self.action_noise = params['action_noise']
        self.param_noise = params['param_noise']
        self.action_range = params['action_range']
        self.return_range = params['action_range']
        self.observation_range = params['observation_range']


if __name__ == '__main__':
    ddpg = DDPG(hyperparams, memory)
