from Models.DeepNet import DeepNet
from Simulation import Simulation

'''
    HYPER PARAMS
'''

hyper_params_1 = {
    'epsilon': 1,
    'epsilon_min': 0.05,
    'epsilon_decay': 0.9995,
    'alpha': 0.1,
    'alpha_decay': 0.9995
}

'''
    TRAIN & TEST
'''

if __name__ == '__main__':
    deep_net = DeepNet(hyper_params_1)
    simulation = Simulation(deep_net)
