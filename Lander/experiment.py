from Models.DeepNet import DeepNet
from Simulation import Simulation

'''
    HYPER-PARAMS
'''

hyper_params_1 = {
    'epsilon': 1,
    'epsilon_min': 0.05,
    'epsilon_decay': 0.95,
    'alpha': 1,
    'alpha_decay': 0.95,
    'batch_size': 3,
    'gamma': 0.99
}

'''
    TRAIN & TEST
'''

if __name__ == '__main__':
    deep_net = DeepNet(hyper_params_1)
    simulation = Simulation(deep_net)
    simulation.run_simulation(10, True)
    simulation.run_simulation(3, False)
