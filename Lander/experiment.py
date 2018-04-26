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
    'batch_size': 32,
    'gamma': 0.99
}

'''
    TRAIN & TEST
'''

run_params = {
    'train_ticks': 10000,
    'eval_ticks': 100
}

if __name__ == '__main__':
    deep_net = DeepNet(hyper_params_1)
    simulation = Simulation(deep_net)
    simulation.run_simulation(run_params['train_ticks'], True)
    simulation.run_simulation(run_params['eval_ticks'], False)
