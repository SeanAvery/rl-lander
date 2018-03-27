from Models.DeepNet import DeepNet

'''
    HYPER PARAMS
'''

hyper_params_1 = {
    'epsilon': 1,
    'epsilon_min': 0.05,
    'epsilon_decay': 0.9995,
    'alpha': 0.1,
}

'''
    TRAIN & TEST
'''

if __name__ == '__main__':
    print(DeepNet)
    deep_net = DeepNet(hyper_params_1)
