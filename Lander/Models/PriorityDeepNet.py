
class PriorityDeepNet():
    def __init__(self, hyper_params):
        self.init_hyper_params(hyper_params)

    def init_hyper_parmas(self, params):
        self.alpha = params['alpha']
        self.alpha_decay = params['alpha_decay']
        self.epsilon = params['epsilon']
        self.epislon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.batch_size = params['batch_size']
        self.gamma = params['gamma']
    
