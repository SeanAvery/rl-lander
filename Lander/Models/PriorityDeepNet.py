import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class PriorityDeepNet():
    def __init__(self, hyper_params):
        self.init_hyper_params(hyper_params)
        self.memory = []

    def init_hyper_parmas(self, params):
        self.alpha = params['alpha']
        self.alpha_decay = params['alpha_decay']
        self.epsilon = params['epsilon']
        self.epislon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.batch_size = params['batch_size']
        self.gamma = params['gamma']
    
    def build_network(self, state_dim, action_dim):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_dim=state_dim)
        model.add(Dense(512, activation='relu'))
        model.add(Dense(action_dim, activation='linear'))
        model.compile(
            optimizer=Adam(lr=self.alpha, decay=self.alpha_decay)
            loss='mse')
                
        

