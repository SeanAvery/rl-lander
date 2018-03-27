import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class DeepNet():
    def __init__(self, hyper_params):
        self.init_hyper_params(hyper_params)

    def init_hyper_params(self, params):
        self.alpha = params['alpha']
        self.alpha_decay = params['alpha_decay']
        self.epsilon = params['epsilon']
        self.epislon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']

    def build_network(self, state_dim, action_dim):
        model = Sequential()
        model.add(512, activation='relu', input_dim=state_dim)
        model.add(512, activation='relu')
        model.add(Dense(action_dim, activation='linear'))
        model.compile(optimizer=Adam(lr=self.alph, decay=self.alpha_decay), loss='mse')
        return model
