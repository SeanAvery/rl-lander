import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import random
import numpy as np

class DeepNet():
    def __init__(self, hyper_params):
        self.init_hyper_params(hyper_params)
        self.memory = []

    def init_hyper_params(self, params):
        self.alpha = params['alpha']
        self.alpha_decay = params['alpha_decay']
        self.epsilon = params['epsilon']
        self.epislon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.batch_size = params['batch_size']
        self.gamma = params['gamma']

    def build_network(self, state_dim, action_dim):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_dim=state_dim))
        model.add(Dense(512, activation='relu'))
        model.add(Dense(action_dim, activation='linear'))
        model.compile(optimizer=Adam(lr=self.alpha, decay=self.alpha_decay), loss='mse')
        self.model = model

    def calc_epsilon(self):
        epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = max(self.epislon_min, epsilon)

    def update_network(self):
        x_batch, y_batch = [], []
        mini_batch = random.sample(self.memory, min(len(self.memory), self.batch_size))

        for old_state, action, reward, new_state, done in mini_batch:
            print('old_state', old_state)
            y_target = self.model.predict(old_state)
            print('y_target', y_target)
            if done:
                # estimated future reward is 0
                y_target[0][action] = reward
            else:
                # find estimated future rewards
                y_target[0][action] = reward + self.gamma * np.max(self.model.predict(new_state)[0])

            x_batch.append(old_state[0])
            y_batch.append(y_target[0])

        self.model.fit(np.array(x_batch), np.array(y_batch), batch_size=len(x_batch))
