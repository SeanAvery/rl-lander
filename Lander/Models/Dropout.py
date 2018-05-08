import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from time import time
import numpy as np

class DropoutModel():
    def __init__(self, hyperparams, Memory):
        self.init_hyperparams(hyperparams)
        self.memory = Memory

    def init_hyperparams(self, params):
        self.alpha = params['alpha']
        self.alpha_decay = params['alpha_decay']
        self.epsilon = params['epsilon']
        self.epsilon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.batch_size = params['batch_size']
        self.gamma = params['gamma']

    def init_model(self, state_dim, action_dim):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=state_dim))
        model.add(Dropout(0.5))
        model.add(Dense(64, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(action_dim, activation='softmax'))

        model.compile(optimizer=Adam(lr=self.alpha, decay=self.alpha_decay), loss='mse')
        self.model = model

        self.tensorboard = TensorBoard(
                log_dir="logs/{}".format(time()),
                histogram_freq=0,
                batch_size=32,
                write_graph=True)

    def update_model(self):
        x, y = [], []
        batch = self.memory.sample(self.batch_size)

        for old_state, action, reward, new_state, done in batch:
            y_target = self.model.predict(old_state)
            if done:
                y_target[0][action] = reward
            else:
                y_target[0][action] = reward + self.gamma * np.max(self.model.predict(new_state)[0])

            x.append(old_state[0])
            y.append(y_target[0])

        self.model.fit(np.array(x), np.array(y), batch_size=len(x), callbacks=[self.tensorboard])

    def update_epsilon(self):
        epsilon = self.epsilon * self.epsilon_decay
        self.epsilon = max(self.epsilon_min, epsilon)
