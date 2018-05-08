import keras
from keras.models import Sequential, Dropout
from keras.layers import Dense
from keras.callbacks import TensorBoard

class Dropout():
    def __init__(self, hyperparams, Memory):
        self.init_hyper_params(hyperparams)
        self.Memory = Memory

    def init_hyperparams(self, params):
        self.alpha = params['alpha']
        self.alpha_decay = params['alpha_decay']
        self.epsilon = params['epsilon']
        self.epislon_min = params['epsilon_min']
        self.epsilon_decay = params['epsilon_decay']
        self.batch_size = params['batch_size']
        self.gamma = params['gamma']

    def init_model(self, state_dim, action_dim):
        model = Sequential()
        model.add(Dense(64, activation='relu'), input_dim=state_dim)
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
        batch = self.Memory.sample()

        for old_state, actoin, new_state, done in mini_batch:
            y_target = self.model.predict(old_state)
            if done:
                y_target[0][action] = reward
            else:
                y_target[0][action] = reward + self.gamma * np.max(self.model.predict(new_state)[0])

            x.append(old_state[0])
            y.append(y_target[0])

        self.model.fit(np.array(x), np.array(y), batch_size=len(x), callbacks=[self.tensorboard])
