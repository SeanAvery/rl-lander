from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

class Actor():
    def __init__(self, hyperparams):
        self.alpha = hyperparams['alpha']
        self.alpha_decay = hyperparams['alpha_decay']

    def build_model(self, state_dim, action_dim):
        model = Sequential()
        model.add(Dense(64, activation='relu', input_dim=state_dim))
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))

        model.compile(
            optimizer=Adam(
                lr=self.alphay,
                decay=self.alpha_decay),
            loss='mse')

        self.model = model

    def update_model(self, samples):
        for old_state, action, reward, new_state, done in samples:
            return 
