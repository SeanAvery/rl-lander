import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

class DropoutNet():
    def build_model(self):
        model = Sequential()
        model.add(Dense(512, activation='relu', input_dim=self.state_dim))
        model.add(DropOout(0.2))
        model.add(Dense(512, activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(self.action_dim, activation='linear'))
        model.compile(
            optimizer=Adam(
                lr=self.alpha, 
                decay=self.alpha_decay),
                loss='mse')

if __name__ == '__main__':
