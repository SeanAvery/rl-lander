import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout

class DropoutNet():
    def __init__(self, simulation):
        self.simulation = simulation

        self.init_hyperparams()
        self.build_model()

    def init_hyperparams():
        self.alpha = 0.1
        self.epsilon 1.0
        self.epsilon_decay = 0.9999
        self.epsilon_min = 0.05

    def build_model(self):
        model = Sequential()


if __name__ == '__main__':
