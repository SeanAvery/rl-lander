from keras.models import Model
from keras.layers import Dense, Input, Add
from keras.optimizers import Adam

class Crtic():
    def __init__(self):
        return

    def build_model(self, state_dim, action_dim):
        state_input =  Input(shape=state_dim)
        state1 = Dense(8, activation='relu')(state_input)

        action_input = Input(shape=action_dim)
        action1 = Dense(8, activation='relu')(action_input)

        added = Add()([state1, action1])

        out = Dense(4, activation='relu')(added)

        semodel = Model(inputs=[state_input, action_input], outputs= out)

        model.compile(
            optimizer=Adam(
                lr=self.alphay,
                decay=self.alpha_decay),
            loss='mse')

        self.model = model
