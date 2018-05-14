from keras.models import Model
from keras.layers import Dense, Input, Add
from keras.optimizers import Adam

class Crtic():
    def __init__(self, hyperparams, ee):
        self.ee = ee
        self.init_hyperparams(hyperparams)

    def init_hyperparams(self, params):
        self.alpha = params['alpha']
        self.alpha_decay = params['alpha_decay']

    def build_model(self, state_dim, action_dim):
        state_input =  Input(shape=state_dim)
        state1 = Dense(8, activation='relu')(state_input)

        action_input = Input(shape=action_dim)
        action1 = Dense(8, activation='relu')(action_input)

        added = Add()([state1, action1])

        out = Dense(4, activation='relu')(added)

        model = Model(inputs=[state_input, action_input], outputs= out)

        model.compile(
            optimizer=Adam(
                lr=self.alphay,
                decay=self.alpha_decay),
            loss='mse')

        self.model = model

    def update_model(self, samples):
        for old_state, action, reward, new_state, done in samples:
            if not done:
                target_action = self.ee.emit('actor_predict', new_state)
                future_reward = self.model.predict(
                    [new_state, target_action])[0][0]
                self.model.fit([old_state, action])
