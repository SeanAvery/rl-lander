from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge Add, Multiply
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf
import numpy as np

class ActorCritic():
    def __init__(self, hyperparams, memory, sess):
        self.memory = memory
        self.init_hyperparams(hyperparams)

    def init_hyperparams(params):
        self.alpha = params['alpha']
        self.alpha_decay = params['alpha_decay']
        self.epsilon = params['epsilon']
        self.epsilon_decay = params['epsilon_decay']
        self.gamma = params['gamma']
        self.tau = params['tau']

    '''
        ACTOR
        chain rule: de/dC * dC/dA => de/dA
    '''

    def init_actor(self):
        self.state_input = Input(shape=state_dim)

        self.actor_model = self.build_actor_model()
        self.target_actor_model = self.build_actor_model()

        # de/dc
        self.actor_critic_grad = tf.placeholder(
            tf.float32,
            [None, state_dim])

        self.actor_model_weights = self.actor_model.trainable_weights

        # dC/dA
        self.actor_grads = tf.gradients(
            self.actor_model.output,
            actor_model_weights,
            -self.actor_critic_grad)

        grads = zip(self.actor_grads, actor_model_weights)

        self.optimize = tf.train.AdamOptimizers(self.alpha).apply_gradients(grads)

    def build_actor_model(self, state_dim, action_dim):
        l1 = Dense(24, activation='relu')(self.state_input)
        l2 = Dense(48, activation='relu')(l1)
        l3 = Dense(24,activation='relu')(l2)
        output = Dense(action_dim, activation='relu')(l3)

        model = Model(input=self.state_input, output=output)
        adam = Adam(self.alpha)
        self.actor_model.compile(loss="mse", optimizer=adam)

        return model

    ''' CRITIC '''

    def init_critic():
        self.critc_state_input, self.critic_action_input, self.critic_model = self.build_critic_model()
        _, _, self.target_critic_model = build_critic_model()

        # de/dC
        self.critic_grads = tf.gradients(
            self.critic_model.output,
            self.critic_action_input)

        self.sess.run(tf.initialize_all_variables())

    def build_critic_model(self, state_dim, action_dim):
        state_input = Input(shape=state_dim)
        state_l1 = Dense(24, activation='relu')(state_input)
        state_l2 = Dense(48)

        action_input = Input(shape=action_dim)
        action_l1 = Dense(48)(action_input)

        merged = Add()([state_l2, action_l1])
        merged_l1 = Dense(24, activation='relu')(merged)

        output = Dense(1, activation='relu')(merged_l1)
        model = Model(input=[state_input, action_input])

        adam = Adam(self.alpha)
        model.compile(loss='mse', optimizers=adam)
        return state_input, action_input, model

    ''' TRAINNIG '''

    def train_actor(self, samples):
        for old_state, action_reward, new_state, done in samples:
            if not done:
                predicted_action = self.actor_model.predict(old_state)
                grads = self.sess.run(
                    self.critic_grads,
                    feed_dict={
                        self.critic_state_input: old_state,
                        self.critic_action_input: predicted_action
                    })[0]

                self.sess.run(
                    self.optimize,
                    feed_dict={
                        self.actor_state_input: old_state,
                        self.actor_criti_grad: grads })


    def train_critic(self, samples):
        for old_state, action_reward, new_state, done in samples:
            if not done:
                target_action = self.target_actor_model.predict(new_state)
                future_reward = self.target_critic_model.predict(
                    [new_state, target_action])[0][0]

                reward += self.gamma * future_reward

            self.critic_model.fit([old_state, action], reard, verbose=0)

    def train(self, samples):
        self.train_critic(samples)
        self.train_actor(samples)

    ''' UPDATE MODEL '''

    def update_actor_target(self):
        actor_model_weights = self.actor_model.get_weights()
        actor_target_weights = self.actor_critic_model.get_weights()

        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]

        self.target_critic_model.set_weights(actor_target_weights)

    def update_ciritc_target(self):
        critic_model_weights = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()

        for i in range(len(critic_target_weights):
            critic_target_weights[i] = critic_model_weights[i]

        self.critic_target_model.set_weights(critic_target_weights)

    def update_target(self):
        self.update_actor_target()
        self.update_critic_target()
        
