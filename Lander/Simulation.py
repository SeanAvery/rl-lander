import gym
import numpy as np
import random
from time import sleep

class Simulation():
    def __init__(self, model):
        # gym environment
        self.env = gym.make('LunarLander-v2')
        self.get_action_dim()
        self.get_space_dim()

        # create model
        self.model = model
        self.model.build_network(self.state_dim_len, self.action_dim_len)

    def get_action_dim(self):
        self.action_dim_len =self.env.action_space.n

    def get_space_dim(self):
        self.state_dim_len = len(self.env.observation_space.low)
        self.state_dim_low = self.env.observation_space.low
        self.state_dim_high = self.env.observation_space.high

    def choose_action(self, state):
        if np.random.rand() <= self.model.epsilon:
            return np.random.randint(self.action_dim_len)
        else:
            return np.argmax(self.model.model.predict(state))

    def reshape(self, state):
        return state.reshape(1, self.state_dim_len)

    # def calc_reward(self, state):

    def run_simulation(self, num_episodes, isTraining):
        self.isTraining = isTraining

        for i in range(num_episodes):
            print('run episode {0}'.format(i))
            self.run_episode()

    def run_episode(self):
        self.num_ticks = 0
        self.total_reward = 0

        self.old_state = self.reshape(self.env.reset())

        while True:
            done = self.run_tick()
            if done:
                break

    def run_tick(self):
        self.num_ticks += 1
        if not self.isTraining:
            self.env.render()
        return self.run_step()

    def run_step(self):
        action = self.choose_action(self.old_state)
        new_state, reward, done, info = self.env.step(action)
        new_state = self.reshape(new_state)
        if self.isTraining:
            self.model.memory.append((self.old_state, action, reward, new_state, done))
            self.model.update_network()
        self.old_state = new_state
        return done
