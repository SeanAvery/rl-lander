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

    def run_simulation(self, num_episodes):
        for i in range(num_episodes):
            self.run_episode()

    def run_episode(self):
        self.num_ticks = 0
        self.total_reward = 0

        self.old_state = self.env.reset()

        while True:
            done = self.run_tick()
            if done:
                break

    def run_tick(self):
        sleep(0.1)
        self.num_ticks += 1
        self.env.render()
        return self.run_step()

    def run_step(self):
        action = self.choose_action(self.old_state)
        new_state, reward, done, info = self.env.step(action)
        self.old_state = new_state
        return done
