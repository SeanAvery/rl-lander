import gym
import random
from time import sleep
from Model import Model
import numpy as np

class Simulation():
    def __init__(self, model):
        self.model = model
        self.env = gym.make('LunarLander-v2')
        self.get_action_dim()
        self.get_space_dim()

    def get_action_dim(self):
        self.action_dim_len =self.env.action_space.n

    def get_space_dim(self):
        self.state_dim_len = len(self.env.observation_space.low)
        self.state_dim_low = self.env.observation_space.low
        self.state_dim_high = self.env.observation_space.high

    def choose_action(self):
        if np.random.random() < self.model.epsilon:
            return np.random.randint(self.action_dim_len)
        else:
            q_state = model.get_q_state()
            return np.argmax(q_sate)

    def run(self, num_episodes):
        for i in range(num_episodes):
            self.env.reset()
            while True:
                self.env.render()
                action = self.choose_action()
                state, reward, done, info = self.env.step(action)
                sleep(0.1)
                if done:
                    break

if __name__ == '__main__':
    model = Model()
    simulation = Simulation()
    simulation.run(3)
