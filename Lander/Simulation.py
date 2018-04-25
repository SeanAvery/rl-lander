import gym
import random
from time import sleep
import numpy as np

class Simulation():
    def __init__(self, model):
        self.env = gym.make('LunarLander-v2')
        self.get_action_dim()
        self.get_state_dim()
        
        self.model = model
        model.build_network(self.state_dim_len, self.action_dim_len)

    def get_action_dim(self):
        self.action_dim_len = self.env.action_space.n

    def get_state_dim(self):
        self.state_dim_len = len(self.env.observation_space.low)
        self.state_dim_low = self.env.observation_space.low
        self.state_dim_high = self.env.observation_space.high

    def choose_action(self):
        if np.random.random() < self.model.epsilon:
            return np.random.randint(self.action_dim_len)
        else:
            return np.argmax(self.model.model.predict(state))
    
    def reshape_state(self, state):
        return state.reshape(1, self.state_dim_len)

    def run_simulation(self, num_episodes, is_training):
        for i in range(num_episodes):
            self.run_episode()

    def run_episode(self):
        self.num_ticks = 0
        self.total_reward = 0

        self.old_state = self.reshape_state(self.env.reset())

        while True:
            done = self.run_tick()
            if done:
                break

    def run_tick(self):
        self.num_ticks += 1
        self.env.render()
        return self.run_step()

    def run_step(self):
        action = self.choose_action()
        new_state, reward, done, info = self.env.step(action)
        new_state = self.reshape_state(new_state)
        self.model.memory.append(
            (self.old_state, action, reward, new_state,
                0.0 if done else 1.0))

        if self.num_ticks % 10 == 0:
            self.model.update_network()

        self.old_state = new_state

        return done

