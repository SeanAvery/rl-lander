import gym
import random
from time import sleep
# from Model import Model
import numpy as np

class Simulation():
    def __init__(self):
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

    def run_simulation(self, num_episodes):
        for i in range(num_episodes):
            self.run_episodes()

    def run_episode(self):
        self.num_ticks = 0
        self.total_reward = 0

        self.old_state = self.env.reset()

        while True:
            done = self.run_tick()
            if done:
                break

    def run_tick(self):
        self.num_ticks += 1
        sleep(0.1)
        self.env.render()
        return self.run_step()

    def run_step(self):
        action = self.choose_action()
        new_state, reward, done, info = self.env.step(action)

        self.model.memory.append(
            (old_state, action, reward, new_state,
                0.0 if done else 1.0))

        if self.num_tick % self.model.update_slow_target_every:
            print('update slow target op')

        old_state = new_state

        return done

#
# if __name__ == '__main__':
#     model = Model()
#     simulation = Simulation()
#     simulation.run(3)
