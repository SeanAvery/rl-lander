import gym
import random
from time import sleep

class Simulation():
    def __init__(self):
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
        print('action_dim_len', self.action_dim_len)
        return random.randint(0, self.action_dim_len - 1)
    
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
    simulation = Simulation()
    simulation.run(3)
