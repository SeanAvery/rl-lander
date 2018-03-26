import gym

class Simulation():
    def __init__(self):
        self.env = gym.make('LunarLander-v2')
        print('self.env', self.env)

if __name__ == '__main__':
    simulation = Simulation()

