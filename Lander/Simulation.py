import gym

class Simulation():
    def __init__(self, model):
        # gym environment
        self.env = gym.make('LunarLander-v2')
        self.get_action_dim()
        self.get_space_dim()

        # create model
        self.model = model
        self.model.build_network(self.state_dim_len)

    def get_action_dim(self):
        self.action_dim_len =self.env.action_space.n

    def get_space_dim(self):
        self.state_dim_len = len(self.env.observation_space.low)
        self.state_dim_low = self.env.observation_space.low
        self.state_dim_high = self.env.observation_space.high
