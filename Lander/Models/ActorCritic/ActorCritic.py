from Models.ActorCritic.Actor import Actor
from Models.ActorCritic.Critic import Critic
from pyee import EvenetEmitter
from random import sample

class ActorCritic():
    def __init__(self, hyperparams, Memory):
        self.init_hyperparams(hyperparams)
        self.create_actor_critic_bridge()
        self.actor = Actor(hyperparams)
        self.critic = Critic(hyperparams)
        self.memory = Memory

    def init_hyperparams(self, params):
        self.batch_size = params['batch_size']

    def create_actor_critic_bridge(self):
        self.ee = EvenetEmitter()

        @self.ee.on('actor_predict')
        def event_handler(new_state):
            return self.actor.model.predict(new_state)

        @self.ee.on('error')
        def error_handler(err):
            print('error', err)

    def update_model(self):
        samples = sample(
            self.memory,
            min(len(self.memory), self.batch_size))

        self.actor.update_model(samples)
        self.critic.update_model(samples)
