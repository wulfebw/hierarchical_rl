
import random

class Agent(object):

    def step(self):
        raise NotImplementedError("Override me")

class TestAgent(Agent):

    def __init__(self, num_actions):
        self.actions = range(num_actions)
        self.steps = 0
        self.episodes = 0

    def step(self, next_state, reward):
        self.steps += 1
        return random.choice(self.actions)

    def start_episode(self, state):
        self.episodes += 1
        return random.choice(self.actions)

class NeuralAgent(Agent):

    def __init__(self):
        pass

    def step(self, next_state, reward):
        pass

    def choose_action(self):
        pass

    def train(self):
        pass

    def start_episode(self):
        pass

    def end_episode(self):
        pass