
import numpy as np
import random

DEFAULT_CAPACITY = 10000

class ReplayMemory(object):
    def __init__(self, batch_size, capacity=DEFAULT_CAPACITY):
        self.memory = {}
        self.batch_size = batch_size
        self.first_index = -1
        self.last_index = -1
        self.capacity = capacity

    def store(self, sars_tuple):
        if self.first_index == -1:
            self.first_index = 0
        self.last_index += 1
        self.memory[self.last_index] = sars_tuple   
        if (self.last_index + 1 - self.first_index) > self.capacity:
            self.discardSample()

    def isFull(self):
        return self.last_index + 1 - self.first_index >= self.capacity

    def discardSample(self):
        rand_index = random.randint(self.first_index, self.last_index)
        first_tuple = self.memory[self.first_index]
        del self.memory[rand_index]
        if rand_index != self.first_index:
            del self.memory[self.first_index]
            self.memory[rand_index] = first_tuple
        self.first_index += 1

    def sample(self):
        if self.first_index == -1:
            return
        rand_sample_index = random.randint(self.first_index, self.last_index)
        return self.memory[rand_sample_index]

    def sample_batch(self):
        states = []
        actions = []
        rewards = []
        next_states = []
        terminals = []
        for idx in range(self.batch_size):
            state, action, reward, next_state, terminal = self.sample()
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            terminals.append(terminal)

        return np.array(states).reshape(self.batch_size, -1),           \
                np.array(actions).reshape(self.batch_size, -1),         \
                np.array(rewards).reshape(self.batch_size, -1),         \
                np.array(next_states).reshape(self.batch_size, -1),     \
                np.array(terminals).reshape(self.batch_size, -1)
