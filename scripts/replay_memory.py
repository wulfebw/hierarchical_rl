
import numpy as np
import random

DEFAULT_CAPACITY = 100000

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

    def isEmpty(self):
        return self.first_index == -1

    def discardSample(self):
        rand_index = random.randint(self.first_index, self.last_index)
        first_tuple = self.memory[self.first_index]
        del self.memory[rand_index]
        if rand_index != self.first_index:
            del self.memory[self.first_index]
            self.memory[rand_index] = first_tuple
        self.first_index += 1

    def sample(self):
        if self.isEmpty():
            raise Exception('Unable to sample from replay memory when empty')
        rand_sample_index = random.randint(self.first_index, self.last_index)
        return self.memory[rand_sample_index]

    def sample_batch(self):
        # must insert data into replay memory before sampling
        if self.isEmpty():
            raise Exception('Unable to sample from replay memory when empty')

        # determine shape of states
        state_shape = np.shape(self.memory.values()[0][0])
        states_shape = (self.batch_size,) + state_shape

        states = np.empty(states_shape)
        actions = np.empty((self.batch_size, 1))
        rewards = np.empty((self.batch_size, 1))
        next_states = np.empty(states_shape)
        terminals = np.empty((self.batch_size, 1))

        # sample batch_size times from the memory
        for idx in range(self.batch_size):
            state, action, reward, next_state, terminal = self.sample()
            states[idx] = state
            actions[idx] = action
            rewards[idx] = reward
            next_states[idx] = next_state
            terminals[idx] = terminal

        return states, actions, rewards, next_states, terminals
