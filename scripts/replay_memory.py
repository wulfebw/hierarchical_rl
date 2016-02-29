
import numpy as np
import random
import theano

DEFAULT_CAPACITY = 10000
SAMPLING_CAPACITY_FACTOR = 100.

class ReplayMemory(object):

    def __init__(self, batch_size, capacity=DEFAULT_CAPACITY):
        self.memory = {}
        self.batch_size = batch_size
        self.first_index = -1
        self.last_index = -1
        self.capacity = capacity
        self.terminal_count = 0

    def store(self, sars_tuple):
        self.terminal_count += sars_tuple[-1]
        if self.first_index == -1:
            self.first_index = 0
        self.last_index += 1
        self.memory[self.last_index] = sars_tuple   
        if (self.last_index + 1 - self.first_index) > self.capacity:
            self.discard_sample()

    def is_full(self):
        return self.last_index + 1 - self.first_index >= self.capacity

    def is_empty(self):
        return self.first_index == -1

    def is_ready_to_sample(self):
        """
        :description: is the replay memory ready to sample from
        """
        return self.last_index + 1 - self.first_index >= self.capacity / SAMPLING_CAPACITY_FACTOR

    def discard_sample(self):
        rand_index = random.randint(self.first_index, self.last_index)
        first_tuple = self.memory[self.first_index]
        del self.memory[rand_index]
        if rand_index != self.first_index:
            del self.memory[self.first_index]
            self.memory[rand_index] = first_tuple
        self.first_index += 1

    def sample(self):
        if self.is_empty():
            raise Exception('Unable to sample from replay memory when empty')
        rand_sample_index = random.randint(self.first_index, self.last_index)
        return self.memory[rand_sample_index]

    def sample_batch(self):
        # must insert data into replay memory before sampling
        if self.is_empty():
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

        return states.astype(theano.config.floatX), actions, \
            rewards.astype(theano.config.floatX), \
            next_states.astype(theano.config.floatX), terminals

class SequenceReplayMemory(object):
    """
    :description: this is from https://github.com/spragunr/deep_q_rl
    """
    
    def __init__(self, input_shape, sequence_length, batch_size, capacity):
        """
        :type input_shape: int or tuple 
        :param: the shape of the state input to the network

        :type sequence_length: int
        :param sequence_length: the length of the sequence used by the network

        :type batch_size: int
        :param batch_size: the size of a minibatch

        :type capacity: int
        :param capacity: maximum size of the replay memory
        """
        self.input_shape = input_shape
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.capacity = capacity
        self.bottom = 0
        self.top = 0
        self.size = 0

        if type(self.input_shape) is int:
            self.input_shape = (self.input_shape, )   

        if self.sequence_length == 1:
            self.sequence_shape = self.input_shape
        else:         
            self.sequence_shape = (self.sequence_length,) + self.input_shape
        self.batch_shape = (self.batch_size, ) + self.sequence_shape

        # Allocate the circular buffers
        self.states = np.zeros(((self.capacity, ) + self.input_shape), dtype='int32')
        self.actions = np.zeros(self.capacity, dtype='int32')
        self.rewards = np.zeros(self.capacity, dtype=theano.config.floatX)
        self.terminals = np.zeros(self.capacity, dtype='bool')

    def store(self, state, action, reward, terminal):
        """
        :description: stores a state, the action taken in that state, and the reward received for 
            for being the state (i.e., we use r(s) not r(s,a)) in the replay memory

        :type state: np.array
        :param state: the current state

        :type action: int 
        :param action: the action taken in this state

        :type reward: float 
        :param reward: the reward received for being in state
        """

        self.states[self.top] = state
        self.actions[self.top] = action
        self.rewards[self.top] = reward
        self.terminals[self.top] = terminal

        if self.size == self.capacity:
            self.bottom = (self.bottom + 1) % self.capacity
        else:
            self.size += 1

        self.top = (self.top + 1) % self.capacity

    def make_last_sequence(self, next_state):
        """
        :description: given a state, this method creates a sequence of sequence_length where
            the last state in that sequence is passed in state. This is used to get an action

        :type next_state: np.array
        :param next_state: the next state to be inserted last into the sequence
        """
        sequence = np.zeros(self.sequence_shape, dtype=theano.config.floatX)

        # if this is not the first state collected and the previous state was not terminal
        # then we want to use the past sequence_len - 1 states in deciding the action
        if len(self.terminals) > 0 and self.terminals[-1] == False:
            indexes = np.arange(self.top - self.sequence_length + 1, self.top)
            sequence[0:self.sequence_length - 1] = self.states.take(indexes, axis=0, mode='wrap')
        sequence[-1] = next_state
        return sequence

    def is_full(self):
        """
        :description: is the replay memory full
        """
        return self.size == self.capacity

    def is_ready_to_sample(self):
        """
        :description: is the replay memory ready to sample from
        """
        return self.size >= self.capacity / SAMPLING_CAPACITY_FACTOR

    def sample_batch(self):
        """
        :description: sample a minibatch of data
        """

        # must insert sufficient data into replay memory before sampling
        if not self.is_ready_to_sample():
            raise Exception('Unable to sample from replay memory when empty')

        # allocate batch containers
        states = np.empty(self.batch_shape)
        actions = np.empty((self.batch_size, 1))
        rewards = np.empty((self.batch_size, 1))
        next_states = np.empty(self.batch_shape)
        terminals = np.empty((self.batch_size, 1))

        # sample batch_size times from the memory
        count = 0 
        while count < self.batch_size:

            index = np.random.randint(self.bottom, self.bottom + self.size - self.sequence_length)
            initial_indices = np.arange(index, index + self.sequence_length)
            transition_indices = initial_indices + 1
            end_index = index + self.sequence_length - 1
            
            # original quote:
            # "Check that the initial state corresponds entirely to a
            # single episode, meaning none but the last frame may be
            # terminal. If the last frame of the initial state is
            # terminal, then the last frame of the transitioned state
            # will actually be the first frame of a new episode, which
            # the Q learner recognizes and handles correctly during
            # training by zeroing the discounted future reward estimate."
            if np.any(self.terminals.take(initial_indices[:-1], mode='wrap')):
                continue

            # Add the state transition to the response.
            states[count] = self.states.take(initial_indices, axis=0, mode='wrap')
            actions[count] = self.actions.take([end_index], mode='wrap')[0]
            rewards[count] = self.rewards.take([end_index], mode='wrap')[0]
            terminals[count] = self.terminals.take([end_index], mode='wrap')[0]
            next_states[count] = self.states.take(transition_indices, axis=0, mode='wrap')
            count += 1

        return states.astype(theano.config.floatX), \
               actions, \
               rewards.astype(theano.config.floatX), \
               next_states.astype(theano.config.floatX), \
               terminals

