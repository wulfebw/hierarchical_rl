"""
:description: This file contains a neural agent class that acts as an in-between for the experiment and the network
"""


import collections
import lasagne
import numpy as np
import random
import sys

import agent
import logger
import replay_memory

class NeuralAgent(agent.Agent):
    """
    :description: A class that wraps a network so it may more easily interact with an experiment. 
    """

    def __init__(self, num_actions, network, input_shape, replay_memory_capacity, batch_size, mean_state_values, logging=False):
        self.actions = range(num_actions)
        self.network = network
        self.replay_memory_capacity = replay_memory_capacity
        self.batch_size = batch_size
        self.mean_state_values = mean_state_values
        self.replay_memory = replay_memory.ReplayMemory()
        self.logger = logger.NeuralLogger(agent_name='QNetworkAgent', logging=logging)

        self.num_iters = 1
        self.prev_state = None
        self.prev_action = None
        
    def step(self, next_state, reward):
        """
        :description: the primary method of this class, which 'steps' the agent and network forward one time step. This includes selecting an action, making use of the new state and reward, and performing training.

        :type next_state: tuple or array
        :param next_state: the next state observed (i.e., s')

        :type reward: int 
        :param reward: the reward associated with having moved from the previous state to the current state

        :type rval: int
        :param rval: returns the action to next be taken within the environment
        """
        # need to transform an external state format to an internal one
        next_state = self.convert_state_to_internal_format(next_state)

        # perform training and related tasks
        self.num_iters += 1
        self.update_replay_memory(self.prev_state, self.prev_action, reward, next_state)
        self.train()

        # retrieve an action
        action = self.get_action(next_state)
        self.prev_state = next_state
        self.prev_action = action

        # log related information
        self.logger.log_reward(reward)
        self.logger.log_action(self.prev_action)

        return action

    def train(self):
        """
        :description: collects a minibatch of experiences and passes them to the network to train
        """
        # collect minibatch
        states, actions, rewards, next_states = self.replay_memory.sample_batch(self.batch_size)

        # pass to network to perform training
        loss = self.network.train(states, actions, rewards, next_states)
        self.logger.log_loss(loss)

    def get_action(self, state):
        """
        :description: gets an action given the current state. Defers to the network for selecting the action.

        :type state: numpy array
        :param state: the state used to determine the action
        """
        # the network decides what policy to follow
        return self.network.get_action(state)

    def update_replay_memory(self, state, action, reward, next_state):
        """
        :description: add a (s,a,r,s') tuple to the replay memory
        """
        sars_tuple = (state, action, reward, next_state)
        self.replay_memory.store(sars_tuple)

    def start_episode(self, state):
        """
        description: determines the first action to take and initializes internal variables
        """
        state = self.convert_state_to_internal_format(state)
        self.prev_state = state
        self.prev_action = self.get_action(state)

        self.logger.log_action(self.prev_action)
        return self.prev_action

    def finish_episode(self, next_state, reward):
        """
        :description: perform tasks at the end of episode
        """
        # todo: add a sample to replay memory
        # next_state should be none
        # when a minibatch is sampled, if next_state == None then set terminal to 1
        self.logger.finish_episode()

    def finish_epoch(self, epoch):
        """
        :description: perform tasks at the end of an epoch
        """
        self.logger.log_epoch(epoch, self.network)

    def convert_state_to_internal_format(self, state):
        """
        :description: converts a state from an extenarl format to an internal one
        """
        # converts it to an array and zero-centers its values
        return np.array(state) - self.mean_state_values

    def get_qvalues(self, state):
        """
        :description: returns the q values associated with a given state. Used for printing out a representation of the mdp with the values included. 
        """
        state = self.convert_state_to_internal_format(state)
        q_values = self.network.get_qvalues(state)
        return q_values
