
import collections
import numpy as np
import random

import learning_utils

class Experiment(object):
    """
    :description: Experiment is a class representing an online reinforcement learning experiment. This class orchestrates the interaction between an agent and an mdp.
    """

    def __init__(self, mdp, agent, num_epochs, epoch_length, test_epoch_length, max_steps, run_tests, 
        value_logging=False):
        """
        :type mdp: object inheriting from MDP
        :param mdp: the markov decision process in which the agent acts

        :type agent: object inheriting from Agent
        :param agent: the agent that acts within the experiment

        :type num_epochs: int 
        :param num_epochs: number of training epochs to run 

        :type epoch_length: int 
        :param epoch_length: length of each epoch in episodes

        :type test_epoch_length: int 
        :param test_epoch_length: length of a test epoch in episodes

        :type max_steps: int 
        :param max_steps: maximum number of steps allowed in a single episode

        :type run_tests: boolean
        :param run_tests: whether or not to run testing epochs

        :type value_logging: boolean
        :param value_logging: whether or not to write a representation of the value function to a file
        """
        self.mdp = mdp
        self.agent = agent
        self.num_epochs = num_epochs
        self.epoch_length = epoch_length
        self.test_epoch_length = test_epoch_length
        self.max_steps = max_steps
        self.run_tests = run_tests
        self.mdp_actions = self.mdp.get_actions()
        self.value_logging = value_logging

    def run(self):
        """
        :description: main method which runs the entire experiment
        """
        for epoch in xrange(self.num_epochs):
            self.run_epoch(epoch, self.epoch_length)
            self.agent.finish_epoch(epoch)
            self.finish_epoch(epoch)

            if self.run_tests:
                self.agent.start_testing()
                self.run_epoch(self.test_epoch_length)
                self.agent.finish_testing(epoch)

        self.finish_experiement(epoch)

    def run_epoch(self, epoch, epoch_length):
        """
        :description: runs a single epoch

        :type epoch_length: int 
        :param epoch_length: length of the current epoch in episodes
        """
        for episode in xrange(epoch_length):
            self.run_episode()

    def run_episode(self):
        """
        :description: runs a single episode
        """
        state = self.mdp.get_start_state()
        action = self.agent.start_episode(state)
        reward = 0
        for step in xrange(self.max_steps):

            # get the next state and reward
            next_state, reward, terminal = self.step(state, action)

            # if episode has ended, then break
            if terminal:
                break

            # otherwise, inform the agent and get a new action
            action = self.agent.step(next_state, reward)
            state = next_state
        
        # store this experience as a terminal one regardless of the loop exit condition
        # because either way the next state will break continuity
        self.agent.finish_episode(next_state, reward)

    def step(self, state, action):
        """
        :description: progresses the experiment forward one time step
        """
        # convert to mdp action format and get transitions
        real_action = self.mdp_actions[action]
        transitions = self.mdp.succ_prob_reward(state, real_action)

        # randomly sample a transition
        i = learning_utils.sample([prob for newState, prob, reward in transitions])
        next_state, prob, reward = transitions[i]

        # if the next state is terminal note that
        terminal = False
        if self.mdp.is_end_state(next_state):
            terminal = True

        return next_state, reward, terminal

    def finish_epoch(self, epoch):
        """
        :description: finalize epoch
        """
        if self.value_logging and self.agent.replay_memory.is_full():
            self.log_value_string()
            # if epoch > 3:
            #     self.log_trajectories()

    def log_trajectories(self):
        self.agent.logger.log_trajectories(self.mdp)

    def log_value_string(self):
        """
        :description: collect the necessary components to print a representation of the optimal value 
            of each state in the mdp.
        """
        V = {}
        for state in self.mdp.states:
            V[state] = np.max(self.agent.get_q_values(state))
        value_string = self.mdp.get_value_string(V)
        self.agent.logger.log_value_string(value_string)
        self.agent.logger.log_values(V)


