
import collections
import copy
import numpy as np
import random
import sys

###########################################################################

class MDP(object):

    def get_start_state(self): 
        raise NotImplementedError("Override me")

    def get_actions(self): 
        raise NotImplementedError("Override me")

    def succ_prob_reward(self, state, action): 
        """
        :description: returns a _list_ of tuples containing (next_state, probability, reward). Where the probability denotes the probability of the next_state and reward.
        """
        raise NotImplementedError("Override me")

    def get_discount(self): 
        raise NotImplementedError("Override me")

    def compute_states(self):
        self.states = set()
        queue = []
        self.states.add(self.start_state)
        queue.append(self.start_state)
        while len(queue) > 0:
            state = queue.pop()
            for action in self.actions(state):
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)

###########################################################################

class LineMDP(MDP):
    """
    :description: A line mdp is just an x axis. Here the rewards are all -1 except for the last state on the right which is +1.
    """

    EXIT_REWARD = 1
    MOVE_REWARD = -.01

    def __init__(self, length):
        self.length = length

    def get_start_state(self):
        return 0

    def get_actions(self):
        return [-1, 1]

    def get_discount(self): 
        return 1

    def succ_prob_reward(self, state, action): 
        if state == self.length:
            return []

        next_state = max(-self.length, state + action)
        reward = 1 if next_state == self.length else -1
        return [(next_state, 1, reward)]

    def print_v(self, V):
        line = ['-'] * (self.length * 2)
        for vidx, lidx in zip(range(-self.length, self.length), range(self.length * 2)):
            if vidx in V:
                line[lidx] = round(V[vidx], 2)
        print line

    def print_pi(self, pi):
        line = ['-'] * (self.length * 2)
        for pidx, lidx in zip(range(-self.length, self.length), range(self.length * 2)):
            if pidx in pi:
                line[lidx] = round(pi[pidx], 2)
        print line

###########################################################################