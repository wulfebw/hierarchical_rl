"""
:description: classes implementing action selection policies
"""

import numpy as np
import random

import learning_utils

class Policy(object):

    def __init__(self, num_actions):
        self.actions = range(num_actions)

    def choose_action(self, q_values):
        raise NotImplementedError("Override me")

    def random_action(self):
        return random.choice(self.actions)

class EpsilonGreedy(Policy):

    def __init__(self, num_actions, exploration_prob, min_exploration_prob, actions_until_min):
        super(EpsilonGreedy, self).__init__(num_actions)
        self.exploration_prob = exploration_prob
        self.min_exploration_prob = min_exploration_prob
        self.actions_until_min = actions_until_min
        assert actions_until_min != 0, 'actions_until_min must be positive'
        self.exploration_reduction = (exploration_prob - min_exploration_prob) / float(actions_until_min)

    def choose_action(self, q_values):
        self.update_parameters()
        if random.random() < self.exploration_prob:
            return random.choice(self.actions)
        else:
            return np.argmax(q_values)

    def update_parameters(self):
        updated_exploration_prob = self.exploration_prob - self.exploration_reduction
        self.exploration_prob = max(self.min_exploration_prob, updated_exploration_prob)

class Softmax(Policy):

    def __init__(self, num_actions, tau, min_tau, actions_until_min):
        super(Softmax, self).__init__(num_actions)
        self.tau = float(tau)
        self.min_tau = min_tau
        self.actions_until_min = actions_until_min
        assert actions_until_min != 0, 'actions_until_min must be positive'
        self.tau_reduction = (tau - min_tau) / float(actions_until_min)

    def choose_action(self, q_values):
        self.update_parameters()
        exp_q_values = np.exp(q_values / (self.tau + 1e-2))
        weights = dict()
        for idx, val in enumerate(exp_q_values):
            weights[idx] = val
        action = learning_utils.weightedRandomChoice(weights)
        return action

    def update_parameters(self):
        updated_tau = self.tau - self.tau_reduction
        self.tau = max(self.min_tau, updated_tau)

