
import random

import learning_utils

class Policy(object):

    def choose_action(self, q_values):
        raise NotImplementedError("Override me")

class EpsilonGreedy(Policy):

    def __init__(self, num_actions, exploration_prob, min_exploration_prob, actions_until_min):
        self.actions = range(num_actions)
        self.exploration_prob = exploration_prob
        self.min_exploration_prob = min_exploration_prob
        self.exploration_reduction = (exploration_prob - min_exploration_prob) / actions_until_min

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
        self.actions = range(num_actions)
        self.tau = float(tau)
        self.min_tau = min_tau
        self.tau_reduction = (tau - min_tau) / actions_until_min

    def choose_action(self, q_values):
        exp_q_values = np.exp(q_values / self.tau)
        weights = dict()
        for idx, val in enumerate(exp_q_vals):
            weights[idx] = val
        action = learning_utils.weightedRandomChoice(weights)
        return action

    def update_parameters(self):
        updated_tau = self.tau - self.tau_reduction
        self.tau = max(self.min_tau, updated_tau)

