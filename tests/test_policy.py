import numpy as np
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import policy

class TestEpsilonGreedy(unittest.TestCase):

    def test_deterministic_action_selection(self):
        p = policy.EpsilonGreedy(num_actions=4, exploration_prob=0, min_exploration_prob=0, actions_until_min=1)
        q_values = [1,2,3,4]
        actual = p.choose_action(q_values)
        expected = 3
        self.assertEquals(actual, expected)

    def test_reduction_decreses_exploration_prob(self):
        p = policy.EpsilonGreedy(num_actions=4, exploration_prob=1, min_exploration_prob=0, actions_until_min=2)
        q_values = [1,2,3,4]
        p.choose_action(q_values)
        self.assertEquals(p.exploration_prob, 0.5)

    def test_reduction_decreses_exploration_prob_completely(self):
        p = policy.EpsilonGreedy(num_actions=4, exploration_prob=1, min_exploration_prob=0, actions_until_min=2)
        q_values = [1,2,3,4]
        p.choose_action(q_values)
        p.choose_action(q_values)
        self.assertEquals(p.exploration_prob, 0)

class TestSoftmax(unittest.TestCase):

    def test_deterministic_action_selection(self):
        p = policy.Softmax(num_actions=4, tau=1e-1, min_tau=0, actions_until_min=100)
        q_values = np.array([1,2,3,4])
        actual = p.choose_action(q_values)
        expected = 3
        self.assertEquals(actual, expected)

    def test_stochastic_action_selection(self):
        p = policy.Softmax(num_actions=4, tau=1e1, min_tau=0, actions_until_min=1000)
        q_values = np.array([1,2,3,4])
        actions = []
        for i in range(1000):
            actions.append(p.choose_action(q_values))
        actions = set(actions)
        expected = 4
        self.assertEquals(len(actions), expected)

    def test_reduction_decreses_exploration_prob(self):
        p = policy.Softmax(num_actions=4, tau=1, min_tau=0, actions_until_min=2)
        q_values = np.array([1,2,3,4])
        p.choose_action(q_values)
        self.assertEquals(p.tau, 0.5)

    def test_reduction_decreses_exploration_prob_completely(self):
        p = policy.Softmax(num_actions=4, tau=1, min_tau=0, actions_until_min=2)
        q_values = np.array([1,2,3,4])
        p.choose_action(q_values)
        p.choose_action(q_values)
        self.assertEquals(p.tau, 0)

if __name__ == '__main__':
    unittest.main()