import numpy as np
import os
import shutil
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import mdps
import policy
import qnetwork

class TestQNetworkConstruction(unittest.TestCase):

    def test_qnetwork_constructor(self):
        input_shape = 2
        batch_size = 100
        num_actions = 4
        num_hidden = 10
        p = policy.EpsilonGreedy(num_actions, 0.5, 0.05, 100000)
        discount = 1
        learning_rate = 1e-2 
        action_selection = 'epsilon_greedy'
        update_rule = 'sgd'
        freeze_interval = 1000
        rng = None
        network = qnetwork.QNetwork(input_shape, batch_size, num_actions, num_hidden, p, discount, learning_rate, update_rule, freeze_interval, rng)

class TestQNetworkTrain(unittest.TestCase):
    pass

class TestQNetworkGetAction(unittest.TestCase):
    pass

class TestQNetworkGetQValues(unittest.TestCase):
    
    def test_that_initial_values_are_all_similar(self):
        pass
        # network = qnetwork.QNetwork()
        # state = np.ones(2)
        # qvalues = network.get_qvalues(state)

class TestQNetworkGetParams(unittest.TestCase):
    pass

class TestQNetworkOperation(unittest.TestCase):

    def test_qnetwork_solves_small_mdp(self):
        pass
        # mdp = mdps.MazeMDP(5, 1)
        # mdp.compute_states()
        # input_shape = np.shape(mdp.get_start_state())
        # num_actions = len(mdp.get_actions(None))
        # discount = mdp.get_discount()
        # mean_state_values = mdp.get_mean_state_values()
        # network = qnetwork.QNetwork()

if __name__ == '__main__':
    unittest.main()