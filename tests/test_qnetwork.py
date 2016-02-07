import numpy as np
import os
import shutil
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import mdps
import qnetwork

class TestQNetworkConstruction(unittest.TestCase):

    def test_qnetwork_constructor_sgd(self):
        input_shape = 2
        batch_size = 100
        num_actions = 4
        num_hidden = 10
        discount = 1
        learning_rate = 1e-2 
        update_rule = 'sgd'
        freeze_interval = 1000
        rng = None
        network = qnetwork.QNetwork(input_shape, batch_size, num_actions, num_hidden, discount, learning_rate, update_rule, freeze_interval, rng)

    def test_qnetwork_constructor_rmsprop(self):
        input_shape = 2
        batch_size = 100
        num_actions = 4
        num_hidden = 10
        discount = 1
        learning_rate = 1e-2 
        update_rule = 'rmsprop'
        freeze_interval = 1000
        rng = None
        network = qnetwork.QNetwork(input_shape, batch_size, num_actions, num_hidden, discount, learning_rate, update_rule, freeze_interval, rng)

    def test_qnetwork_constructor_adam(self):
        input_shape = 2
        batch_size = 100
        num_actions = 4
        num_hidden = 10
        discount = 1
        learning_rate = 1e-2 
        update_rule = 'adam'
        freeze_interval = 1000
        rng = None
        network = qnetwork.QNetwork(input_shape, batch_size, num_actions, num_hidden, discount, learning_rate, update_rule, freeze_interval, rng)

class TestQNetworkGetQValues(unittest.TestCase):

    def test_that_q_values_are_retrievable(self):
        input_shape = 2
        batch_size = 100
        num_actions = 4
        num_hidden = 10
        discount = 1
        learning_rate = 1e-2 
        update_rule = 'sgd'
        freeze_interval = 1000
        rng = None
        network = qnetwork.QNetwork(input_shape, batch_size, num_actions, num_hidden, discount, learning_rate, update_rule, freeze_interval, rng)

        state = np.array([1,1])
        q_values = network.get_q_values(state) 
        actual = np.shape(q_values)
        expected = (num_actions,)
        self.assertEquals(actual, expected)

    def test_that_initial_values_are_all_similar(self):
        input_shape = 2
        batch_size = 100
        num_actions = 4
        num_hidden = 10
        discount = 1
        learning_rate = 1e-2 
        update_rule = 'sgd'
        freeze_interval = 1000
        rng = None
        network = qnetwork.QNetwork(input_shape, batch_size, num_actions, num_hidden, discount, learning_rate, update_rule, freeze_interval, rng)

        states = [[1,1],[-1,-1],[-1,1],[1,-1]]
        for state in states:
            q_values = network.get_q_values(state) 
            self.assertTrue(max(abs(q_values)) < 2)

class TestQNetworkGetParams(unittest.TestCase):

    def test_params_retrievable(self):
        input_shape = 2
        batch_size = 100
        num_actions = 4
        num_hidden = 10
        discount = 1
        learning_rate = 1e-2 
        update_rule = 'sgd'
        freeze_interval = 1000
        rng = None
        network = qnetwork.QNetwork(input_shape, batch_size, num_actions, num_hidden, discount, learning_rate, update_rule, freeze_interval, rng)

        params = network.get_params()
        self.assertTrue(params is not None)

class TestQNetworkTrain(unittest.TestCase):
    
    def test_loss_with_zero_reward_same_next_state_is_zero(self):
        input_shape = 2
        batch_size = 1
        num_actions = 4
        num_hidden = 10
        discount = 1
        learning_rate = 1e-2 
        update_rule = 'sgd'
        freeze_interval = 1000
        rng = None
        network = qnetwork.QNetwork(input_shape, batch_size, num_actions, num_hidden, discount, learning_rate, update_rule, freeze_interval, rng)

        states = np.ones(1,2)
        actions = np.array([0])
        rewards = np.zeros(1)
        next_states = np.ones(1,2)
        terminals = np.zeros(1)

        loss = network.train(states, actions, rewards, next_states, terminals)
        actual = loss
        expected = 0
        self.assertEquals(actual, expected)


class TestQNetworkFullOperation(unittest.TestCase):

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