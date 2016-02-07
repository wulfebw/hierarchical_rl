import lasagne
import numpy as np
import os
import shutil
import sys
import theano
import theano.tensor as T
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import agent
import experiment
import mdps
import policy
import qnetwork
import replay_memory

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
        # loss is still not zero because the selected action might not be the maximum value action
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

        states = np.zeros((1,2))
        actions = np.zeros((1,1), dtype='int32')
        rewards = np.zeros((1,1))
        next_states = np.zeros((1,2))
        terminals = np.zeros((1,1), dtype='int32')

        loss = network.train(states, actions, rewards, next_states, terminals)
        actual = loss
        expected = 2
        self.assertTrue(actual < expected)

    def test_loss_with_nonzero_reward_same_next_state_is_nonzero(self):
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

        lasagne.layers.helper.set_all_param_values(network.l_out, 0)

        states = np.ones((1,2), dtype=float)
        actions = np.zeros((1,1), dtype='int32')
        rewards = np.ones((1,1), dtype='int32')
        next_states = np.ones((1,2), dtype=float)
        terminals = np.zeros((1,1), dtype='int32')

        loss = network.train(states, actions, rewards, next_states, terminals)
        actual = loss
        expected = 1
        self.assertEquals(actual, expected)

class TestQNetworkFullOperation(unittest.TestCase):

    def test_qnetwork_solves_small_mdp(self):
        room_size = 5
        mdp = mdps.MazeMDP(room_size, 1)
        mdp.compute_states()
        mdp.EXIT_REWARD = 1
        mdp.MOVE_REWARD = -0.1
        discount = mdp.get_discount()
        num_actions = len(mdp.get_actions(None))
        mean_state_values = mdp.get_mean_state_values()
        batch_size = 100
        network = qnetwork.QNetwork(input_shape=2, batch_size=batch_size, num_actions=4, num_hidden=10, discount=discount, learning_rate=1e-3, update_rule='adam', freeze_interval=1000, rng=None)
        p = policy.EpsilonGreedy(num_actions, 0.5, 0.05, 10000)
        rm = replay_memory.ReplayMemory(batch_size)
        a = agent.NeuralAgent(network=network, policy=p, replay_memory=rm, 
                mean_state_values=mean_state_values, logging=True)
        num_epochs = 160
        epoch_length = 5
        test_epoch_length = 0
        max_steps = 100
        run_tests = False
        e = experiment.Experiment(mdp, a, num_epochs, epoch_length, test_epoch_length, max_steps, run_tests, value_logging=True)
        e.run()

        states = []
        for ridx in range(room_size):
            for cidx in range(room_size):
                states.append(np.array((ridx, cidx)))

        for state in states:
            q_values = network.get_q_values(state)
            self.assertAlmostEqual(q_values.tolist(), np.ones(num_actions))

if __name__ == '__main__':
    unittest.main()