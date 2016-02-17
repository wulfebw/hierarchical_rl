import collections
import lasagne
import numpy as np
import os
import random
import shutil
import sys
import theano
import theano.tensor as T
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import agent
import experiment
import logger
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
        regularization = 0
        rng = None
        network = qnetwork.QNetwork(input_shape, batch_size, num_actions, num_hidden, discount, learning_rate, regularization, update_rule, freeze_interval, rng)

    def test_qnetwork_constructor_rmsprop(self):
        input_shape = 2
        batch_size = 100
        num_actions = 4
        num_hidden = 10
        discount = 1
        learning_rate = 1e-2 
        update_rule = 'rmsprop'
        freeze_interval = 1000
        regularization = 0
        rng = None
        network = qnetwork.QNetwork(input_shape, batch_size, num_actions, num_hidden, discount, learning_rate, regularization, update_rule, freeze_interval, rng)

    def test_qnetwork_constructor_adam(self):
        input_shape = 2
        batch_size = 100
        num_actions = 4
        num_hidden = 10
        discount = 1
        learning_rate = 1e-2 
        update_rule = 'adam'
        freeze_interval = 1000
        regularization = 0
        rng = None
        network = qnetwork.QNetwork(input_shape, batch_size, num_actions, num_hidden, discount, learning_rate, regularization, update_rule, freeze_interval, rng)

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
        regularization = 0
        rng = None
        network = qnetwork.QNetwork(input_shape, batch_size, num_actions, num_hidden, discount, learning_rate, regularization, update_rule, freeze_interval, rng)

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
        regularization = 0
        rng = None
        network = qnetwork.QNetwork(input_shape, batch_size, num_actions, num_hidden, discount, learning_rate, regularization, update_rule, freeze_interval, rng)

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
        regularization = 0
        rng = None
        network = qnetwork.QNetwork(input_shape, batch_size, num_actions, num_hidden, discount, learning_rate, regularization, update_rule, freeze_interval, rng)

        params = network.get_params()
        self.assertTrue(params is not None)

@unittest.skipIf(__name__ != '__main__', "this test class does not run unless this file is called directly")
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
        regularization = 0
        rng = None
        network = qnetwork.QNetwork(input_shape, batch_size, num_actions, num_hidden, discount, learning_rate, regularization, update_rule, freeze_interval, rng)

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
        regularization = 0
        rng = None
        network = qnetwork.QNetwork(input_shape, batch_size, num_actions, num_hidden, discount, learning_rate, regularization, update_rule, freeze_interval, rng)

        values = np.array(lasagne.layers.helper.get_all_param_values(network.l_out)) * 0
        lasagne.layers.helper.set_all_param_values(network.l_out, values)
        lasagne.layers.helper.set_all_param_values(network.next_l_out, values)

        states = np.ones((1,2), dtype=float)
        actions = np.zeros((1,1), dtype='int32')
        rewards = np.ones((1,1), dtype='int32')
        next_states = np.ones((1,2), dtype=float)
        terminals = np.zeros((1,1), dtype='int32')

        loss = network.train(states, actions, rewards, next_states, terminals)
        actual = loss
        expected = 0.5
        self.assertEquals(actual, expected)

    def test_overfit_simple_artificial_dataset(self):
        input_shape = 1
        batch_size = 10
        num_actions = 2
        num_hidden = 2
        discount = 1
        learning_rate = 1
        update_rule = 'adam'
        freeze_interval = 100
        regularization = 0
        rng = None
        network = qnetwork.QNetwork(input_shape, batch_size, num_actions, num_hidden, discount, learning_rate, regularization, update_rule, freeze_interval, rng)

        rm = replay_memory.ReplayMemory(batch_size)
        # state 0 to state 1 reward +1
        for idx in range(20):
            state = np.array([0])
            next_state = np.array([1])
            action = 1
            reward = 1
            terminal = 1
            rm.store((state, action, reward, next_state, terminal))

        # state 0 to state 0 reward -1
        for idx in range(20):
            switch = random.randint(0,1)
            state = np.array([0])
            next_state = np.array([0])
            action = 0
            reward = -1
            terminal = 0
            rm.store((state, action, reward, next_state, terminal))

        print rm.terminal_count
        print_data = False
        l = logger.Logger('test')
        counter = 0
        while True:
            counter += 1
            states, actions, rewards, next_states, terminals = rm.sample_batch()
            loss = network.train(states, actions, rewards, next_states, terminals)
            l.log_loss(loss)


            if counter % 100 == 0:
                l.log_epoch(counter)
                Q = {}
                s0 = network.get_q_values(np.array([0]))
                Q['s0_a0'] = s0[0]
                Q['s0_a1'] = s0[1]
                s1 = network.get_q_values(np.array([1]))
                Q['s1_a0'] = s1[0]
                Q['s1_a1'] = s1[1]

@unittest.skipIf(__name__ != '__main__', "this test class does not run unless this file is called directly")
class TestQNetworkFullOperationFlattnedState(unittest.TestCase):

    def test_qnetwork_solves_small_mdp(self):

        def run(learning_rate, freeze_interval, num_hidden, reg):
            room_size = 5
            num_rooms = 2
            mdp = mdps.MazeMDP(room_size, num_rooms)
            mdp.compute_states()
            mdp.EXIT_REWARD = 1
            mdp.MOVE_REWARD = -0.1
            discount = 1
            num_actions = len(mdp.get_actions(None))
            mean_state_values = mdp.get_mean_state_values()
            batch_size = 100
            network = qnetwork.QNetwork(input_shape=2 * (room_size * 
                num_rooms), batch_size=batch_size, num_actions=4, num_hidden=num_hidden, discount=discount, learning_rate=learning_rate, regularization=reg, update_rule='adam', freeze_interval=freeze_interval, rng=None)
            num_epochs = 200
            epoch_length = 50
            test_epoch_length = 0
            max_steps = (room_size * num_rooms) ** 2 
            epsilon_decay = (num_epochs * epoch_length * max_steps) / 1.5
            p = policy.EpsilonGreedy(num_actions, 0.5, 0.05, epsilon_decay)
            rm = replay_memory.ReplayMemory(batch_size, capacity=50000)
            a = agent.NeuralAgent(network=network, policy=p, replay_memory=rm, 
                    mean_state_values=mean_state_values, logging=True)
            run_tests = False
            e = experiment.Experiment(mdp, a, num_epochs, epoch_length, test_epoch_length, max_steps, run_tests, value_logging=True)
            e.run()

        for idx in range(20):
            lr = random.choice([1e-4])  # learning rate
            fi = random.choice([10000]) # freeze interval
            nh = random.choice([4]) # num hidden
            reg = random.choice([5e-4]) # regularization
            print 'run number: {}'.format(idx)
            print lr, fi, nh, reg
            run(lr, fi, nh, reg)

        states = []
        for ridx in range(5):
            for cidx in range(5):
                states.append(np.array((ridx, cidx)))

        for state in states:
            q_values = network.get_q_values(state)
            self.assertAlmostEqual(q_values.tolist(), np.ones(num_actions))

@unittest.skipIf(__name__ != '__main__', "this test class does not run unless this file is called directly")
class TestQNetworkFullOperation2DState(unittest.TestCase):

    def test_qnetwork_solves_small_mdp(self):

        def run(learning_rate, freeze_interval, num_hidden, reg):
            room_size = 5
            num_rooms = 2
            mdp = mdps.MazeMDP(room_size, num_rooms)
            mdp.compute_states()
            mdp.EXIT_REWARD = 1
            mdp.MOVE_REWARD = 0
            discount = .95
            num_actions = len(mdp.get_actions(None))
            mean_state_values = mdp.get_mean_state_values()
            batch_size = 50
            network = qnetwork.ConvQNetwork(input_shape=(10,10), batch_size=batch_size, num_actions=4, num_hidden=num_hidden, discount=discount, learning_rate=learning_rate, regularization=reg, update_rule='adam', freeze_interval=freeze_interval, rng=None)
            
            num_epochs = 2
            epoch_length = 3
            test_epoch_length = 0
            max_steps = 1000
            run_tests = False
            epsilon_decay = (num_epochs * epoch_length * max_steps) / 2
            p = policy.EpsilonGreedy(num_actions, 0.5, 0.05, epsilon_decay)
            rm = replay_memory.ReplayMemory(batch_size)
            a = agent.NeuralAgent(network=network, policy=p, replay_memory=rm, 
                    mean_state_values=mean_state_values, logging=True)
            e = experiment.Experiment(mdp, a, num_epochs, epoch_length, test_epoch_length, max_steps, run_tests, value_logging=True)
            e.run()

        for idx in range(25):
            lr = 1e-4 #np.random.random() * 10 ** np.random.uniform(-1, -3)  # learning rate
            fi = 10000 #np.random.random() * 10 ** np.random.uniform(4, 5)   # freeze interval
            nh = 32 #int(np.random.uniform(8, 32)) # num hidden
            reg = 1e-4 #np.random.random() * 10 ** np.random.uniform(-2, -5)  # regularization
            print 'run number: {}'.format(idx)
            print lr, fi, nh, reg
            run(lr, fi, nh, reg)

        states = []
        for ridx in range(5):
            for cidx in range(5):
                states.append(np.array((ridx, cidx)))

        for state in states:
            q_values = network.get_q_values(state)
            self.assertAlmostEqual(q_values.tolist(), np.ones(num_actions))

if __name__ == '__main__':
    unittest.main()