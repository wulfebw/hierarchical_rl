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
import aws_s3_utility
import experiment
import learning_utils
import mdps
import policy
import recurrent_qnetwork
import replay_memory

class TestRecurrentQNetworkConstruction(unittest.TestCase):

    def test_qnetwork_constructor_sgd(self):
        input_shape = 2
        batch_size = 10
        sequence_length = 1
        num_actions = 4
        num_hidden = 10
        discount = 1
        learning_rate = 1e-2 
        update_rule = 'sgd'
        freeze_interval = 1000
        regularization = 1e-4
        rng = None
        network = recurrent_qnetwork.RecurrentQNetwork(input_shape, sequence_length, batch_size, num_actions, 
                num_hidden, discount, learning_rate, regularization, update_rule, freeze_interval, rng)


class TestRecurrentQNetworkTrain(unittest.TestCase):
    
    def test_loss_with_zero_reward_same_next_state_is_zero(self):
        input_shape = 2
        batch_size = 1
        sequence_length = 1
        num_actions = 4
        num_hidden = 5
        discount = 1
        learning_rate = 1e-2 
        update_rule = 'sgd'
        freeze_interval = 1000
        regularization = 1e-4
        rng = None
        network = recurrent_qnetwork.RecurrentQNetwork(input_shape, sequence_length, batch_size, num_actions, 
                num_hidden, discount, learning_rate, regularization, update_rule, freeze_interval, rng)

        states = np.zeros((1,1,2))
        actions = np.zeros((1,1), dtype='int32')
        rewards = np.zeros((1,1))
        next_states = np.zeros((1,1,2))
        terminals = np.zeros((1,1), dtype='int32')

        loss = network.train(states, actions, rewards, next_states, terminals)
        actual = loss
        expected = 2
        self.assertTrue(actual < expected)

    def test_loss_with_nonzero_reward_same_next_state_is_nonzero(self):
        input_shape = 2
        batch_size = 1
        sequence_length = 1
        num_actions = 4
        num_hidden = 10
        discount = 1
        learning_rate = 1e-2 
        update_rule = 'sgd'
        freeze_interval = 1000
        regularization = 1e-4
        rng = None
        network = recurrent_qnetwork.RecurrentQNetwork(input_shape, sequence_length, batch_size, num_actions, 
                num_hidden, discount, learning_rate, regularization, update_rule, freeze_interval, rng)

        values = np.array(lasagne.layers.helper.get_all_param_values(network.l_out)) * 0
        lasagne.layers.helper.set_all_param_values(network.l_out, values)
        lasagne.layers.helper.set_all_param_values(network.next_l_out, values)

        states = np.ones((1,1,2), dtype=float)
        actions = np.zeros((1,1), dtype='int32')
        rewards = np.ones((1,1), dtype='int32')
        next_states = np.ones((1,1,2), dtype=float)
        terminals = np.zeros((1,1), dtype='int32')

        loss = network.train(states, actions, rewards, next_states, terminals)
        actual = loss
        expected = 0.5
        self.assertEquals(actual, expected)

    def test_loss_with_nonzero_reward_same_next_state_is_nonzero_large_batch_size(self):
        input_shape = 2
        batch_size = 10
        sequence_length = 1
        num_actions = 4
        num_hidden = 10
        discount = 1
        learning_rate = 1e-2 
        update_rule = 'sgd'
        freeze_interval = 1000
        regularization = 1e-4
        rng = None
        network = recurrent_qnetwork.RecurrentQNetwork(input_shape, sequence_length, batch_size, num_actions, 
                num_hidden, discount, learning_rate, regularization, update_rule, freeze_interval, rng)

        values = np.array(lasagne.layers.helper.get_all_param_values(network.l_out)) * 0
        lasagne.layers.helper.set_all_param_values(network.l_out, values)
        lasagne.layers.helper.set_all_param_values(network.next_l_out, values)

        states = np.ones((10,1,2), dtype=float)
        actions = np.zeros((10,1), dtype='int32')
        rewards = np.ones((10,1), dtype='int32')
        next_states = np.ones((10,1,2), dtype=float)
        terminals = np.zeros((10,1), dtype='int32')

        loss = network.train(states, actions, rewards, next_states, terminals)
        actual = loss
        expected = 0.5
        self.assertEquals(actual, expected)

@unittest.skipIf(__name__ != '__main__', "this test class does not run unless this file is called directly")
class TestRecurrentQNetworkFullOperationFlattnedState(unittest.TestCase):

    def test_qnetwork_solves_small_mdp(self):

        def run(learning_rate, freeze_interval, num_hidden, reg, seq_len, eps):
            room_size = 5
            num_rooms = 2
            print 'building mdp...'
            mdp = mdps.MazeMDP(room_size, num_rooms)
            mdp.compute_states()
            mdp.EXIT_REWARD = 1
            mdp.MOVE_REWARD = -0.1
            discount = 1
            sequence_length = seq_len
            num_actions = len(mdp.get_actions(None))
            batch_size = int(2**6)
            print 'building network...'
            network = recurrent_qnetwork.RecurrentQNetwork(input_shape=2 * (room_size * 
                num_rooms), sequence_length=sequence_length, batch_size=batch_size, 
                num_actions=4, num_hidden=num_hidden, discount=discount, learning_rate=
                learning_rate, regularization=reg, update_rule='adam', freeze_interval=
                freeze_interval, rng=None)            
            num_epochs = 100
            epoch_length = 10
            test_epoch_length = 0
            max_steps = 2 * (room_size * num_rooms) ** 2
            epsilon_decay = (num_epochs * epoch_length * max_steps) / 2
            print 'building policy...'
            p = policy.EpsilonGreedy(num_actions, eps, 0.05, epsilon_decay)
            print 'building replay memory...'
            rm = replay_memory.SequenceReplayMemory(input_shape=2*(room_size * num_rooms),
                            sequence_length=sequence_length, batch_size=batch_size, capacity=100000)
            print 'building agent...'
            a = agent.RecurrentNeuralAgent(network=network, policy=p, replay_memory=rm, logging=True)
            run_tests = False
            print 'building experiment...'
            e = experiment.Experiment(mdp, a, num_epochs, epoch_length, test_epoch_length, 
                max_steps, run_tests, value_logging=True)
            print 'running experiment...'
            e.run()
            
            ak = ''
            sk = ''
            bucket = 'hierarchical'
            try:
                aws_util = aws_s3_utility.S3Utility(ak, sk, bucket)
                aws_util.upload_directory(e.agent.logger.log_dir)
            except Exception as e:
                print 'error uploading to s3: {}'.format(e)

        for idx in range(50):
            lr = random.choice([5e-4, 1e-4]) 
            fi = random.choice([1e3, 2.5e3, 5e3]) 
            nh = random.choice([4, 8, 12]) 
            reg = random.choice([1e-4, 5e-4]) 
            seq_len = random.choice([2 , 3, 4])
            eps = random.choice([.3, .4, .5, .6])
            print 'run number: {}'.format(idx)
            print 'learning_rate: {}\tfrozen_interval: {}\tnum_hidden: {}\treg: {}\tsequence_length: {}\teps: {}'.format(lr,fi,nh, reg, seq_len, eps)
            run(lr, fi, nh, reg, seq_len, eps)

if __name__ == '__main__':
    unittest.main()
