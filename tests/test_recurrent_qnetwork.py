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
import file_utils
import learning_utils
import logger
import mdps
import policy
import recurrent_qnetwork
import replay_memory
import state_adapters

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
        network_type = 'single_layer_rnn'
        rng = None
        network = recurrent_qnetwork.RecurrentQNetwork(input_shape, 
                    sequence_length, batch_size, num_actions, num_hidden, 
                    discount, learning_rate, regularization, update_rule, 
                    freeze_interval, network_type, rng)

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
        network_type = 'single_layer_rnn'
        rng = None
        network = recurrent_qnetwork.RecurrentQNetwork(input_shape, 
                    sequence_length, batch_size, num_actions, num_hidden, 
                    discount, learning_rate, regularization, update_rule, 
                    freeze_interval, network_type, rng)

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
        network_type = 'single_layer_rnn'
        rng = None
        network = recurrent_qnetwork.RecurrentQNetwork(input_shape, 
                    sequence_length, batch_size, num_actions, num_hidden, 
                    discount, learning_rate, regularization, update_rule, 
                    freeze_interval, network_type, rng)

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
        network_type = 'single_layer_rnn'
        rng = None
        network = recurrent_qnetwork.RecurrentQNetwork(input_shape, 
                    sequence_length, batch_size, num_actions, num_hidden, 
                    discount, learning_rate, regularization, update_rule, 
                    freeze_interval, network_type, rng)

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
        expected = 5.0
        self.assertEquals(actual, expected)

    def test_loss_not_impacted_by_hid_init(self):
        input_shape = 2
        batch_size = 10
        sequence_length = 1
        num_actions = 4
        num_hidden = 10
        discount = 1
        learning_rate = 0 
        update_rule = 'sgd'
        freeze_interval = 1000
        regularization = 1e-4
        network_type = 'single_layer_rnn'
        rng = None
        network = recurrent_qnetwork.RecurrentQNetwork(input_shape, 
                    sequence_length, batch_size, num_actions, num_hidden, 
                    discount, learning_rate, regularization, update_rule, 
                    freeze_interval, network_type, rng)

        values = np.array(lasagne.layers.helper.get_all_param_values(network.l_out)) * 0
        lasagne.layers.helper.set_all_param_values(network.l_out, values)
        lasagne.layers.helper.set_all_param_values(network.next_l_out, values)

        states = np.ones((10,1,2), dtype=float)
        actions = np.zeros((10,1), dtype='int32')
        rewards = np.ones((10,1), dtype='int32')
        next_states = np.ones((10,1,2), dtype=float)
        terminals = np.zeros((10,1), dtype='int32')

        loss_before_q_values = network.train(states, actions, rewards, next_states, terminals)

        state = np.ones((1,1,2), dtype=float)
        q_values_without_hid_init = network.get_q_values(state).tolist()

        loss_after_q_values = network.train(states, actions, rewards, next_states, terminals)

        self.assertEquals(loss_before_q_values, loss_after_q_values)

class TestRecurrentQNetworkGetQValues(unittest.TestCase):
    
    def test_get_q_values_hid_init_impacts_q_values(self):
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
        network_type = 'single_layer_rnn'
        rng = None
        network = recurrent_qnetwork.RecurrentQNetwork(input_shape, 
                    sequence_length, batch_size, num_actions, num_hidden, 
                    discount, learning_rate, regularization, update_rule, 
                    freeze_interval, network_type, rng)

        state = np.ones((1,1,2), dtype=float)
        q_values_without_hid_init = network.get_q_values(state).tolist()
        q_values_with_hid_init = network.get_q_values(state).tolist()
        self.assertNotEquals(q_values_without_hid_init, q_values_with_hid_init)

    def test_get_q_values_hid_init_does_not_impact_q_values(self):
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
        network_type = 'single_layer_rnn'
        rng = None
        network = recurrent_qnetwork.RecurrentQNetwork(input_shape, 
                    sequence_length, batch_size, num_actions, num_hidden, 
                    discount, learning_rate, regularization, update_rule, 
                    freeze_interval, network_type, rng)

        state = np.ones((1,1,2), dtype=float)
        network.finish_episode()
        q_values_without_hid_init = network.get_q_values(state).tolist()
        network.finish_episode()
        q_values_after_hid_init = network.get_q_values(state).tolist()
        self.assertEquals(q_values_without_hid_init, q_values_after_hid_init)

    def test_initial_q_values(self):
        # if just one of these is 1, (or two are 1) why does a pattern arise?
        input_shape = 20
        batch_size = 10
        sequence_length = 2
        num_actions = 4
        num_hidden = 4
        discount = 1
        learning_rate = 1e-2 
        update_rule = 'adam'
        freeze_interval = 1000
        regularization = 1e-4
        network_type = 'single_layer_lstm'
        rng = None
        network = recurrent_qnetwork.RecurrentQNetwork(input_shape, 
                    sequence_length, batch_size, num_actions, num_hidden, 
                    discount, learning_rate, regularization, update_rule, 
                    freeze_interval, network_type, rng)

        values = []
        for r in range(10):
            row_values = []
            for c in range(10):
                r_state = np.zeros(10, dtype=float)
                c_state = np.zeros(10, dtype=float)
                r_state[r] = 1
                c_state[c] = 1
                state = np.hstack((r_state, c_state))
                max_q_value = max(network.get_q_values(state).tolist())
                row_values.append(max_q_value)
            values.append(row_values)


    # why is cell init nonzero?
    def test_for_zero_cell_init_with_len_1_sequences(self):
        input_shape = 2
        batch_size = 2
        sequence_length = 1
        num_actions = 2
        num_hidden = 1
        discount = 1
        learning_rate = 1
        update_rule = 'adam'
        freeze_interval = 1
        regularization = 1e-4
        network_type = 'single_layer_lstm'
        rng = None
        network = recurrent_qnetwork.RecurrentQNetwork(input_shape, 
                    sequence_length, batch_size, num_actions, num_hidden, 
                    discount, learning_rate, regularization, update_rule, 
                    freeze_interval, network_type, rng)

        print 'BEFORE'
        params = lasagne.layers.get_all_params(network.l_out)
        param_values = lasagne.layers.get_all_param_values(network.l_out)
        for p, v in zip(params, param_values):
            print p
            print v 
            print '\n'

        states = np.ones((batch_size, sequence_length, input_shape))
        actions = np.ones((batch_size, 1), dtype='int32')
        rewards = np.ones((batch_size, 1))
        next_states = np.ones((batch_size, sequence_length, input_shape))
        terminals = np.zeros((batch_size, 1), dtype='int32')
        network.train(states, actions, rewards, next_states, terminals)

        print 'AFTER 1'
        params = lasagne.layers.get_all_params(network.l_out)
        param_values = lasagne.layers.get_all_param_values(network.l_out)
        for p, v in zip(params, param_values):
            print p
            print v 
            print '\n'

        network.train(states, actions, rewards, next_states, terminals)

        print 'AFTER 2'
        params = lasagne.layers.get_all_params(network.l_out)
        param_values = lasagne.layers.get_all_param_values(network.l_out)
        for p, v in zip(params, param_values):
            print p
            print v 
            print '\n'

class TestRecurrentQNetworkSaturation(unittest.TestCase):
    
    def test_negative_saturation_rnn(self):
        input_shape = 2
        batch_size = 2
        sequence_length = 2
        num_actions = 2
        num_hidden = 1
        discount = 1
        learning_rate = 1
        update_rule = 'adam'
        freeze_interval = 1
        regularization = 1e-4
        network_type = 'single_layer_rnn'
        rng = None
        network = recurrent_qnetwork.RecurrentQNetwork(input_shape, 
                    sequence_length, batch_size, num_actions, num_hidden, 
                    discount, learning_rate, regularization, update_rule, 
                    freeze_interval, network_type, rng)

        reward_multiplier = -10000

        for idx in range(100):
            states = np.ones((batch_size, sequence_length, input_shape))

            action_multiplier = random.choice([0,1])
            actions = np.ones((batch_size, 1), dtype='int32') * action_multiplier
            rewards = np.ones((batch_size, 1)) * reward_multiplier
            next_states = np.ones((batch_size, sequence_length, input_shape))
            terminals = np.zeros((batch_size, 1), dtype='int32')
            network.train(states, actions, rewards, next_states, terminals)

        q_values = network.get_q_values(states[0]).tolist()
        print q_values
        self.assertTrue(sum(q_values) < 0)

    def test_negative_saturation_lstm(self):
        input_shape = 2
        batch_size = 2
        sequence_length = 2
        num_actions = 2
        num_hidden = 1
        discount = 1
        learning_rate = 1
        update_rule = 'adam'
        freeze_interval = 1
        regularization = 1e-4
        network_type = 'single_layer_lstm'
        rng = None
        network = recurrent_qnetwork.RecurrentQNetwork(input_shape, 
                    sequence_length, batch_size, num_actions, num_hidden, 
                    discount, learning_rate, regularization, update_rule, 
                    freeze_interval, network_type, rng)

        reward_multiplier = -10000

        for idx in range(100):
            states = np.ones((batch_size, sequence_length, input_shape))

            action_multiplier = random.choice([0,1])
            actions = np.ones((batch_size, 1), dtype='int32') * action_multiplier
            rewards = np.ones((batch_size, 1)) * reward_multiplier
            next_states = np.ones((batch_size, sequence_length, input_shape))
            terminals = np.zeros((batch_size, 1), dtype='int32')
            network.train(states, actions, rewards, next_states, terminals)

        # all the params in the lstm layer become positive
        # all the params in linear output layer become negative
        # params = lasagne.layers.get_all_params(network.l_out)
        # param_values = lasagne.layers.get_all_param_values(network.l_out)
        # for p, v in zip(params, param_values):
        #     print p
        #     print v 
        #     print '\n'

        q_values = network.get_q_values(states[0]).tolist()
        self.assertTrue(sum(q_values) < 0)

    def test_positive_saturation_lstm(self):
        input_shape = 2
        batch_size = 2
        sequence_length = 2
        num_actions = 2
        num_hidden = 1
        discount = 1
        learning_rate = 1
        update_rule = 'adam'
        freeze_interval = 1
        regularization = 1e-4
        network_type = 'single_layer_lstm'
        rng = None
        network = recurrent_qnetwork.RecurrentQNetwork(input_shape, 
                    sequence_length, batch_size, num_actions, num_hidden, 
                    discount, learning_rate, regularization, update_rule, 
                    freeze_interval, network_type, rng)

        reward_multiplier = 10000

        for idx in range(100):
            states = np.ones((batch_size, sequence_length, input_shape))

            action_multiplier = random.choice([0,1])
            actions = np.ones((batch_size, 1), dtype='int32') * action_multiplier
            rewards = np.ones((batch_size, 1)) * reward_multiplier
            next_states = np.ones((batch_size, sequence_length, input_shape))
            terminals = np.zeros((batch_size, 1), dtype='int32')
            network.train(states, actions, rewards, next_states, terminals)

        # # everything becomes positive
        # params = lasagne.layers.get_all_params(network.l_out)
        # param_values = lasagne.layers.get_all_param_values(network.l_out)
        # for p, v in zip(params, param_values):
        #     print p
        #     print v 
        #     print '\n'

        q_values = network.get_q_values(states[0]).tolist()
        self.assertTrue(sum(q_values) > 0)

@unittest.skipIf(__name__ != '__main__', "this test class does not run unless \
    this file is called directly")
class TestRecurrentQNetworkFullOperationFlattnedState(unittest.TestCase):

    def test_qnetwork_solves_small_mdp(self):

        def run(learning_rate, freeze_interval, num_hidden, reg, seq_len, eps, nt, update):
            room_size = 5
            num_rooms = 2
            input_shape = 2 * room_size
            print 'building mdp...'
            mdp = mdps.MazeMDP(room_size, num_rooms)
            mdp.compute_states()
            mdp.EXIT_REWARD = 1
            mdp.MOVE_REWARD = -0.01
            network_type = nt
            discount = 1
            sequence_length = seq_len
            num_actions = len(mdp.get_actions(None))
            batch_size = 100
            update_rule = update
            print 'building network...'
            network = recurrent_qnetwork.RecurrentQNetwork(input_shape=input_shape, 
                        sequence_length=sequence_length, batch_size=batch_size, 
                        num_actions=4, num_hidden=num_hidden, discount=discount, 
                        learning_rate=learning_rate, regularization=reg, 
                        update_rule=update_rule, freeze_interval=freeze_interval, 
                        network_type=network_type, rng=None)            

            # take this many steps because (very loosely):
            # let l be the step length
            # let d be the difference in start and end locations
            # let N be the number of steps for the agent to travel a distance d
            # then N ~ (d/l)^2  // assuming this is a random walk
            # with l = 1, this gives d^2 in order to make it N steps away
            # the desired distance here is to walk along both dimensions of the maze
            # which is equal to two times the num_rooms * room_size
            # so squaring that gives a loose approximation to the number of 
            # steps needed (discounting that this is actually a lattice (does it really matter?))
            # (also discounting the walls)
            # see: http://mathworld.wolfram.com/RandomWalk2-Dimensional.html
            max_steps = (2 * room_size * num_rooms) ** 2
            num_epochs = 350
            epoch_length = 1
            test_epoch_length = 0
            epsilon_decay = (num_epochs * epoch_length * max_steps) / 4
            print 'building adapter...'
            adapter = state_adapters.CoordinatesToSingleRoomRowColAdapter(room_size=room_size)
            print 'building policy...'
            p = policy.EpsilonGreedy(num_actions, eps, 0.05, epsilon_decay)
            print 'building replay memory...'
            # want to track at minimum the last 50 episodes
            capacity = max_steps * 50
            rm = replay_memory.SequenceReplayMemory(input_shape=input_shape,
                    sequence_length=sequence_length, batch_size=batch_size, capacity=capacity)
            print 'building logger...'
            log = logger.NeuralLogger(agent_name=network_type)
            print 'building agent...'
            a = agent.RecurrentNeuralAgent(network=network, policy=p, 
                    replay_memory=rm, log=log, state_adapter=adapter)
            run_tests = False
            print 'building experiment...'
            e = experiment.Experiment(mdp, a, num_epochs, epoch_length, 
                test_epoch_length, max_steps, run_tests, value_logging=True)
            print 'running experiment...'
            e.run()
            
            ak = file_utils.load_key('../access_key.key')
            sk = file_utils.load_key('../secret_key.key')
            bucket = 'hierarchical8'
            try:
                aws_util = aws_s3_utility.S3Utility(ak, sk, bucket)
                aws_util.upload_directory(e.agent.logger.log_dir)
            except Exception as e:
                print 'error uploading to s3: {}'.format(e)

        net_types = ['single_layer_lstm', 'stacked_lstm', 'stacked_lstm_with_merge', 'hierarchical_stacked_lstm_with_merge']
        # net_types = ['hierarchical_stacked_lstm_with_merge']
        for idx in range(50):
            lr = random.choice([.01]) 
            fi = random.choice([100])
            nh = random.choice([64]) 
            reg = random.choice([1e-4]) 
            seq_len = random.choice([13])
            eps = random.choice([.5])
            nt = net_types[idx % len(net_types)]
            up = random.choice(['sgd+nesterov'])
           
            print 'run number: {}'.format(idx)
            print 'learning_rate: {}  frozen_interval: \
            {}  num_hidden: {}  reg: {}  sequence_length: \
            {}  eps: {}  network_type: {}'.format(lr,fi,nh, reg, seq_len, eps, nt)
            run(lr, fi, nh, reg, seq_len, eps, nt, up)

if __name__ == '__main__':
    unittest.main()
