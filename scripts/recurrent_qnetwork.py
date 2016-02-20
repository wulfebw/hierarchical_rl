"""
:description: This file contains the recurrent q network class. 
"""


import lasagne
from lasagne.regularization import regularize_network_params, l2
import numpy as np
import theano
import theano.tensor as T

import learning_utils

class RecurrentQNetwork(object):

    def __init__(self, input_shape, sequence_length, batch_size, num_actions, num_hidden, discount, learning_rate, regularization, update_rule, freeze_interval, rng):
        """
        :type input_shape: int
        :param input_shape: the dimension of the input representation of the state

        :type sequence_length: int
        :param sequence_length: the length to back propagate through time

        :type batch_size: int
        :param batch_size: number of samples to use in computing the loss / updates

        :type num_hidden_layers: int
        :param num_hidden_layers: number of hidden layers to use in the network

        :type num_actions: int
        :param num_actions: the output dimension of the network measured in number of possible actions

        :type num_hidden: int
        :param num_hidden: number of hidden nodes to use in each layer (const across layers)

        :type discount: float
        :param discount: discount factor to use in computing Q-learning target values

        :type learning_rate: float
        :param learning_rate: the learning rate to use (no decay schedule since ADAM update assumed) 

        :type regularization: float
        :param regularization: l2 regularization constant applied to weights

        :type update_rule: string
        :param update_rule: the type of update rule to use, suggest using 'adam'

        :type freeze_interval: int
        :param freeze_interval: the number of updates between updating the target network weights

        :type rng: rng
        :param rng: rng for running deterministically, o/w just leave as None

        :example call: 
        network = qnetwork.QNetwork(input_shape=20, batch_size=64, num_hidden_layers=2, num_actions=4, 
            num_hidden=4, discount=1, learning_rate=1e-3, regularization=1e-4, 
            update_rule='adam', freeze_interval=1e5, rng=None)

        """
        self.input_shape = input_shape
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.num_actions = num_actions
        self.num_hidden = num_hidden
        self.discount = discount
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.update_rule = update_rule
        self.freeze_interval = freeze_interval
        self.rng = rng if rng else np.random.RandomState()
        self.initialize_network()
        self.update_counter = 0

    def train(self, states, actions, rewards, next_states, terminals):
        """
        :description: Perform a q-learning update using the (s,a,r,s') tuples provided

        :type states: np.array(dtype=theano.config.floatX)
        :param states: batch of states, shape (N,D) = (batch_size, input_shape)

        :type actions: np.array(dtype='int32')
        :param actions: the actions taken by the agent in the corresponding state from states
                        shape = (N,)

        :type rewards: np.array(dtype=theano.config.floatX)
        :param rewards: rewards associated with being in state s and taking action a, shape = (N,)

        :type next_states: np.array(dtype=theano.config.floatX)
        :param next_states: batch of next_states, shape (N,D) = (batch_size, input_shape)

        :type terminals: np.array(dtype='int32')
        :param terminals: whether the corresponding state was a terminal state. If so, this
                            will cause the max_a' Q(s',a') term to be zero in the q-learning loss.

        """
        if self.update_counter % self.freeze_interval == 0:
            self.reset_target_network()
        self.update_counter += 1

        self.states_shared.set_value(states)
        self.actions_shared.set_value(actions.astype('int32'))
        self.rewards_shared.set_value(rewards)
        self.next_states_shared.set_value(next_states)
        self.terminals_shared.set_value(terminals.astype('int32'))

        loss, q_values = self._train()
        return loss

    def get_q_values(self, state):
        """
        :description: Returns the q_values associated with a single state for the purposes of 
                        deciding which action to take.

        :type state: np.array(dtype=theano.config.floatX)
        :param state: state to compute q_values for, shape = (D,)

        :example call:
        state = np.array([[1,2],[1,3]])
        network.get_q_values(state)
        """
        states = np.zeros((self.batch_size, self.sequence_length, self.input_shape), dtype=theano.config.floatX)
        states[0, :, :] = state
        self.states_shared.set_value(states)
        q_values = self._get_q_values()[0]
        return q_values

    def get_params(self):
        """
        :description: Return a numpy array containing all of the parameters of the network. 
                    Used for retrieving weights to save.
        """
        return lasagne.layers.helper.get_all_param_values(self.l_out)

    def set_params(self, params):
        """
        :description: Set the parameters of the network to the provided parameters. Used for 
                    loading saved weights.
        """
        lasagne.layers.set_all_param_values(self.l_out, params)
        self.reset_target_network()
    
    def reset_target_network(self):
        """
        :description: Set the target weights to the current weights.
        """
        all_params = lasagne.layers.helper.get_all_param_values(self.l_out)
        lasagne.layers.helper.set_all_param_values(self.next_l_out, all_params)

    ##########################################################################################
    #### Network and Learning Initialization below
    ##########################################################################################

    def initialize_network(self):
        """
        :description: this method initializes the network, updates, and theano functions for training and 
            retrieving q values. Here's an outline: 

            1. build the q network and target q network
            2. initialize theano symbolic variables used for compiling functions
            3. initialize the theano numeric variables used as input to functions
            4. formulate the symbolic loss 
            5. formulate the symbolic updates 
            6. compile theano functions for training and for getting q_values
        """
        batch_size, input_shape = self.batch_size, self.input_shape
        lasagne.random.set_rng(self.rng)

        # 1. build the q network and target q network
        self.l_out = self.build_stacked_concat_network(input_shape, self.sequence_length, batch_size, self.num_actions)

        self.next_l_out = self.build_stacked_concat_network(input_shape, self.sequence_length, batch_size, self.num_actions)
        self.reset_target_network()

        # 2. initialize theano symbolic variables used for compiling functions
        states = T.tensor3('states')
        actions = T.icol('actions')
        rewards = T.col('rewards')
        next_states = T.tensor3('next_states')
        # terminals are used to indicate a terminal state in the episode and hence a mask over the future
        # q values i.e., Q(s',a')
        terminals = T.icol('terminals')

        # 3. initialize the theano numeric variables used as input to functions
        self.states_shape = (batch_size,) + (self.sequence_length,) + (self.input_shape, )
        self.states_shared = theano.shared(np.zeros(self.states_shape, dtype=theano.config.floatX))
        self.next_states_shared = theano.shared(np.zeros(self.states_shape, dtype=theano.config.floatX))
        self.rewards_shared = theano.shared(np.zeros((batch_size, 1), dtype=theano.config.floatX), 
            broadcastable=(False, True))
        self.actions_shared = theano.shared(np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))
        self.terminals_shared = theano.shared(np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))

        # 4. formulate the symbolic loss 
        q_vals = lasagne.layers.get_output(self.l_out, states)
        next_q_vals = lasagne.layers.get_output(self.next_l_out, next_states)
        target = (rewards +
                 (T.ones_like(terminals) - terminals) *
                  self.discount * T.max(next_q_vals, axis=1, keepdims=True))
        # reshape((-1,)) == 'make a row vector', reshape((-1, 1) == 'make a column vector'
        diff = target - q_vals[T.arange(batch_size), actions.reshape((-1,))].reshape((-1, 1))

        # a lot of the recent work clips the td error at 1 so we do that here
        # the problem is that gradient backpropagating through this minimum node
        # will be zero if diff is larger then 1.0 (because changing params before
        # the minimum does not impact the output of the minimum). To account for 
        # this we take the part of the td error (magnitude) greater than 1.0 and simply
        # add it to the loss, which allows gradient to backprop but just linearly
        # in the td error rather than quadratically
        quadratic_part = T.minimum(abs(diff), 1.0)
        linear_part = abs(diff) - quadratic_part
        loss = 0.5 * quadratic_part ** 2 + linear_part
        loss = T.mean(loss) + self.regularization * regularize_network_params(self.l_out, l2)
        
        # 5. formulate the symbolic updates 
        params = lasagne.layers.helper.get_all_params(self.l_out)  
        updates = self.initialize_updates(self.update_rule, loss, params, self.learning_rate)

        # 6. compile theano functions for training and for getting q_values
        givens = {
            states: self.states_shared,
            next_states: self.next_states_shared,
            rewards: self.rewards_shared,
            actions: self.actions_shared,
            terminals: self.terminals_shared
        }
        self._train = theano.function([], [loss, q_vals], updates=updates, givens=givens)
        self._get_q_values = theano.function([], q_vals, givens={states: self.states_shared})

    def initialize_updates(self, update_rule, loss, params, learning_rate):
        if update_rule == 'adam':
            updates = lasagne.updates.adam(loss, params, learning_rate)
        elif update_rule == 'rmsprop':
            updates = lasagne.updates.rmsprop(loss, params, learning_rate)
        elif update_rule == 'sgd':
            updates = lasagne.updates.sgd(loss, params, learning_rate)
            updates = lasagne.updates.apply_nesterov_momentum(updates)
        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))
        return updates

    def build_network(self, input_shape, sequence_length, batch_size, output_shape):
        """
        :description: Build a basic LSTM RNN network.
        """

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, sequence_length, input_shape)
        )

        # l_mask = lasagne.layers.InputLayer(
        #     shape=(batch_size, sequence_length)
        # )

        l_lstm1 = lasagne.layers.LSTMLayer(
            l_in, 
            num_units=self.num_hidden, 
            #mask_input=l_mask, 
            grad_clipping=10,
            only_return_final=True
        )
        
        l_out = lasagne.layers.DenseLayer(
            l_lstm1,
            num_units=output_shape,
            nonlinearity=None,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(0)
        )

        return l_out

    def build_stacked_network(self, input_shape, sequence_length, batch_size, output_shape):
        """
        :description: Build a stack of LSTMs RNNs.
        """

        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, sequence_length, input_shape)
        )

        # l_mask = lasagne.layers.InputLayer(
        #     shape=(batch_size, sequence_length)
        # )

        l_lstm1 = lasagne.layers.LSTMLayer(
            l_in, 
            num_units=self.num_hidden, 
            #mask_input=l_mask, 
            grad_clipping=10,
            only_return_final=False
        )

        l_lstm2 = lasagne.layers.LSTMLayer(
            l_lstm1, 
            num_units=self.num_hidden, 
            #mask_input=l_mask, 
            grad_clipping=10,
            only_return_final=True
        )
        
        l_out = lasagne.layers.DenseLayer(
            l_lstm2,
            num_units=output_shape,
            nonlinearity=None,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(0)
        )

        return l_out


    def build_stacked_concat_network(self, input_shape, sequence_length, batch_size, output_shape):
        """
        :description: Build a stack of LSTM RNNs that each contribute their final output to 
                    a fully connected layer. 
        """


        l_in = lasagne.layers.InputLayer(
            shape=(batch_size, sequence_length, input_shape)
        )

        # l_mask = lasagne.layers.InputLayer(
        #     shape=(batch_size, sequence_length)
        # )

        l_lstm1 = lasagne.layers.LSTMLayer(
            l_in, 
            num_units=self.num_hidden, 
            #mask_input=l_mask, 
            grad_clipping=10,
            only_return_final=False
        )

        l_lstm2 = lasagne.layers.LSTMLayer(
            l_lstm1, 
            num_units=self.num_hidden, 
            #mask_input=l_mask, 
            grad_clipping=10,
            only_return_final=True
        )
        
        # l_lstm3 = lasagne.layers.LSTMLayer(
        #     l_lstm2, 
        #     num_units=self.num_hidden, 
        #     #mask_input=l_mask, 
        #     grad_clipping=10,
        #     only_return_final=True
        # )

        # use the output from all of the stacked lstms as input to the output layer
        l_slice1 = lasagne.layers.SliceLayer(l_lstm1, -1, 1)
        # l_slice2 = lasagne.layers.SliceLayer(l_lstm2, -1, 1)   
        # l_merge = lasagne.layers.ConcatLayer([l_slice1, l_slice2, l_lstm3])
        l_merge = lasagne.layers.ConcatLayer([l_slice1, l_lstm2])

        l_hidden1 = lasagne.layers.DenseLayer(
            l_merge,
            num_units=self.num_hidden,
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(.1)
        )

        l_out = lasagne.layers.DenseLayer(
            l_hidden1,
            num_units=output_shape,
            nonlinearity=None,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(0)
        )

        return l_out
