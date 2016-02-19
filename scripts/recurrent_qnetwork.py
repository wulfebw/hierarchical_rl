import lasagne
from lasagne.regularization import regularize_network_params, l2
import learning_utils
import numpy as np
import theano
import theano.tensor as T

class RecurrentQNetwork(object):

    def __init__(self, input_shape, sequence_length, batch_size, num_actions, num_hidden, discount, learning_rate, regularization, update_rule, freeze_interval, rng):
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
        states = np.zeros((self.batch_size, self.sequence_length, self.input_shape), dtype=theano.config.floatX)
        states[0, :, :] = state
        self.states_shared.set_value(states)
        q_values = self._get_q_values()[0]
        return q_values

    def get_params(self):
        return lasagne.layers.helper.get_all_param_values(self.l_out)

    def set_params(self, params):
        lasagne.layers.set_all_param_values(self.l_out, params)
        self.reset_target_network()
    
    def reset_target_network(self):
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
        
        l_out = lasagne.layers.DenseLayer(
            l_lstm2,
            num_units=output_shape,
            nonlinearity=None,
            W=lasagne.init.HeNormal(),
            b=lasagne.init.Constant(0)
        )

        return l_out


    def build_stacked_concat_network(self, input_shape, sequence_length, batch_size, output_shape):

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
