
import lasagne
from lasagne.regularization import regularize_network_params, l2
import numpy as np
import os
import random
import sys
import theano
import theano.tensor as T
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

def build_hierachical_stacked_lstm_network_with_merge(input_shape, sequence_length, batch_size, output_shape, start=1, downsample=2):

    l_in = lasagne.layers.InputLayer(
        shape=(batch_size, sequence_length, input_shape)
    )

    default_gate = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.HeNormal(), W_hid=lasagne.init.HeNormal(),
        b=lasagne.init.Constant(0.))
    forget_gate = lasagne.layers.recurrent.Gate(
        W_in=lasagne.init.HeNormal(), W_hid=lasagne.init.HeNormal(),
        b=lasagne.init.Constant(5.))
    l_lstm1 = lasagne.layers.LSTMLayer(
        l_in, 
        num_units=10, 
        nonlinearity=lasagne.nonlinearities.tanh,
        cell=default_gate,
        ingate=default_gate,
        outgate=default_gate,
        forgetgate=forget_gate,
        grad_clipping=2,
        only_return_final=False
    )

    # does this slice out the correct values?
    l_slice1_up = lasagne.layers.SliceLayer(l_lstm1, slice(start, sequence_length, downsample), 1)

    l_lstm2 = lasagne.layers.LSTMLayer(
        l_slice1_up, 
        num_units=10, 
        nonlinearity=lasagne.nonlinearities.tanh,
        cell=default_gate,
        ingate=default_gate,
        outgate=default_gate,
        forgetgate=forget_gate,
        grad_clipping=2,
        only_return_final=True
    )

    l_slice1_out = lasagne.layers.SliceLayer(l_lstm1, -1, 1)
    l_merge = lasagne.layers.ConcatLayer([l_slice1_out, l_lstm2])
    l_out = lasagne.layers.DenseLayer(
        l_merge,
        num_units=output_shape,
        nonlinearity=None,
        W=lasagne.init.HeNormal(),
        b=lasagne.init.Constant(0)
    )

    return l_out, l_lstm1, l_slice1_up

class TestBuildHierarchicalStackedLSTMWithMerge(unittest.TestCase):

    def test_build_hierachical_stacked_lstm_network_with_merge_correct_slice(self):
        input_shape = 14
        sequence_length = 4
        batch_size = 1
        _, l_lstm, l_slice = build_hierachical_stacked_lstm_network_with_merge(
                                    input_shape=input_shape,
                                    sequence_length=sequence_length,
                                    batch_size=batch_size,
                                    output_shape=4)

        states = T.tensor3('states')
        lstm_out = lasagne.layers.get_output(l_lstm, states)
        slice_out = lasagne.layers.get_output(l_slice, states)
        run = theano.function([states], [lstm_out, slice_out])
        sample_states = np.zeros((batch_size, sequence_length, input_shape))
        sample_lstm_out, sample_slice_out = run(sample_states)

        self.assertEquals(sample_lstm_out[:, 1::2, :].tolist(), sample_slice_out.tolist())

    def test_build_hierachical_stacked_lstm_network_with_merge_correct_slice_short_seq(self):
        input_shape = 14
        sequence_length = 2
        batch_size = 1
        _, l_lstm, l_slice = build_hierachical_stacked_lstm_network_with_merge(
                                    input_shape=input_shape,
                                    sequence_length=sequence_length,
                                    batch_size=batch_size,
                                    output_shape=4)

        states = T.tensor3('states')
        lstm_out = lasagne.layers.get_output(l_lstm, states)
        slice_out = lasagne.layers.get_output(l_slice, states)
        run = theano.function([states], [lstm_out, slice_out])
        sample_states = np.zeros((batch_size, sequence_length, input_shape))
        sample_lstm_out, sample_slice_out = run(sample_states)

        self.assertEquals(sample_lstm_out[:, 1::2, :].tolist(), sample_slice_out.tolist())


    def test_build_hierachical_stacked_lstm_network_with_merge_correct_slice_len_1_seq(self):
        input_shape = 14
        sequence_length = 1
        batch_size = 1
        l_out, l_lstm, l_slice = build_hierachical_stacked_lstm_network_with_merge(
                                    input_shape=input_shape,
                                    sequence_length=sequence_length,
                                    batch_size=batch_size,
                                    output_shape=4,
                                    start=0,
                                    downsample=3)

        states = T.tensor3('states')
        l_out_out = lasagne.layers.get_output(l_out, states)
        lstm_out = lasagne.layers.get_output(l_lstm, states)
        slice_out = lasagne.layers.get_output(l_slice, states)
        run = theano.function([states], [l_out_out, lstm_out, slice_out])
        sample_states = np.zeros((batch_size, sequence_length, input_shape))
        sample_out, sample_lstm_out, sample_slice_out = run(sample_states)

        self.assertEquals(sample_lstm_out[:, 0::3, :].tolist(), sample_slice_out.tolist())

    def test_build_hierachical_stacked_lstm_network_with_merge_correct_slice_longer_len_seq(self):
        input_shape = 14
        sequence_length = 7
        batch_size = 1
        l_out, l_lstm, l_slice = build_hierachical_stacked_lstm_network_with_merge(
                                    input_shape=input_shape,
                                    sequence_length=sequence_length,
                                    batch_size=batch_size,
                                    output_shape=4,
                                    start=0,
                                    downsample=3)

        states = T.tensor3('states')
        l_out_out = lasagne.layers.get_output(l_out, states)
        lstm_out = lasagne.layers.get_output(l_lstm, states)
        slice_out = lasagne.layers.get_output(l_slice, states)
        run = theano.function([states], [l_out_out, lstm_out, slice_out])
        sample_states = np.zeros((batch_size, sequence_length, input_shape))
        sample_out, sample_lstm_out, sample_slice_out = run(sample_states)

        self.assertEquals(sample_lstm_out[:, 0::3, :].tolist(), sample_slice_out.tolist())

    def test_build_hierachical_stacked_lstm_network_with_merge_correct_slice_shared_var(self):
        input_shape = 14
        sequence_length = 1
        batch_size = 1
        _, l_lstm, l_slice = build_hierachical_stacked_lstm_network_with_merge(
                                    input_shape=input_shape,
                                    sequence_length=sequence_length,
                                    batch_size=batch_size,
                                    output_shape=4)

        states = T.tensor3('states')
        lstm_out = lasagne.layers.get_output(l_lstm, states)
        slice_out = lasagne.layers.get_output(l_slice, states)

        states_shared = theano.shared(np.zeros((batch_size, sequence_length, input_shape)))
        run = theano.function([], [lstm_out, slice_out], givens={states: states_shared})
        sample_states = np.zeros((batch_size, sequence_length, input_shape))
        states_shared.set_value(sample_states)
        sample_lstm_out, sample_slice_out = run()

        self.assertEquals(sample_lstm_out[:, 1::2, :].tolist(), sample_slice_out.tolist())

        
if __name__ == '__main__':
    unittest.main()
