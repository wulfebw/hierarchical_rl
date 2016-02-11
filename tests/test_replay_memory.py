import numpy as np
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import replay_memory

class TestReplayMemorySampleBatch(unittest.TestCase):

    def test_minibatch_sample_shapes_1D_state(self):
        batch_size = 100
        state_shape = 2
        rm = replay_memory.ReplayMemory(batch_size)
        for idx in range(1000):
            state = np.ones(state_shape)
            action = 0
            reward = 0
            next_state = np.ones(state_shape)
            terminal = 0
            rm.store((state, action, reward, next_state, terminal))

        states, actions, rewards, next_states, terminals = rm.sample_batch()
        self.assertEquals(states.shape, (batch_size, state_shape))
        self.assertEquals(actions.shape, (batch_size, 1))
        self.assertEquals(rewards.shape, (batch_size, 1))
        self.assertEquals(next_states.shape, (batch_size, state_shape))
        self.assertEquals(terminals.shape, (batch_size, 1))

    def test_minibatch_sample_shapes_multidimensional_state(self):
        batch_size = 100
        state_shape = (1,2,2)
        rm = replay_memory.ReplayMemory(batch_size)
        for idx in range(1000):
            state = np.ones(state_shape)
            action = 0
            reward = 0
            next_state = np.ones(state_shape)
            terminal = 0
            rm.store((state, action, reward, next_state, terminal))

        states, actions, rewards, next_states, terminals = rm.sample_batch()
        expected_states_shape = (batch_size,) + state_shape

        self.assertEquals(states.shape, expected_states_shape)
        self.assertEquals(actions.shape, (batch_size, 1))
        self.assertEquals(rewards.shape, (batch_size, 1))
        self.assertEquals(next_states.shape, expected_states_shape)
        self.assertEquals(terminals.shape, (batch_size, 1))


    def test_minibatch_sample_shapes_multidimensional_state_broadcast_check(self):
        batch_size = 100
        state_shape = (1,2,1)
        rm = replay_memory.ReplayMemory(batch_size)
        for idx in range(1000):
            state = np.ones(state_shape)
            action = 0
            reward = 0
            next_state = np.ones(state_shape)
            terminal = 0
            rm.store((state, action, reward, next_state, terminal))

        states, actions, rewards, next_states, terminals = rm.sample_batch()
        expected_states_shape = (batch_size,) + state_shape

        self.assertEquals(states.shape, expected_states_shape)
        self.assertEquals(actions.shape, (batch_size, 1))
        self.assertEquals(rewards.shape, (batch_size, 1))
        self.assertEquals(next_states.shape, expected_states_shape)
        self.assertEquals(terminals.shape, (batch_size, 1))


if __name__ == '__main__':
    unittest.main()