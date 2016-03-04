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

class TestSequenceReplayMemorySampleBatch(unittest.TestCase):

    def test_minibatch_sample_shapes_1D_state_sequence_length_1(self):
        batch_size = 100
        state_shape = 2
        sequence_length = 1
        capacity = 1000
        rm = replay_memory.SequenceReplayMemory(state_shape, sequence_length, batch_size, capacity)
        for idx in range(1000):
            state = np.ones(state_shape)
            action = 0
            reward = 0
            next_state = np.ones(state_shape)
            terminal = False
            rm.store(state, action, reward, terminal)

        states, actions, rewards, next_states, terminals = rm.sample_batch()
        self.assertEquals(states.shape, (batch_size, sequence_length, state_shape))
        self.assertEquals(actions.shape, (batch_size, 1))
        self.assertEquals(rewards.shape, (batch_size, 1))
        self.assertEquals(next_states.shape, (batch_size, sequence_length, state_shape))
        self.assertEquals(terminals.shape, (batch_size, 1))

    def test_minibatch_sample_shapes_1D_state_sequence_length_2(self):
        batch_size = 10
        state_shape = 2
        sequence_length = 2
        capacity = 1000
        rm = replay_memory.SequenceReplayMemory(state_shape, sequence_length, batch_size, capacity)
        for idx in range(1000):
            state = np.ones(state_shape)
            action = 0
            reward = 0
            next_state = np.ones(state_shape)
            terminal = False
            rm.store(state, action, reward, terminal)

        states, actions, rewards, next_states, terminals = rm.sample_batch()
        self.assertEquals(states.shape, (batch_size, sequence_length, state_shape))
        self.assertEquals(states.sum(), batch_size * sequence_length * state_shape)
        self.assertEquals(actions.shape, (batch_size, 1))
        self.assertEquals(rewards.shape, (batch_size, 1))
        self.assertEquals(next_states.shape, (batch_size, sequence_length, state_shape))
        self.assertEquals(next_states.sum(), batch_size * sequence_length * state_shape)
        self.assertEquals(terminals.shape, (batch_size, 1))

    def test_minibatch_sample_shapes_1D_state_terminal(self):
        batch_size = 200
        state_shape = 2
        sequence_length = 2
        capacity = 1000
        rm = replay_memory.SequenceReplayMemory(state_shape, sequence_length, batch_size, capacity)
        prev_state_terminal = False
        for idx in range(1, 1001):
            action = 0
            reward = 0
            state = np.ones(state_shape) * idx
            state = state if not prev_state_terminal else np.zeros(state_shape)
            prev_state_terminal = False if np.random.random() < .8 else True
            rm.store(state, action, reward, prev_state_terminal)

        states, actions, rewards, next_states, terminals = rm.sample_batch()
        for state, next_state, terminal in zip(states, next_states, terminals):
            if terminal:
                self.assertEquals(next_state.tolist()[-1], np.zeros(state_shape).tolist())

    def test_minibatch_sample_shapes_multidimensional_state_sequence_length_1(self):
        batch_size = 100
        state_shape = (1,2,2)
        sequence_length = 1
        capacity = 1000
        rm = replay_memory.SequenceReplayMemory(state_shape, sequence_length, batch_size, capacity)
        for idx in range(1000):
            state = np.ones(state_shape)
            action = 0
            reward = 0
            next_state = np.ones(state_shape)
            terminal = False
            rm.store(state, action, reward, terminal)

        states, actions, rewards, next_states, terminals = rm.sample_batch()
        expected_states_shape = (batch_size,) + (sequence_length,) + state_shape

        self.assertEquals(states.shape, expected_states_shape)
        self.assertEquals(actions.shape, (batch_size, 1))
        self.assertEquals(rewards.shape, (batch_size, 1))
        self.assertEquals(next_states.shape, expected_states_shape)
        self.assertEquals(terminals.shape, (batch_size, 1))

    def test_minibatch_sample_shapes_multidimensional_state_sequence_length_2(self):
        batch_size = 100
        state_shape = (1,2,2)
        sequence_length = 2
        capacity = 1000
        rm = replay_memory.SequenceReplayMemory(state_shape, sequence_length, batch_size, capacity)
        for idx in range(1000):
            state = np.ones(state_shape)
            action = 0
            reward = 0
            next_state = np.ones(state_shape)
            terminal = False
            rm.store(state, action, reward, terminal)

        states, actions, rewards, next_states, terminals = rm.sample_batch()
        expected_states_shape = (batch_size,) + (sequence_length,) + state_shape

        self.assertEquals(states.shape, expected_states_shape)
        self.assertEquals(actions.shape, (batch_size, 1))
        self.assertEquals(rewards.shape, (batch_size, 1))
        self.assertEquals(next_states.shape, expected_states_shape)
        self.assertEquals(terminals.shape, (batch_size, 1))


    def test_minibatch_sample_shapes_multidimensional_state_broadcast_check(self):
        batch_size = 100
        state_shape = (1,2,1)
        sequence_length = 2
        capacity = 1000
        rm = replay_memory.SequenceReplayMemory(state_shape, sequence_length, batch_size, capacity)
        for idx in range(1000):
            state = np.ones(state_shape)
            action = 0
            reward = 0
            next_state = np.ones(state_shape)
            terminal = False 
            rm.store(state, action, reward, terminal)

        states, actions, rewards, next_states, terminals = rm.sample_batch()
        expected_states_shape = (batch_size,) + (sequence_length,) + state_shape

        self.assertEquals(states.shape, expected_states_shape)
        self.assertEquals(actions.shape, (batch_size, 1))
        self.assertEquals(rewards.shape, (batch_size, 1))
        self.assertEquals(next_states.shape, expected_states_shape)
        self.assertEquals(terminals.shape, (batch_size, 1))

class TestSequenceReplayMemoryMakeLastSequence(unittest.TestCase):

    def test_make_last_sequence_basic_operation(self):
        batch_size = 10
        state_shape = 2
        sequence_length = 3
        capacity = 30
        rm = replay_memory.SequenceReplayMemory(state_shape, sequence_length, batch_size, capacity)

        for idx in range(4):
            state = np.ones(state_shape)
            action = 0
            reward = 0
            next_state = np.ones(state_shape)
            terminal = False
            rm.store(state, action, reward, terminal)

        actual = rm.make_last_sequence(np.arange(state_shape)).tolist()
        expected = [[1, 1], [1, 1], [0, 1]]
        self.assertEquals(actual, expected)

    def test_make_last_sequence_preceding_state_terminal(self):
        batch_size = 10
        state_shape = 2
        sequence_length = 3
        capacity = 30
        rm = replay_memory.SequenceReplayMemory(state_shape, sequence_length, batch_size, capacity)

        state = np.ones(state_shape)
        action = 0
        reward = 0
        next_state = np.ones(state_shape)
        terminal = False
        rm.store(state, action, reward, terminal)
        terminal = True
        rm.store(state, action, reward, terminal)
        actual = rm.make_last_sequence(np.arange(state_shape)).tolist()
        expected = [[0, 0], [0, 0], [0, 1]]
        self.assertEquals(actual, expected)

    def test_make_last_sequence_some_previous_state_terminal_not_in_sequence(self):
        batch_size = 10
        state_shape = 2
        sequence_length = 3
        capacity = 30
        rm = replay_memory.SequenceReplayMemory(state_shape, sequence_length, batch_size, capacity)

        state = np.ones(state_shape)
        action = 0
        reward = 0
        next_state = np.ones(state_shape)
        terminal = True
        rm.store(state, action, reward, terminal)
        terminal = False
        for idx in range(10):
            rm.store(state, action, reward, terminal)

        actual = rm.make_last_sequence(np.arange(state_shape)).tolist()
        expected = [[1, 1], [1, 1], [0, 1]]
        self.assertEquals(actual, expected)

    def test_make_last_sequence_terminal_state_within_sequence_but_not_preceding(self):
        batch_size = 10
        state_shape = 2
        sequence_length = 4
        capacity = 30
        rm = replay_memory.SequenceReplayMemory(state_shape, sequence_length, batch_size, capacity)

        # tuple 1
        state = np.ones(state_shape)
        action = 0
        reward = 0
        next_state = np.ones(state_shape)
        terminal = False
        rm.store(state, action, reward, terminal)

        # tuple 2
        terminal = True
        rm.store(state, action, reward, terminal)

        # tuple 3
        terminal = False
        rm.store(state, action, reward, terminal)

        actual = rm.make_last_sequence(np.arange(state_shape)).tolist()
        expected = [[0, 0], [0, 0], [1, 1], [0, 1]]
        self.assertEquals(actual, expected)

    def test_make_last_sequence_terminal_state_first_in_made_sequence(self):
        batch_size = 10
        state_shape = 2
        sequence_length = 4
        capacity = 30
        rm = replay_memory.SequenceReplayMemory(state_shape, sequence_length, batch_size, capacity)

        # tuple 1
        state = np.ones(state_shape)
        action = 0
        reward = 0
        next_state = np.ones(state_shape)
        terminal = True
        rm.store(state, action, reward, terminal)

        # tuple 2
        terminal = False
        rm.store(state, action, reward, terminal)

        # tuple 3
        terminal = False
        rm.store(state, action, reward, terminal)

        actual = rm.make_last_sequence(np.arange(state_shape)).tolist()
        expected = [[0, 0], [1, 1], [1, 1], [0, 1]]
        self.assertEquals(actual, expected)

    def test_make_last_sequence_terminal_state_first_in_made_sequence_wrap(self):
        batch_size = 10
        state_shape = 2
        sequence_length = 4
        capacity = 30
        rm = replay_memory.SequenceReplayMemory(state_shape, sequence_length, batch_size, capacity)

        # tuple 1
        state = np.ones(state_shape)
        action = 0
        reward = 0
        next_state = np.ones(state_shape)
        terminal = False
        for i in range(capacity - 1):
            rm.store(state, action, reward, terminal)


        terminal = True
        rm.store(state, action, reward, terminal)

        # tuple 2
        terminal = False
        rm.store(state, action, reward, terminal)

        # tuple 3
        terminal = False
        rm.store(state, action, reward, terminal)

        actual = rm.make_last_sequence(np.arange(state_shape)).tolist()
        expected = [[0, 0], [1, 1], [1, 1], [0, 1]]
        self.assertEquals(actual, expected)


    def test_make_last_sequence_insufficient_samples_for_full_sequence(self):
        batch_size = 10
        state_shape = 2
        sequence_length = 4
        capacity = 30
        rm = replay_memory.SequenceReplayMemory(state_shape, sequence_length, batch_size, capacity)

        # tuple 1
        state = np.ones(state_shape)
        action = 0
        reward = 0
        next_state = np.ones(state_shape)
        terminal = False
        rm.store(state, action, reward, terminal)

        # tuple 2
        terminal = False
        rm.store(state, action, reward, terminal)

        actual = rm.make_last_sequence(np.arange(state_shape)).tolist()
        expected = [[0, 0], [1, 1], [1, 1], [0, 1]]
        self.assertEquals(actual, expected)

    def test_make_last_sequence_empty(self):
        batch_size = 10
        state_shape = 2
        sequence_length = 4
        capacity = 30
        rm = replay_memory.SequenceReplayMemory(state_shape, sequence_length, batch_size, capacity)

        actual = rm.make_last_sequence(np.arange(state_shape)).tolist()
        expected = [[0, 0], [0, 0], [0, 0], [0, 1]]
        self.assertEquals(actual, expected)
        

if __name__ == '__main__':
    unittest.main()