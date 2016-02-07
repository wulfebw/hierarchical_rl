import numpy as np
import os
import shutil
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import neural_agent
import experiment
import mdps
import qnetwork


class TestNeuralAgent(unittest.TestCase):

    def test_agent(self):
        mdp = mdps.MazeMDP(5, 1)
        mdp.compute_states()
        input_shape = np.shape(mdp.get_start_state())
        mdp.EXIT_REWARD = 1
        mdp.MOVE_REWARD = -0.1
        num_actions = len(mdp.get_actions(None))
        mean_state_values = mdp.get_mean_state_values()
        network = qnetwork.QNetwork()
        a = neural_agent.NeuralAgent(num_actions=num_actions, network=network, input_shape=input_shape,exploration_prob=.5, replay_memory_capacity=10000, batch_size=10, exploration_prob_reduction_steps=100000, min_exploration_prob=0.05, mean_state_values=mean_state_values, logging=True)
        num_epochs = 2
        epoch_length = 10
        test_epoch_length = 0
        max_steps = 10
        run_tests = False
        e = experiment.Experiment(mdp, a, num_epochs, epoch_length, test_epoch_length, max_steps, run_tests, value_logging=True)
        e.run()

if __name__ == '__main__':
    unittest.main()