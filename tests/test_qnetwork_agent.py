import numpy as np
import os
import shutil
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import qnetwork_agent
import experiment
import mdps


class TestQNetwork(unittest.TestCase):

    def test_qnetwork_action_taking(self):
        mdp = mdps.MazeMDP(5, 1)
        mdp.compute_states()
        input_shape = np.shape(mdp.get_start_state())
        mdp.EXIT_REWARD = 1
        mdp.MOVE_REWARD = -0.1
        num_actions = len(mdp.get_actions(None))
        discount = mdp.get_discount()
        a = qnetwork_agent.QNetworkAgent(num_actions=num_actions, input_shape=input_shape, discount=discount, exploration_prob=.5, step_size=1e-3, frozen_update_period=10000, replay_memory_capacity=10000, batch_size=10, exploration_prob_reduction_steps=100000, min_exploration_prob=0.05, logging=False)
        num_epochs = 2
        epoch_length = 10
        test_epoch_length = 0
        max_steps = 100
        run_tests = False
        e = experiment.Experiment(mdp, a, num_epochs, epoch_length, test_epoch_length, max_steps, run_tests)
        e.run()

        actual = len(set(e.agent.logger.actions))
        expected = num_actions
        self.assertEquals(actual, expected)

    def test_qnetwork_training(self):
        mdp = mdps.MazeMDP(5, 5)
        mdp.compute_states()
        input_shape = np.shape(mdp.get_start_state())
        mdp.EXIT_REWARD = 1
        mdp.MOVE_REWARD = -0.1
        num_actions = len(mdp.get_actions(None))
        discount = mdp.get_discount()
        a = qnetwork_agent.QNetworkAgent(num_actions=num_actions, input_shape=input_shape, discount=discount, exploration_prob=.5, step_size=1e-3, frozen_update_period=10000, replay_memory_capacity=10000, batch_size=200, exploration_prob_reduction_steps=1000000, min_exploration_prob=0.05, logging=True)
        num_epochs = 1
        epoch_length = 20
        test_epoch_length = 0
        max_steps = 1000
        run_tests = False
        e = experiment.Experiment(mdp, a, num_epochs, epoch_length, test_epoch_length, max_steps, run_tests)
        e.run()


        


if __name__ == '__main__':
    unittest.main()
