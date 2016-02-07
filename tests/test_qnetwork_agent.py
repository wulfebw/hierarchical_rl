import numpy as np
import os
import shutil
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import qnetwork_agent
import experiment
import mdps

def get_V(e):
    V = {}
    for state in e.mdp.states:
        V[state] = e.agent.get_qopt(state)
    return V

class TestQNetworkAgent(unittest.TestCase):

    def test_qnetwork_initialization(self):
        mdp = mdps.MazeMDP(5, 3)
        mdp.compute_states()
        input_shape = np.shape(mdp.get_start_state())
        mdp.EXIT_REWARD = 1
        mdp.MOVE_REWARD = -0.1
        num_actions = len(mdp.get_actions(None))
        discount = mdp.get_discount()
        mean_state_values = mdp.get_mean_state_values()
        a = qnetwork_agent.QNetworkAgent(num_actions=num_actions, input_shape=input_shape, discount=discount, exploration_prob=.5, step_size=1e-2, frozen_update_period=1000, replay_memory_capacity=100000, batch_size=200, exploration_prob_reduction_steps=100000, min_exploration_prob=0.05, 
            mean_state_values=mean_state_values, logging=True)

        qvalues = []
        for state in mdp.states:
            qvalues.append(a.get_qvalues(state))
        actual = np.mean(qvalues)
        expected_min = -2
        expected_max = 2
        self.assertTrue(expected_min < actual and expected_max > actual)

    def test_qnetwork_training(self):
        mdp = mdps.MazeMDP(5, 3)
        mdp.compute_states()
        input_shape = np.shape(mdp.get_start_state())
        mdp.EXIT_REWARD = 1
        mdp.MOVE_REWARD = -0.1
        num_actions = len(mdp.get_actions(None))
        discount = mdp.get_discount()
        mean_state_values = mdp.get_mean_state_values()
        a = qnetwork_agent.QNetworkAgent(num_actions=num_actions, input_shape=input_shape, discount=discount, exploration_prob=.5, step_size=1e-2, frozen_update_period=1000, replay_memory_capacity=100000, batch_size=200, exploration_prob_reduction_steps=100000, min_exploration_prob=0.05, 
            mean_state_values=mean_state_values, logging=False)
        num_epochs = 1
        epoch_length = 1
        test_epoch_length = 0
        max_steps = 1000
        run_tests = False
        e = experiment.Experiment(mdp, a, num_epochs, epoch_length, test_epoch_length, max_steps, run_tests, value_logging=True)
        e.run()

if __name__ == '__main__':
    unittest.main()
