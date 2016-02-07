import numpy as np
import os
import shutil
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import agent
import experiment
import mdps
import policy
import qnetwork
import replay_memory

class TestNeuralAgent(unittest.TestCase):

    def test_agent(self):
        mdp = mdps.MazeMDP(5, 1)
        mdp.compute_states()
        mdp.EXIT_REWARD = 1
        mdp.MOVE_REWARD = -0.1
        discount = mdp.get_discount()
        num_actions = len(mdp.get_actions(None))
        mean_state_values = mdp.get_mean_state_values()
        network = qnetwork.QNetwork(input_shape=2, batch_size=1, num_actions=4, num_hidden=10, discount=discount, learning_rate=1e-3, update_rule='sgd', freeze_interval=10000, rng=None)
        p = policy.EpsilonGreedy(num_actions, 0.5, 0.05, 10000)
        rm = replay_memory.ReplayMemory(1)
        a = agent.NeuralAgent(network=network, policy=p, replay_memory=rm, 
                mean_state_values=mean_state_values, logging=False)
        num_epochs = 2
        epoch_length = 10
        test_epoch_length = 0
        max_steps = 10
        run_tests = False
        e = experiment.Experiment(mdp, a, num_epochs, epoch_length, test_epoch_length, max_steps, run_tests, value_logging=False)
        e.run()

if __name__ == '__main__':
    unittest.main()