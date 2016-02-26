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
import state_adapters

@unittest.skipIf(__name__ != '__main__', "this test class does not run unless this file is called directly")
class TestNeuralAgent(unittest.TestCase):

    def test_agent(self):
        room_size = 5
        mdp = mdps.MazeMDP(room_size, 1)
        mdp.compute_states()
        mdp.EXIT_REWARD = 1
        mdp.MOVE_REWARD = -0.1
        discount = mdp.get_discount()
        num_actions = len(mdp.get_actions(None))
        network = qnetwork.QNetwork(input_shape=2 * room_size, batch_size=1, num_actions=4, num_hidden=10, discount=discount, learning_rate=1e-3, update_rule='sgd', freeze_interval=10000, rng=None)
        p = policy.EpsilonGreedy(num_actions, 0.5, 0.05, 10000)
        rm = replay_memory.ReplayMemory(1)
        log = logger.NeuralLogger(agent_name='QNetwork')
        adapter = state_adapters.CoordinatesToSingleRoomRowColAdapter(room_size=room_size)
        a = agent.NeuralAgent(network=network, policy=p, replay_memory=rm, logger=log, state_adapter=adapter)
        num_epochs = 2
        epoch_length = 10
        test_epoch_length = 0
        max_steps = 10
        run_tests = False
        e = experiment.Experiment(mdp, a, num_epochs, epoch_length, test_epoch_length, max_steps, run_tests, value_logging=False)
        e.run()

if __name__ == '__main__':
    unittest.main()