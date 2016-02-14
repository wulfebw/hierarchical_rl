import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import agent
import experiment
import logger
import mdps

class TestMazeMDP(unittest.TestCase):

    def test_log_epoch_empty_log(self):
        l = logger.Logger(agent_name='test')
        l.log_epoch(epoch=0)
        log_dir = l.log_dir
        self.assertTrue(os.path.isfile(os.path.join(log_dir, 'actions.npz')))
        self.assertTrue(os.path.isfile(os.path.join(log_dir, 'rewards.npz')))
        self.assertTrue(os.path.isfile(os.path.join(log_dir, 'losses.npz')))
        shutil.rmtree(log_dir)

# class TestMovingAverage(unittest.TestCase):

#     def test_moving_average_single_item_window(self):
#         arr = [1,2,3]
#         actual = logger.moving_average(arr, 1)
#         self.assertSequenceEqual(actual, arr)

#     def test_moving_average_small_window(self):
#         arr = [1,2,3,4,5,6,7]
#         actual = logger.moving_average(arr, 2)
#         expected = [0.5, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5]
#         self.assertSequenceEqual(actual, expected)

#     def test_moving_average_small_window_large_variance(self):
#         arr = [0,9,0,9,0]
#         actual = logger.moving_average(arr, 3)
#         expected = [3, 3, 6, 3, 3]
#         self.assertSequenceEqual(actual, expected)

#     def test_moving_average_large_window_large_variance(self):
#         arr = [0,9,0,9,0]
#         actual = logger.moving_average(arr, 4)
#         expected = [2.25, 2.25, 4.5, 4.5, 2.25]
#         self.assertSequenceEqual(actual, expected)


class testLoggerGraphing(unittest.TestCase):

    def test_graphs_are_plotted_and_saved_during_experiment(self):
        mdp = mdps.MazeMDP(5, 3)
        mdp.compute_states()
        mdp.EXIT_REWARD = 1
        mdp.MOVE_REWARD = -0.1
        num_actions = len(mdp.get_actions(None))
        discount = mdp.get_discount()
        exploration_prob = .5
        step_size = 1
        a = agent.QLearningAgent(num_actions=num_actions, discount=discount, exploration_prob=exploration_prob, step_size=step_size, logging=True)
        num_epochs = 1
        epoch_length = 100
        test_epoch_length = 0
        max_steps = 1000
        run_tests = False
        e = experiment.Experiment(mdp, a, num_epochs, epoch_length, test_epoch_length, max_steps, run_tests)
        e.run()

        log_dir = e.agent.logger.log_dir
        self.assertTrue(os.path.isfile(os.path.join(log_dir, 'actions_graph.png')))
        self.assertTrue(os.path.isfile(os.path.join(log_dir, 'losses_graph.png')))
        self.assertTrue(os.path.isfile(os.path.join(log_dir, 'rewards_graph.png')))
        shutil.rmtree(log_dir)
        
if __name__ == '__main__':
    unittest.main()

