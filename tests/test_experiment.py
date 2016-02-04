import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import collections
import cv2

import unittest
import numpy as np

import experiment
import mdps
import agent

class TestExperiment(unittest.TestCase):

    def setUp(self):
        self.mdp = mdps.LineMDP(5)
        self.a = agent.TestAgent(len(self.mdp.get_actions()))
        
class TestExperimentBasicRunTests(TestExperiment):

    def test_run_basic_mdp_and_agent_episodes(self):
        num_epochs = 1
        epoch_length = 10
        test_epoch_length = 0
        max_steps = 100
        run_tests = False
        e = experiment.Experiment(self.mdp, self.a, num_epochs, epoch_length, test_epoch_length, max_steps, run_tests)
        e.run()
        actual = e.agent.episodes
        expected = e.num_epochs * e.epoch_length
        self.assertEquals(actual, expected)

    def test_run_basic_mdp_and_agent_many_episodes(self):
        num_epochs = 5
        epoch_length = 10
        test_epoch_length = 0
        max_steps = 100
        run_tests = False
        e = experiment.Experiment(self.mdp, self.a, num_epochs, epoch_length, test_epoch_length, max_steps, run_tests)
        e.run()
        actual = e.agent.episodes
        expected = e.num_epochs * e.epoch_length
        self.assertEquals(actual, expected)


