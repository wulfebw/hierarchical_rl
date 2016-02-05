import numpy as np
import os
import shutil
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import agent
import experiment
import mdps

def get_V(e):
    V = {}
    e.agent.exploration_prob = 0
    for state in e.mdp.states:
        qopt = max((e.agent.getQ(state, action), action) for action in e.agent.actions)[0]
        V[state] = qopt
    return V

class TestExperiment(unittest.TestCase):

    def setUp(self):
        pass
        
class TestExperimentBasicRuns(TestExperiment):

    def test_run_basic_mdp_and_agent_episodes(self):
        mdp = mdps.LineMDP(5)
        a = agent.TestAgent(len(mdp.get_actions()))
        num_epochs = 1
        epoch_length = 10
        test_epoch_length = 0
        max_steps = 100
        run_tests = False
        e = experiment.Experiment(mdp, a, num_epochs, epoch_length, test_epoch_length, max_steps, run_tests)
        e.run()
        actual = e.agent.episodes
        expected = e.num_epochs * e.epoch_length
        self.assertEquals(actual, expected)

    def test_run_basic_mdp_and_agent_many_episodes(self):
        mdp = mdps.LineMDP(5)
        a = agent.TestAgent(len(mdp.get_actions()))
        num_epochs = 5
        epoch_length = 10
        test_epoch_length = 0
        max_steps = 100
        run_tests = False
        e = experiment.Experiment(mdp, a, num_epochs, epoch_length, test_epoch_length, max_steps, run_tests)
        e.run()
        actual = e.agent.episodes
        expected = e.num_epochs * e.epoch_length
        self.assertEquals(actual, expected)

class TestExperimentMazeSolving(TestExperiment):

    def test_run_with_maze_mdp_and_working_agent_completes(self):
        mdp = mdps.MazeMDP(5, 1)
        num_actions = len(mdp.get_actions(None))
        discount = mdp.get_discount()
        exploration_prob = .3
        step_size = 1e-2
        a = agent.QLearningAgent(num_actions=num_actions, discount=discount, exploration_prob=exploration_prob, step_size=step_size, logging=False)
        num_epochs = 1
        epoch_length = 1
        test_epoch_length = 0
        max_steps = 10000
        run_tests = False
        e = experiment.Experiment(mdp, a, num_epochs, epoch_length, test_epoch_length, max_steps, run_tests)
        e.run()
        max_reward = np.max(e.agent.logger.rewards)
        self.assertEquals(max_reward, mdp.EXIT_REWARD)

    def test_run_with_small_maze_mdp_q_learning_agent_correct_V(self):
        mdp = mdps.MazeMDP(5, 1)
        mdp.compute_states()
        mdp.EXIT_REWARD = 1
        mdp.MOVE_REWARD = -0.1
        num_actions = len(mdp.get_actions(None))
        discount = mdp.get_discount()
        exploration_prob = .5
        step_size = 5e-1
        a = agent.QLearningAgent(num_actions=num_actions, discount=discount, exploration_prob=exploration_prob, step_size=step_size, logging=False)
        num_epochs = 10
        epoch_length = 300
        test_epoch_length = 0
        max_steps = 1000
        run_tests = False
        e = experiment.Experiment(mdp, a, num_epochs, epoch_length, test_epoch_length, max_steps, run_tests)
        e.run()

        V = get_V(e)
        expected = {(0,0):0.5, (1,0):0.6, (2,0):0.7, (3,0):0.8, (4,0):0.7,
                    (0,1):0.6, (1,1):0.7, (2,1):0.8, (3,1):0.9, (4,1):0.8,
                    (0,2):0.7, (1,2):0.8, (2,2):0.9, (3,2):1.0, (4,2):0.9,
                    (0,3):0.8, (1,3):0.9, (2,3):1.0, (3,3):0.0, (4,3):1.0,
                    (0,4):0.7, (1,4):0.8, (2,4):0.9, (3,4):1.0, (4,4):0.9}

        max_diff = 1e-2
        for k in expected.keys():
            self.assertTrue(k in V)
            self.assertTrue(np.abs(V[k] - expected[k]) < max_diff)

    def test_run_with_large_maze_mdp_q_learning_agent_correct_V(self):
        mdp = mdps.MazeMDP(5, 3)
        mdp.compute_states()
        mdp.EXIT_REWARD = 1
        mdp.MOVE_REWARD = -0.1
        num_actions = len(mdp.get_actions(None))
        discount = mdp.get_discount()
        exploration_prob = .5
        step_size = 1
        a = agent.QLearningAgent(num_actions=num_actions, discount=discount, exploration_prob=exploration_prob, step_size=step_size, logging=False)
        num_epochs = 10
        epoch_length = 100
        test_epoch_length = 0
        max_steps = 1000
        run_tests = False
        e = experiment.Experiment(mdp, a, num_epochs, epoch_length, test_epoch_length, max_steps, run_tests)
        e.run()

        V = get_V(e)
        actual_total = 0
        for k, v in V.iteritems():
            actual_total += v
        expected_total_min = -110
        expected_total_max = -40
        self.assertTrue(actual_total < expected_total_max)
        self.assertTrue(actual_total > expected_total_min)

if __name__ == '__main__':
    unittest.main()

