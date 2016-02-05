import numpy as np
import os
import shutil
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import logger

class TestMazeMDP(unittest.TestCase):

    def test_log_epoch_empty_log(self):
        l = logger.Logger(agent_name='test')
        l.log_epoch(epoch=0)
        log_dir = l.log_dir
        self.assertTrue(os.path.isfile(os.path.join(log_dir, 'actions_epoch_0.npz')))
        self.assertTrue(os.path.isfile(os.path.join(log_dir, 'rewards_epoch_0.npz')))
        self.assertTrue(os.path.isfile(os.path.join(log_dir, 'losses_epoch_0.npz')))
        shutil.rmtree(log_dir)



if __name__ == '__main__':
    unittest.main()
