
import numpy as np
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import learning_utils


class TestMakeHeatMap(unittest.TestCase):

    def test_make_heat_map(self):
        filepath = '/Users/wulfe/Dropbox/School/Stanford/winter_2016/cs239/project/hierarchical_rl/logs/rqn_4_step_stacked_2roomx5x5_row_col/value_image.txt'
        epoch = 1
        learning_utils.make_heat_map(filepath, epoch)

if __name__ == '__main__':
    unittest.main()