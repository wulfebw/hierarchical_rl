import numpy as np
import os
import sys
import unittest

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'scripts')))

import state_adapters

class TestCoordinatesToSingleRoomRowColAdapter(unittest.TestCase):

    def test_convert_state_to_agent_format_first_room(self):
        adapter = state_adapters.CoordinatesToSingleRoomRowColAdapter(room_size=3)
        mdp_formatted_state = (2, 2)
        expected = [0,0,1,0,0,1]
        actual = adapter.convert_state_to_agent_format(mdp_formatted_state).tolist()
        self.assertEquals(actual, expected)

    def test_convert_state_to_agent_format_fourth_room(self):
        adapter = state_adapters.CoordinatesToSingleRoomRowColAdapter(room_size=3)
        mdp_formatted_state = (4, 4)
        expected = [0,1,0,0,1,0]
        actual = adapter.convert_state_to_agent_format(mdp_formatted_state).tolist()
        self.assertEquals(actual, expected)

    def test_convert_state_to_agent_format_off_diagonal_room(self):
        adapter = state_adapters.CoordinatesToSingleRoomRowColAdapter(room_size=3)
        mdp_formatted_state = (0, 4)
        expected = [1,0,0,0,1,0]
        actual = adapter.convert_state_to_agent_format(mdp_formatted_state).tolist()
        self.assertEquals(actual, expected)

    def test_convert_state_to_agent_format_off_fourth_room_first_square(self):
        adapter = state_adapters.CoordinatesToSingleRoomRowColAdapter(room_size=3)
        mdp_formatted_state = (3, 3)
        expected = [1,0,0,1,0,0]
        actual = adapter.convert_state_to_agent_format(mdp_formatted_state).tolist()
        self.assertEquals(actual, expected)

class TestCoordinatesToRowColAdapter(unittest.TestCase):

    def test_convert_state_to_agent_format_first_room(self):
        adapter = state_adapters.CoordinatesToRowColAdapter(room_size=3, num_rooms=2)
        mdp_formatted_state = (2, 2)
        expected = [0,0,1,0,0,0,0,0,1,0,0,0]
        actual = adapter.convert_state_to_agent_format(mdp_formatted_state).tolist()
        self.assertEquals(actual, expected)

    def test_convert_state_to_agent_format_fourth_room(self):
        adapter = state_adapters.CoordinatesToRowColAdapter(room_size=3, num_rooms=2)
        mdp_formatted_state = (4, 4)
        expected = [0,0,0,0,1,0,0,0,0,0,1,0]
        actual = adapter.convert_state_to_agent_format(mdp_formatted_state).tolist()
        self.assertEquals(actual, expected)

    def test_convert_state_to_agent_format_off_diagonal_room(self):
        adapter = state_adapters.CoordinatesToRowColAdapter(room_size=3, num_rooms=2)
        mdp_formatted_state = (0, 4)
        expected = [1,0,0,0,0,0,0,0,0,0,1,0]
        actual = adapter.convert_state_to_agent_format(mdp_formatted_state).tolist()
        self.assertEquals(actual, expected)

    def test_convert_state_to_agent_format_off_fourth_room_first_square(self):
        adapter = state_adapters.CoordinatesToRowColAdapter(room_size=3, num_rooms=2)
        mdp_formatted_state = (3, 3)
        expected = [0,0,0,1,0,0,0,0,0,1,0,0]
        actual = adapter.convert_state_to_agent_format(mdp_formatted_state).tolist()
        self.assertEquals(actual, expected)

class TestCoordinatesToRowColRoomAdapter(unittest.TestCase):

    def test_convert_state_to_agent_format_first_room(self):
        adapter = state_adapters.CoordinatesToSingleRoomRowColAdapter(room_size=3)
        mdp_formatted_state = (2, 2)
        expected = [0,0,1,0,0,1]
        actual = adapter.convert_state_to_agent_format(mdp_formatted_state).tolist()
        self.assertEquals(actual, expected)

    def test_convert_state_to_agent_format_fourth_room(self):
        adapter = state_adapters.CoordinatesToSingleRoomRowColAdapter(room_size=3)
        mdp_formatted_state = (4, 4)
        expected = [0,1,0,0,1,0]
        actual = adapter.convert_state_to_agent_format(mdp_formatted_state).tolist()
        self.assertEquals(actual, expected)

    def test_convert_state_to_agent_format_off_diagonal_room(self):
        adapter = state_adapters.CoordinatesToSingleRoomRowColAdapter(room_size=3)
        mdp_formatted_state = (0, 4)
        expected = [1,0,0,0,1,0]
        actual = adapter.convert_state_to_agent_format(mdp_formatted_state).tolist()
        self.assertEquals(actual, expected)

    def test_convert_state_to_agent_format_off_fourth_room_first_square(self):
        adapter = state_adapters.CoordinatesToSingleRoomRowColAdapter(room_size=3)
        mdp_formatted_state = (3, 3)
        expected = [1,0,0,1,0,0]
        actual = adapter.convert_state_to_agent_format(mdp_formatted_state).tolist()
        self.assertEquals(actual, expected)




if __name__ == '__main__':
    unittest.main()