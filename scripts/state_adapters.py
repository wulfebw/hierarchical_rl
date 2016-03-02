
import numpy as np

class CoordinatesToSingleRoomRowColAdapter(object):

    def __init__(self, room_size):
        self.room_size = room_size

    def convert_state_to_agent_format(self, state):
        """
        Convert states in format (x, y) to a single room, row, column one-hot vector
        example: 
        >>>state = (4, 4)
        >>>adapter = CoordinatesToSingleRoomRowColAdapter(room_size=3)
        >>>adapter.convert_state_to_agent_format(state)
        [1,0,0,1,0,0]
        """
        ridx, cidx = state

        # find where the agent is in the room
        row = np.zeros(self.room_size)
        row[ridx % self.room_size] = 1
        col = np.zeros(self.room_size)
        col[cidx % self.room_size] = 1

        # concat the two vectors
        formatted_state = np.hstack((row, col))

        return formatted_state

class CoordinatesToRowColAdapter(object):

    def __init__(self, room_size, num_rooms):
        self.room_size = room_size
        self.num_rooms = num_rooms

    def convert_state_to_agent_format(self, state):
        """
        Convert states in format (x, y) to a single room, row, column one-hot vector
        example: 
        >>>state = (4, 4)
        >>>adapter = CoordinatesToSingleRoomRowColAdapter(room_size=3, num_rooms=2)
        >>>adapter.convert_state_to_agent_format(state)
        [0,0,0,0,1,0,0,0,0,0,1,0]
        """
        ridx, cidx = state

        # find where the agent is in the room
        row = np.zeros(self.room_size * self.num_rooms)
        row[ridx] = 1
        col = np.zeros(self.room_size * self.num_rooms)
        col[cidx] = 1

        # concat the two vectors
        formatted_state = np.hstack((row, col))

        return formatted_state

class CoordinatesToRowColRoomAdapter(object):

    def __init__(self, room_size, num_rooms):
        self.room_size = room_size
        self.num_rooms = num_rooms

    def convert_state_to_agent_format(self, state):
        """
        Convert states in format (x, y) to a single room, row, column one-hot vector
        _with_ an additional one-hot vector identifying the room
        example: 
        >>>state = (4, 4)
        >>>adapter = CoordinatesToSingleRoomRowColAdapter(room_size=3, num_rooms=2)
        >>>adapter.convert_state_to_agent_format(state)
        [1,0,0,1,0,0,0,0,0,1]
        """
        ridx, cidx = state

        # find where the agent is in the room
        row = np.zeros(self.room_size)
        row[ridx % self.room_size] = 1
        col = np.zeros(self.room_size)
        col[cidx % self.room_size] = 1
        room = np.zeros(self.num_rooms ** 2)
        room_row = cidx / self.room_size
        room_col = ridx / self.room_size
        room_idx = room_row * self.num_rooms + room_col
        room[room_idx] = 1
        # concat the three vectors
        formatted_state = np.hstack((row, col, room))

        return formatted_state

class CoordinatesToFlattenedGridAdapter(object):

    def __init__(self, room_size):
        self.room_size = room_size
        self.num_rooms = num_rooms

    def convert_state_to_agent_format(self, state):
        """
        Convert states in format (x, y) to the full grid
        """
        ridx, cidx = state

        # find where the agent is in the room
        grid = np.zeros((room_size * num_rooms, room_size * num_rooms))
        grid[ridx, cidx] = 1

        # flatten grid
        formatted_state = grid.flatten()

        return formatted_state

class IdentityAdapter(object):

    def convert_state_to_agent_format(self, state):
        """
        Returns the state as is. Exists to keep the interface consistent.
        """
        return state
