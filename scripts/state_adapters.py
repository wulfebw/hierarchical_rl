
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
        [[1,0,0,1,0,0]]
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
