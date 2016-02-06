
import collections
import copy
import numpy as np
import random
import sys

###########################################################################

class MDP(object):

    def get_start_state(self): 
        raise NotImplementedError("Override me")

    def get_actions(self): 
        raise NotImplementedError("Override me")

    def succ_prob_reward(self, state, action): 
        """
        :description: returns a _list_ of tuples containing (next_state, probability, reward). Where the probability denotes the probability of the next_state and reward.
        """
        raise NotImplementedError("Override me")

    def get_discount(self): 
        raise NotImplementedError("Override me")

    def compute_states(self):
        self.states = set()
        queue = []
        self.states.add(self.get_start_state())
        queue.append(self.get_start_state())
        while len(queue) > 0:
            state = queue.pop()
            for action in self.get_actions(state):
                for newState, prob, reward in self.succ_prob_reward(state, action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)

###########################################################################

class LineMDP(MDP):
    """
    :description: A line mdp is just an x axis. Here the rewards are all -1 except for the last state on the right which is +1.
    """

    EXIT_REWARD = 1
    MOVE_REWARD = -.01

    def __init__(self, length):
        self.length = length

    def get_start_state(self):
        return 0

    def get_actions(self, state=None):
        return [-1, 1]

    def get_discount(self): 
        return 1

    def succ_prob_reward(self, state, action): 
        if state == self.length:
            return []

        next_state = max(-self.length, state + action)
        reward = 1 if next_state == self.length else -1
        return [(next_state, 1, reward)]

    def print_v(self, V):
        line = ['-'] * (self.length * 2)
        for vidx, lidx in zip(range(-self.length, self.length), range(self.length * 2)):
            if vidx in V:
                line[lidx] = round(V[vidx], 2)
        print line

    def print_pi(self, pi):
        line = ['-'] * (self.length * 2)
        for pidx, lidx in zip(range(-self.length, self.length), range(self.length * 2)):
            if pidx in pi:
                line[lidx] = round(pi[pidx], 2)
        print line

###########################################################################

class MazeMDP(MDP):
    """
    :description: an MDP specifying a maze, where that maze is a square, consists of num_rooms and each room having room_size discrete squares in it. So can have 1x1, 2x2, 3x3, etc size mazes. Rooms are separated by walls with a single entrance between them. The start state is always the bottom left of the maze, 1 position away from each wall of the first room. The end state is always in the top right room of the maze, again 1 position away from each wall. So the 1x1 maze looks like:

         _______
        |       |
        |     E |
        | S     |
        |       |
         -------

         the 2x2 maze would be

         _______ _______   
        |       |       |
        |             E |
        |               |
        |       |       |
         --   -- --   --
         __   __ __   __
        |       |       |
        |               |
        | S             |
        |       |       |
         ------- -------


         state is represented in absolute terms, so the bottom left corner of all mazes is (0,0) and to top right corner of all mazes is (room_size * num_rooms - 1, room_size * num_rooms - 1). In other words, the state ignores the fact that there are rooms or walls or anything, it's just the coordinates.

         actions are N,E,S,W movement by 1 direction. No stochasticity for now. moving into a wall leaves agent in place. Rewards are nothing except finding the exit is worth a lot 

         room_size must be odd
    """

    EXIT_REWARD = 1
    MOVE_REWARD = -0.1

    def __init__(self, room_size, num_rooms):
        self.room_size = room_size
        self.num_rooms = num_rooms
        self.max_position = self.room_size * self.num_rooms - 1
        self.end_state = (self.max_position - 1, self.max_position - 1)

    def get_default_action(self):
        return (1,0)

    def get_actions(self, state=None):
        return [(1,0),(-1,0),(0,1),(0,-1)]

    def get_start_state(self):
        return (1,1)   

    def get_discount(self):
        return 1

    def get_mean_state_values(self):
        return np.repeat(self.max_position / 2., 2)

    def calculate_next_state(self, state, action):
        next_state = (state[0] + action[0], state[1] + action[1])
        return next_state

    def runs_into_wall(self, state, action):
        next_state = self.calculate_next_state(state, action)

        # 1. check for leaving the maze
        if next_state[0] > self.max_position or next_state[0] < 0 \
                            or next_state[1] > self.max_position or next_state[1] < 0:
            return True

        # 2. check if movement was through doorway and if so return false
        doorway_position = (self.room_size) / 2
        # check horizontal movement through doorway
        if next_state[0] != state[0]:
            if next_state[1] % self.room_size == doorway_position:
                return False

        # check vertical movement through doorway
        if next_state[1] != state[1]:
            if next_state[0] % self.room_size == doorway_position:
                return False

        # 3. check if movement was through a wall
        room_size = self.room_size
        # move right to left through wall
        if state[0] % room_size == room_size - 1 and next_state[0] % room_size == 0:
            return True

        # move left to right through wall
        if next_state[0] % room_size == room_size - 1 and state[0] % room_size == 0:
            return True

        # move up through wall
        if state[1] % room_size == room_size - 1 and next_state[1] % room_size == 0:
            return True

        # move down through wall
        if next_state[1] % room_size == room_size - 1 and state[1] % room_size == 0:
            return True

        # if none of the above conditions meet, then have not passed through wall
        return False

    def succ_prob_reward(self, state, action): 

        # if we reach the end state then the episode ends
        if np.array_equal(state, self.end_state):
            return []

        if self.runs_into_wall(state, action):
            # if the action runs us into a wall do nothing
            next_state = state
        else:
            # o/w determine the next position
            next_state = self.calculate_next_state(state, action)

        # if next state is exit, then set reward
        reward = self.MOVE_REWARD
        if np.array_equal(next_state, self.end_state):
            reward = self.EXIT_REWARD

        # print 'state: {}'.format(state)
        # print 'action: {}'.format(action)
        # print 'next_state: {}'.format(next_state)
        # raw_input()

        return [(next_state, 1, reward)]

    def normalize_state(self, state):
        mean_subtracted_vars = []
        for var in state:
            mean_subtracted_vars.append(var - float(self.max_position) / 2)

        return tuple(mean_subtracted_vars)

    def print_v(self, V):
        for ridx in reversed(range(self.max_position + 1)):
            for cidx in range(self.max_position + 1):
                if (ridx, cidx) in V:
                    print round(V[(ridx, cidx)], 1),
            print('\n')

    def get_value_string(self, V):
        value_string = []
        for ridx in reversed(range(self.max_position + 1)):
            for cidx in range(self.max_position + 1):
                if (ridx, cidx) in V:
                    if (ridx, cidx) == self.get_start_state():
                        value_string.append('S ')
                    elif (ridx, cidx) == self.end_state:
                        value_string.append('E ')
                    else:
                        value_string.append(round(V[(ridx, cidx)], 1))
                    value_string.append(' ')
            value_string.append('\n')
        return ''.join([str(v) for v in value_string])

    def print_maze(self, coordinates):
        for row in range(self.room_size):
            for col in range(self.room_size):
                if coordinates == (row,col):
                    print '*',
                elif self.end_state == (row,col):
                    print 'e',
                else:
                    print '-',
            print '\n'
        print '\n'

    def print_trajectory(self, actions):
        coordinates = self.start_state
        self.print_maze(coordinates)
        for action in actions:
            coordinates = (coordinates[0] + action[0], coordinates[1] + action[1])
            self.print_maze(coordinates)

###########################################################################

