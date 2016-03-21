
import collections
import copy
import matplotlib.pyplot as plt
from multiprocessing.pool import ThreadPool
import numpy as np
import random
import sys
import time

import learning_utils

# threading constants
NUM_THREADS = 2

# global variables for AsyncSarsa
WEIGHTS = collections.defaultdict(lambda: 0)

# global variables for AsyncAdvantageActorCritic
WEIGHTS = collections.defaultdict(lambda: 0)
VALUE_WEIGHTS = collections.defaultdict(lambda: 0)

# logging
REWARDS = []
START_STATE_VALUES = []

class MazeMDP(object):
   
    EXIT_REWARD = 1
    MOVE_REWARD = -.01
    ACTIONS = [(1,0),(-1,0),(0,1),(0,-1)] 
    DISCOUNT = 1
    START_STATE = (0,0)

    def __init__(self, room_size, num_rooms):
        self.room_size = room_size
        self.num_rooms = num_rooms
        self.max_position = self.room_size * self.num_rooms - 1
        self.end_state = (self.max_position, self.max_position) 
        self.computeStates() 

    def calculate_next_state(self, state, action):
        return state[0] + action[0], state[1] + action[1]

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

    def succAndProbReward(self, state, action): 

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

        return [(next_state, 1, reward)]

    def computeStates(self):
        self.states = set()
        queue = []
        self.states.add(self.START_STATE)
        queue.append(self.START_STATE)
        while len(queue) > 0:
            state = queue.pop()
            for action in self.ACTIONS:
                for newState, prob, reward in self.succAndProbReward(state, action):
                    if newState not in self.states:
                        self.states.add(newState)
                        queue.append(newState)

    def print_state_values(self):
        V = {}
        for state in self.states:
            #state_value = max(WEIGHTS[(state, action)] for action in self.ACTIONS)
            state_value = VALUE_WEIGHTS[(state, None)]
            V[state] = state_value

        for ridx in reversed(range(self.max_position + 1)):
            for cidx in range(self.max_position + 1):
                if (ridx, cidx) in V:
                    print '{0:.5f}'.format(V[(ridx, cidx)]),
            print('\n')

class Experiment(object):

    def __init__(self, mdp, agent, num_episodes, max_steps):
        self.mdp = mdp
        self.agent = agent
        self.num_episodes = num_episodes
        self.max_steps = max_steps

    def run(self, agent_id):
        print 'running experiment with agent number {}...'.format(agent_id)

        total_rewards = []
        total_reward = 0

        for episode in range(self.num_episodes):
            if episode % 100 == 0:
                print 'running episode {} for agent {}...'.format(episode, agent_id)
            state = self.mdp.START_STATE
            action = self.agent.get_action(state)

            for step in range(self.max_steps):
                transitions = self.mdp.succAndProbReward(state, action)

                if len(transitions) == 0:
                    reward = 0
                    new_state = None
                    break

                new_state, prob, reward = transitions[0]
                total_reward += reward
                action = self.agent.incorporateFeedback(state, action, reward, new_state)
                state = new_state

            self.agent.incorporateFeedback(state, action, reward, new_state)
            total_rewards.append(total_reward)
            REWARDS.append(total_reward)
            #START_STATE_VALUES.append(max(WEIGHTS[((0,0), action)] for action in self.mdp.ACTIONS))
            START_STATE_VALUES.append(VALUE_WEIGHTS[((0,0), None)])
            total_reward = 0
        
        print 'average reward of agent {}: {}'.format(agent_id, np.mean(total_rewards))

class MultithreadedExperiment(object):

    def __init__(self, experiment, num_agents):
        self.experiment = experiment
        self.num_agents = num_agents

    def run(self):
        pool = ThreadPool(self.num_agents)
        for idx in range(self.num_agents):
            pool.apply_async(self.run_experiement, args=(self.experiment, idx))

        pool.close()
        pool.join()

    @staticmethod
    def run_experiement(experiment, agent_id):
        print 'starting experiment with agent number {}...'.format(agent_id)
        experiment_copy = copy.deepcopy(experiment)
        experiment.run(agent_id)

class AsyncSarsa(object):

    def __init__(self, actions, discount, exploration_prob, learning_rate):
        self.actions = actions
        self.discount = discount
        self.exploration_prob = exploration_prob
        self.learning_rate = learning_rate
        self.num_iters = 0

    def feature_extractor(self, state, action):
        return [((state, action), 1)]

    def getQ(self, state, action):
        score = 0
        for f, v in self.feature_extractor(state, action):
            score += WEIGHTS[f] * v
        return score

    def get_action(self, state):
        self.num_iters += 1

        if self.exploration_prob > .05:
            self.exploration_prob -= 1e-8

        if random.random() < self.exploration_prob:
            action = random.choice(self.actions)
        else:
            action = max((self.getQ(state, action), action) for action in self.actions)[1]
        return action

    def incorporateFeedback(self, state, action, reward, new_state):
        prediction = self.getQ(state, action)
        target = reward
        new_action = None

        if new_state != None:
            new_action = self.get_action(new_state)
            target += self.discount * self.getQ(new_state, new_action)

        for f, v in self.feature_extractor(state, action):
            WEIGHTS[f] = WEIGHTS[f] + self.learning_rate * (target - prediction) * v

        return new_action

class AsyncAdvantageActorCritic(object):

    def __init__(self, actions, discount, tau, learning_rate):
        self.actions = actions
        self.discount = discount
        self.tau = tau
        self.learning_rate = learning_rate
        self.num_iters = 0

    def feature_extractor(self, state, action=None):
        return [((state, action), 1)]

    def getV(self, state):
        score = 0
        for f, v in self.feature_extractor(state):
            score += VALUE_WEIGHTS[f] * v
        return score

    def getQ(self, state, action):
        score = 0
        for f, v in self.feature_extractor(state, action):
            score += WEIGHTS[f] * v
        return score

    def get_action(self, state):
        self.num_iters += 1
        # if self.tau > 1e-9:
        #     self.tau *= .9999
        #     print self.tau

        q_values = np.array([self.getQ(state, action) for action in self.actions])
        exp_q_values = np.exp(q_values / (self.tau + 1e-2))
        weights = dict()
        for idx, val in enumerate(exp_q_values):
            weights[idx] = val
        action_idx = learning_utils.weightedRandomChoice(weights)
        action = self.actions[action_idx]
        return action

    def incorporateFeedback(self, state, action, reward, new_state):
        prediction = self.getV(state)
        target = reward
        new_action = None

        if new_state != None:
            new_action = self.get_action(new_state)
            target += self.discount * self.getV(new_state)

        update = self.learning_rate * (target - prediction)
        for f, v in self.feature_extractor(state):
            VALUE_WEIGHTS[f] = VALUE_WEIGHTS[f] + 2 * update

        for f, v in self.feature_extractor(state, action):
            WEIGHTS[f] = WEIGHTS[f] + update * 1

        return new_action

def plot_values(values, ylabel):
    values = np.mean(np.reshape(values, (-1, 4)), axis=1).reshape(-1)
    plt.scatter(range(len(values)), values)
    plt.xlabel('episodes (1 per actor-learner)')
    plt.ylabel(ylabel)
    plt.show()

def run():
    start = time.time()
    room_size = 5
    num_rooms = 2
    mdp = MazeMDP(room_size=room_size, num_rooms=num_rooms)
    # agent = AsyncSarsa(actions=mdp.ACTIONS, discount=mdp.DISCOUNT, 
    #             exploration_prob=0.3, learning_rate=.5)
    agent = AsyncAdvantageActorCritic(actions=mdp.ACTIONS, discount=mdp.DISCOUNT, 
                tau=.3, learning_rate=.5)
    max_steps = (2 * room_size * num_rooms) ** 2
    experiment = Experiment(mdp=mdp, agent=agent, num_episodes=800, max_steps=max_steps)
    multiexperiment = MultithreadedExperiment(experiment=experiment, num_agents=NUM_THREADS)
    multiexperiment.run()
    end = time.time()
    print 'took {} seconds'.format(end - start)
    mdp.print_state_values()
    plot_values(REWARDS, 'rewards')
    plot_values(START_STATE_VALUES, 'start state value')



if __name__ =='__main__':
    run()