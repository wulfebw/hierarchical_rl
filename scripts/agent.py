
import collections
import numpy as np
import random

import logger

class Agent(object):

    def step(self, next_state, reward):
        """
        :description: this method implements the agents deciding which action to take and updating its parameters
        """
        raise NotImplementedError("Override me")

    def start_episode(self, state):
        """
        :description: initializes an agent for an episode and returns an initial action to take
        """
        raise NotImplementedError("Override me")

    def finish_episode(self):
        """
        :description: finalizes an episode for an agent
        """

    def finish_epoch(self, epoch):
        """
        :description: performs logging tasks at the end of an epoch
        """
        raise NotImplementedError("Override me")

    def start_testing(self):
        pass

    def finish_testing(self):
        pass


class TestAgent(Agent):

    def __init__(self, num_actions):
        self.actions = range(num_actions)
        self.steps = 0
        self.episodes = 0

    def step(self, next_state, reward):
        self.steps += 1
        return random.choice(self.actions)

    def start_episode(self, state):
        self.episodes += 1
        return random.choice(self.actions)

    def finish_episode(self):
        pass

    def finish_epoch(self, epoch):
        pass
        

class QLearningAgent(Agent):

    def __init__(self, num_actions, discount, exploration_prob, step_size, logging=True):
        self.actions = range(num_actions)
        self.discount = discount
        self.exploration_prob = exploration_prob
        self.step_size = step_size
        self.num_iters = 1
        self.weights = collections.Counter()
        self.logger = logger.Logger(agent_name='QLearningAgent', logging=logging)
        self.prev_state = None
        self.prev_action = None

    def step(self, next_state, reward):
        self.incorporate_feedback(self.prev_state, self.prev_action, reward, next_state)
        action = self.get_action(next_state)
        self.prev_state = next_state
        self.prev_action = action

        self.logger.log_action(action)
        self.logger.log_reward(reward)
        return action

    def feature_extractor(self, state, action):
        """
        :description: this is the identity feature extractor, so we use tables here for the function
        """
        return [((state, action), 1)]

    def getQ(self, state, action):
        """
        :description: returns the Q value associated with this state-action pair

        :type state: numpy array
        :param state: the state of the game

        :type action: int
        :param action: the action for which to retrieve the Q-value
        """
        score = 0
        for f, v in self.feature_extractor(state, action):
            score += self.weights[f] * v
        return score

    def get_action(self, state):
        """
        :description: returns an action accoridng to epsilon-greedy policy

        :type state: dictionary
        :param state: the state of the game
        """
        self.num_iters += 1

        if random.random() < self.exploration_prob:
            return random.choice(self.actions)
        else:
            max_action = max((self.getQ(state, action), action) for action in self.actions)[1]
        return max_action

    def incorporate_feedback(self, state, action, reward, next_state):
        """
        :description: performs a Q-learning update

        :type reward: float
        :param reward: reward associated with transitioning to next_state

        :type next_state: numpy array
        :param next_state: the new state of the game
        """
        step_size = self.step_size
        prediction = self.getQ(state, action)
        target = reward
        if next_state != None:
            target += self.discount * max(self.getQ(next_state, next_action) for next_action in self.actions)

        diff = target - prediction
        loss = .5 * diff ** 2
        for f, v in self.feature_extractor(state, action):
            self.weights[f] = self.weights[f] + step_size * diff * v

        self.logger.log_loss(loss)
        self.logger.log_weights(self.weights)

    def start_episode(self, state):
        self.prev_state = state
        self.prev_action = self.get_action(state)

        self.logger.log_action(self.prev_action)
        return self.prev_action

    def finish_episode(self):
        self.incorporate_feedback(self.prev_state, self.prev_action, 0, None)
        self.logger.finish_episode()

    def finish_epoch(self, epoch):
        self.logger.log_epoch(epoch)


class NeuralAgent(Agent):
    """
    :description: A class that wraps a network so it may more easily interact with an experiment. 
    """
    def __init__(self, network, policy, replay_memory, mean_state_values, logging=False):
        self.network = network
        self.policy = policy
        self.replay_memory = replay_memory
        self.mean_state_values = mean_state_values
        self.logger = logger.NeuralLogger(agent_name='NeuralAgent', logging=logging)
        self.logger.log_hyperparameters(network, policy, replay_memory)

        self.prev_state = None
        self.prev_action = None
        
    def step(self, next_state, reward):
        """
        :description: the primary method of this class, which 'steps' the agent and network forward one time step. This includes selecting an action, making use of the new state and reward, and performing training.

        :type next_state: tuple or array
        :param next_state: the next state observed (i.e., s')

        :type reward: int 
        :param reward: the reward associated with having moved from the previous state to the current state

        :type rval: int
        :param rval: returns the action to next be taken within the environment
        """
        # need to transform an external state format to an internal one
        next_state = self.convert_state_to_internal_format(next_state)

        # store current (s,a,r,s') tuple
        self.replay_memory.store((self.prev_state, self.prev_action, reward, next_state, 0))

        # perform training
        self.train()

        # retrieve an action
        action = self.get_action(next_state)

        # set previous values
        self.prev_state = next_state
        self.prev_action = action

        # log information
        self.logger.log_reward(reward)
        self.logger.log_action(self.prev_action)

        return action

    def train(self):
        """
        :description: collects a minibatch of experiences and passes them to the network to train
        """
        # wait until replay memory has samples
        if self.replay_memory.isEmpty():
            return

        # collect minibatch
        states, actions, rewards, next_states, terminals = self.replay_memory.sample_batch()

        # pass to network to perform training
        loss = self.network.train(states, actions, rewards, next_states, terminals)
        self.logger.log_loss(loss)

    def get_action(self, state):
        """
        :description: gets an action given the current state. Defers to the network for selecting the action.

        :type state: numpy array
        :param state: the state used to determine the action
        """
        q_values = self.network.get_q_values(state)
        return self.policy.choose_action(q_values)

    def start_episode(self, state):
        """
        description: determines the first action to take and initializes internal variables
        """
        state = self.convert_state_to_internal_format(state)
        self.prev_state = state
        self.prev_action = self.get_action(state)

        self.logger.log_action(self.prev_action)
        return self.prev_action

    def finish_episode(self):
        """
        :description: perform tasks at the end of episode
        """
        # This is a terminal next_state, so set to None and then when passing to 
        # network set terminal to 1 so that the network output for this next_state will 
        # not be considered
        reward = 0
        next_state = np.zeros(self.prev_state.shape)
        terminal = 1
        self.replay_memory.store((self.prev_state, self.prev_action, reward, next_state, terminal))
        self.logger.finish_episode()

    def finish_epoch(self, epoch):
        """
        :description: perform tasks at the end of an epoch
        """
        self.logger.log_epoch(epoch, self.network, self.policy)

    def get_q_values(self, state):
        """
        :description: returns the q values associated with a given state. Used for printing out a representation of the mdp with the values included. 
        """
        state = self.convert_state_to_internal_format(state)
        q_values = self.network.get_q_values(state)
        return q_values

    def convert_state_to_internal_format(self, state):
        """
        :description: converts a state from an extenarl format to an internal one
        """
        # fc
        formatted_state = np.zeros((9,9))
        formatted_state[state[0], state[1]] = 1
        formatted_state = formatted_state.flatten()

        # conv
        # formatted_state = np.zeros((1,5,5))
        # formatted_state[0, state[0], state[1]] = 1

        return formatted_state
