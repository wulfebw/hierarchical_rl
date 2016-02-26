
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

    def finish_episode(self, next_state, reward):
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

    def finish_episode(self, next_state, reward):
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
        self.incorporate_feedback(self.prev_state, self.prev_action, reward, next_state, False)
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

    def incorporate_feedback(self, state, action, reward, next_state, terminal):
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
        if not terminal:
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

    def finish_episode(self, next_state, reward):
        self.incorporate_feedback(self.prev_state, self.prev_action, reward, next_state, True)
        self.logger.finish_episode()

    def finish_epoch(self, epoch):
        self.logger.log_epoch(epoch)

class NeuralAgent(Agent):
    """
    :description: A class that wraps a network so it may more easily interact with an experiment. 
    """

    def __init__(self, network, policy, replay_memory, log, state_adapter):
        """
        :type network: a network class (see e.g., qnetwork.py)
        :param network: the network the agent uses to evaluate states

        :type policy: a policy class (see policy.py)
        :param policy: a class that decides which action to take given the values of those actions

        :type replay_memory: replay memory class (see replay_memory.py)
        :param replay_memory: replay memory used to store dataset as it is gathered.
        """

        self.network = network
        self.policy = policy
        self.replay_memory = replay_memory
        self.logger = log
        self.logger.log_hyperparameters(network, policy, replay_memory)
        self.state_adapter = state_adapter

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
        next_state = self.state_adapter.convert_state_to_agent_format(next_state)

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
        if not self.replay_memory.is_ready_to_sample():
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
        self.prev_state = self.state_adapter.convert_state_to_agent_format(state)
        self.prev_action = self.get_action(self.prev_state)

        self.logger.log_action(self.prev_action)
        return self.prev_action

    def finish_episode(self, next_state, reward):
        """
        :description: perform tasks at the end of episode
        """

        terminal = 1
        next_state = self.state_adapter.convert_state_to_agent_format(next_state)
        self.replay_memory.store((self.prev_state, self.prev_action, reward, next_state, terminal))
        self.logger.log_reward(reward)
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
        state = self.state_adapter.convert_state_to_agent_format(state)
        q_values = self.network.get_q_values(state)
        return q_values

class RecurrentNeuralAgent(Agent):
    """
    :description: A class that wraps a recuurent network so it may more easily 
        interact with an experiment. 
    """
    def __init__(self, network, policy, replay_memory, state_adapter, log):
        self.network = network
        self.policy = policy
        self.replay_memory = replay_memory
        self.logger = log
        self.logger.log_hyperparameters(network, policy, replay_memory)
        self.state_adapter = state_adapter

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
        next_state = self.state_adapter.convert_state_to_agent_format(next_state)

        # store current (s,a,r,s') tuple
        self.replay_memory.store(self.prev_state, self.prev_action, reward, terminal=False)

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
        if not self.replay_memory.is_ready_to_sample():
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
        # wait until agent starts learning to use network to decide action
        if not self.replay_memory.is_ready_to_sample():
            return self.policy.random_action()

        sequence = self.replay_memory.make_last_sequence(state)
        q_values = self.network.get_q_values(sequence)
        return self.policy.choose_action(q_values)

    def start_episode(self, state):
        """
        description: determines the first action to take and initializes internal variables
        """
        self.prev_state = self.state_adapter.convert_state_to_agent_format(state)
        self.prev_action = self.get_action(self.prev_state)

        self.logger.log_action(self.prev_action)
        return self.prev_action

    def finish_episode(self, next_state, reward):
        """
        :description: perform tasks at the end of episode. We don't store the next_state value
            because the previous state must have been a terminal one. It's in the method
            definition to stay consistent with the other replay memory implementation.
        """
        self.replay_memory.store(self.prev_state, self.prev_action, reward, True)
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
        state = self.state_adapter.convert_state_to_agent_format(state)
        q_values = self.network.get_q_values(state)
        return q_values
        
