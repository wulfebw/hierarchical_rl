
import collections
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

    def finish_episode(self, next_state, reward):
        self.incorporate_feedback(self.prev_state, self.prev_action, 0, None)
        self.logger.finish_episode()

    def finish_epoch(self, epoch):
        self.logger.log_epoch(epoch)

