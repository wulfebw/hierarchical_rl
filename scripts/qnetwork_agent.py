
import collections
import numpy as np
import random
import sys

from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam

import agent
import logger
import replay_memory

class QNetworkAgent(agent.Agent):

    def __init__(self, num_actions, input_shape, discount, exploration_prob, step_size, 
        frozen_update_period, replay_memory_capacity, batch_size, exploration_prob_reduction_steps, 
        min_exploration_prob, mean_state_values, logging=True):
        self.actions = range(num_actions)
        self.discount = discount
        self.exploration_prob = exploration_prob
        self.exploration_reduction = (exploration_prob - min_exploration_prob) / exploration_prob_reduction_steps
        self.step_size = step_size
        self.num_iters = 1
        self.prev_state = None
        self.prev_action = None
        self.min_exploration_prob = min_exploration_prob
        self.frozen_update_period = frozen_update_period
        self.replay_memory_capacity = replay_memory_capacity
        self.batch_size = batch_size
        self.mean_state_values = mean_state_values
        self.replay_memory = replay_memory.ReplayMemory()
        self.logger = logger.NeuralLogger(agent_name='QNetworkAgent', logging=logging)

        batch_shape = (batch_size,) + input_shape
        self.network = build_fully_connected_network(input_shape=batch_shape, output_units=num_actions)
        self.target_network = build_fully_connected_network(input_shape=batch_shape, output_units=num_actions, target=True)
        
    def step(self, next_state, reward):
        next_state = self.convert_state_to_internal_format(next_state)

        self.num_iters += 1
        self.update_parameters()
        self.update_target_network()
        self.update_replay_memory(self.prev_state, self.prev_action, reward, next_state)
        self.train()

        action = self.get_action(next_state)
        self.prev_state = next_state
        self.prev_action = action

        self.logger.log_reward(reward)
        self.logger.log_action(self.prev_action)

        return action

    def train(self):
        states, actions, rewards, next_states = self.replay_memory.sample_batch(self.batch_size)
        next_q_values = self.target_network.predict(next_states, batch_size=self.batch_size)
        targets = rewards + self.discount * np.max(next_q_values, axis=1)

        y = self.network.predict(states, batch_size=self.batch_size)
        y[np.arange(self.batch_size), actions] = targets
        history = self.network.fit(states, y, batch_size=self.batch_size, nb_epoch=1, verbose=0)

        self.logger.log_loss(history.totals['loss'])

    def get_action(self, state):
        if random.random() < self.exploration_prob:
            action = random.choice(self.actions)
        else:
            state = self.convert_state_to_batch(state)
            q_values = self.network.predict(state, batch_size=1)
            action = np.argmax(q_values)
        return action

    def update_parameters(self):
        # reduce exploration probability
        updated_exploration_prob = self.exploration_prob - self.exploration_reduction
        self.exploration_prob = max(self.min_exploration_prob, updated_exploration_prob)

    def update_target_network(self):
        # update frozen network
        if self.num_iters % self.frozen_update_period == 0:
            self.transfer_weights()

    def update_replay_memory(self, state, action, reward, next_state):
        sars_tuple = (state, action, reward, next_state)
        self.replay_memory.store(sars_tuple)

    def transfer_weights(self):
        num_layers = len(self.network.layers)
        for layer_idx in range(num_layers):
            weights = self.network.layers[layer_idx].get_weights()
            self.target_network.layers[layer_idx].set_weights(weights)

    def start_episode(self, state):
        state = self.convert_state_to_internal_format(state)
        self.prev_state = state
        self.prev_action = self.get_action(state)

        self.logger.log_action(self.prev_action)
        return self.prev_action

    def finish_episode(self, next_state, reward):
        # add a sample to replay memory
        self.logger.finish_episode()

    def finish_epoch(self, epoch):
        self.logger.log_epoch(epoch, self.network)

    def convert_state_to_internal_format(self, state):
        return np.array(state) - self.mean_state_values

    def convert_state_to_batch(self, state):
        return state.reshape(1,-1)

    def get_qvalues(self, state):
        state = self.convert_state_to_internal_format(state)
        state = self.convert_state_to_batch(state)
        q_values = self.network.predict(state, batch_size=1)
        return q_values

def build_fully_connected_network(input_shape, output_units, hidden_layer_size=5,
            learning_rate=1e-3, target=False, weight_init='he_normal', 
            activation='relu'):
    trainable = True
    if target: 
        trainable = False

    # network
    network = Sequential()

    network.add(Dense(hidden_layer_size, init=weight_init, batch_input_shape=input_shape, 
            trainable=trainable))
    #network.add(BatchNormalization())
    network.add(Activation(activation))

    network.add(Dense(hidden_layer_size, init=weight_init, trainable=trainable))
    #network.add(BatchNormalization())
    network.add(Activation(activation))

    network.add(Dense(hidden_layer_size, init=weight_init, trainable=trainable))
    #network.add(BatchNormalization())
    network.add(Activation(activation))

    network.add(Dense(output_units, init=weight_init, trainable=trainable))
    network.add(Activation('linear')) 

    adam = Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    network.compile(loss='mse', optimizer=adam)
    return network
