import os
import sys
import random 
from collections import defaultdict 
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt

from keras.models import Model
from keras.layers import Input, Conv1D, Dense, BatchNormalization, Flatten, concatenate, Lambda
import keras.backend as K
from keras.losses import binary_crossentropy
from keras.layers import (
    Input,
    Embedding,
    Dropout,
    Conv1D,
    Lambda,
    GlobalAveragePooling1D,
    GlobalMaxPooling1D,
    Concatenate,
    SpatialDropout1D,
    Dense,
    Activation,
    BatchNormalization,
    Concatenate,
    Reshape,
    Multiply,
    Add,
    Dot,
    Flatten,
    Lambda,
    LSTM,
    GRU,
    Bidirectional
)
from keras.layers.wrappers import TimeDistributed
from keras.regularizers import l1, l2

from explorers.base_explorer import Base_explorer 
from utils.sequence_utils import *
from utils.replay_buffers import PrioritizedReplayBuffer 

class BO_Explorer(Base_explorer):
    def __init__(self, batch_size=100, alphabet='UCGA',
                 virtual_screen=10, path="./simulations/", debug=False, 
                 seq_len=40, method = 'EI', train_epochs=10):
        super(BO_Explorer, self).__init__(batch_size=batch_size, alphabet=alphabet, virtual_screen=virtual_screen, path=path, debug=debug)
        self.explorer_type='BO_Ensemble'
        self.alphabet_len = len(alphabet)
        self.seq_len = seq_len
        self.method = method
        self.batch_size = batch_size 
        self.train_epochs = train_epochs
        self.virtual_screen = virtual_screen
        self.best_fitness = 0
        self.top_sequence = []
        # use PER buffer, same as in DQN 
        self.ensemble = self.create_ensemble()
        self.total_actions = 0

    def create_model(self, kernel_initializer, kernel_regularizer, bias_initializer):
        input_layer = Input(shape=(self.alphabet_len*self.seq_len,))
        x = Dense(self.seq_len, activation='relu',
                    kernel_initializer=kernel_initializer, 
                    kernel_regularizer=kernel_regularizer,
                    bias_initializer=bias_initializer)(input_layer)
        x = BatchNormalization()(x)
        x = Dense(self.alphabet_len, activation='relu',
                    kernel_initializer=kernel_initializer, 
                    kernel_regularizer=kernel_regularizer,
                    bias_initializer=bias_initializer)(x)
        x = BatchNormalization()(x)
        output_layer = Dense(1, activation='relu')(x)
        model = Model(inputs=input_layer, outputs=output_layer)
        model.compile(loss='mse', optimizer='adam')

        return model 
        
    def create_ensemble(self):
        ensemble = [
            self.create_model('glorot_uniform', None, 'zeros'),
            self.create_model('lecun_uniform', l2(0.01), 'zeros'),
            self.create_model('random_uniform', l2(0.01), 'zeros'),
            self.create_model('lecun_uniform', l1(0.01), 'zeros'),
            self.create_model('random_uniform', l1(0.01), 'zeros'),
        ]
        return ensemble 
    
    def train_models(self):
        for i in range(len(self.ensemble)):
            for epoch in range(self.train_epochs):
                batch = self.memory.sample_batch()
                rewards, actions, states, next_states = \
                batch['rews'], batch['acts'], batch['obs'], batch['next_obs']
                self.ensemble[i].fit(next_states, rewards, epochs=1, verbose=0)
                
    def EI(self, vals):
        return np.mean([max(val - self.best_fitness, 0) for val in vals])
        
    def UCB(self, vals):
        discount = 0.01 
        return np.mean(vals) - discount*np.std(vals)

    def initialize_data_structures(self):
        start_sequence = list(self.model.measured_sequences)[0]
        self.state = translate_string_to_one_hot(start_sequence, self.alphabet)
        self.seq_len = len(start_sequence)
        self.memory = PrioritizedReplayBuffer(self.alphabet_len*self.seq_len, 
                                              100000, self.batch_size, 0.6) 
                   
    def pick_action(self):
        if self.total_actions == 0:
            # if model/start sequence got reset
            self.initialize_data_structures()
        state = self.state.copy()
        possible_actions = [(i, j) for i in range(self.alphabet_len) 
                            for j in range(self.seq_len) if state[i, j] == 0]
        action_inds = np.random.choice(len(possible_actions), 
                                   size=self.virtual_screen, 
                                   replace=False)
        actions = [possible_actions[ind] for ind in action_inds]
        actions_to_screen = []
        states_to_screen = []
        for i in range(self.virtual_screen):
            x = np.zeros((self.alphabet_len, self.seq_len))
            x[actions[i]] = 1 
            actions_to_screen.append(x)
            state_to_screen = construct_mutant_from_sample(x, state)
            states_to_screen.append(state_to_screen)
        states_to_screen = np.array(states_to_screen).reshape((len(states_to_screen), -1))
        ensemble_preds = [model.predict(states_to_screen, verbose=0).ravel() 
                          for model in self.ensemble]
        ensemble_preds = list(zip(*ensemble_preds))
        method_pred = [self.EI(vals) for vals in ensemble_preds] if self.method == 'EI' \
            else [self.UCB(vals) for vals in ensemble_preds]
        action_ind = np.argmax(method_pred)
        action = actions_to_screen[action_ind]
        new_state = states_to_screen[action_ind].reshape((self.alphabet_len, self.seq_len))
        self.state = new_state
        new_state_string = translate_one_hot_to_string(new_state, self.alphabet)
        reward = self.model.get_fitness(new_state_string)
        if not new_state_string in self.model.measured_sequences:
            if reward >= self.best_fitness:
                self.top_sequence.append((reward, new_state, self.model.cost))
            self.best_fitness = max(self.best_fitness, reward)
            self.memory.store(state.ravel(), action.ravel(), reward, new_state.ravel())
        if self.model.cost % self.batch_size == 0 and self.model.cost > 0:
            self.train_models()
        self.total_actions += 1
            
    def propose_samples(self):
        samples = []
        for _ in range(self.batch_size):
            self.pick_action()
            samples.append(translate_one_hot_to_string(self.state,self.alphabet))
        return samples 